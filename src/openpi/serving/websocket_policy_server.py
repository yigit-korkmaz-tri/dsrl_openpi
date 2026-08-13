import asyncio
import http
import logging
import time
import traceback

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    def _dispatch(self, method: str, obs: dict):
        if method == "infer":
            # The client folds `noise` into the observation dict (there is nowhere else to put it
            # on the wire), but Policy.infer takes it as a keyword-only argument.
            noise = obs.pop("noise", None) if isinstance(obs, dict) else None
            if noise is None:
                return self._policy.infer(obs)
            return self._policy.infer(obs, noise=noise)
        if method == "get_prefix_rep":
            get_prefix_rep = getattr(self._policy, "get_prefix_rep", None)
            if get_prefix_rep is None:
                raise ValueError(f"{type(self._policy).__name__} does not implement get_prefix_rep.")
            return get_prefix_rep(obs)
        raise ValueError(f"Unknown method {method!r}; expected 'infer' or 'get_prefix_rep'.")

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                method, obs = _unpack_request(obs)

                infer_time = time.monotonic()
                action = self._dispatch(method, obs)
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _unpack_request(request) -> tuple[str, dict]:
    """Accept both wire formats and return `(method, obs)`.

    This repo's `openpi_client.WebsocketClientPolicy` wraps every message in an envelope,
    `{"method": ..., "obs": ...}`, while upstream openpi's client sends the bare observation dict.
    The server used to hand whatever arrived straight to `policy.infer`, so this repo's own client
    could not talk to this repo's own server: the transform saw `{"method", "obs"}` and died with
    `KeyError: 'observation/image_head'`. Deployments worked around it by importing the client from
    an upstream checkout instead -- which in turn made anything added to this client (such as the
    real-time chunking broker) unreachable on the robot. Supporting both formats here fixes the
    skew without breaking either client.
    """
    if isinstance(request, dict) and "obs" in request and set(request) <= {"method", "obs"}:
        return str(request.get("method", "infer")), request["obs"]
    return "infer", request


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
