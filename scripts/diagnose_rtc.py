"""Probe a running openpi server's real-time chunking behavior, without a robot.

Answers the question "are the RTC actions wrong, or is the client mis-timing them?" by asking the
live server for the same two chunks the broker would ask for -- one unguided, one conditioned on
the first -- and printing them in ROBOT action space, next to each other.

    uv run --no-sync python scripts/diagnose_rtc.py --host localhost --port 8000 \
        --prompt "pick up the banana" --execute-horizon 4 --inference-delay 2

What to look at:

  * "per-step motion" -- the mean |a[i+1] - a[i]| over the chunk, in robot units. If the guided
    number is far below the unguided one, guidance is over-constraining and the arm will creep or
    stall. That is a tuning problem: lower --max-guidance-weight, or raise --execute-horizon so
    the prefix attention horizon (H - execute_horizon) covers less of what you execute.
  * "executed rows" -- the actions the broker would actually send, indices [delay, delay+execute).
    Compare them against the unguided chunk's same rows.
  * "inference time" -- guidance costs a VJP per denoising step. If guided latency exceeds
    inference_delay control steps at your --action-hz, chunks land past their pinned prefix.
"""

import argparse
import sys
import time

import numpy as np

sys.path.insert(0, "packages/openpi-client/src")

from openpi_client import action_chunk_broker as acb
from openpi_client import image_tools
from openpi_client import websocket_client_policy

_CAM_KEYS = ("observation/image_head", "observation/image_left_wrist", "observation/image_right_wrist")


def _observation(args) -> dict:
    """A syntactically valid observation. Images are synthetic -- we are comparing guided against
    unguided on the SAME input, so the absolute actions need not be task-meaningful."""
    rng = np.random.default_rng(0)
    obs = {
        key: image_tools.resize_with_pad(rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8), 224, 224)
        for key in _CAM_KEYS
    }
    obs["observation/state"] = np.asarray(args.state, dtype=np.float32)
    if args.prompt:
        obs["prompt"] = args.prompt
    return obs


def _stats(name: str, chunk: np.ndarray, dof: int) -> None:
    motion = np.abs(np.diff(chunk[:, :dof], axis=0)).mean()
    print(f"    {name:<10} |a|mean={np.abs(chunk[:, :dof]).mean():8.4f}   per-step motion={motion:8.5f}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--prompt", default="")
    p.add_argument("--execute-horizon", type=int, default=4)
    p.add_argument("--inference-delay", type=int, default=2)
    p.add_argument("--schedule", default="exp", choices=["linear", "exp", "ones", "zeros"])
    p.add_argument("--max-guidance-weight", type=float, default=5.0)
    p.add_argument("--action-hz", type=float, default=15.0)
    p.add_argument("--dof", type=int, default=14)
    p.add_argument("--state", type=float, nargs="*", default=[0.0] * 14)
    args = p.parse_args()

    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    print(f"connected: {client.get_server_metadata()}\n")
    obs = _observation(args)

    # Warm up BOTH jit signatures first. Guided sampling has a different input signature (the
    # previous chunk goes from absent to present), so its first call pays a full XLA compile --
    # tens of seconds. Timing that instead of steady-state latency would be meaningless, and on
    # the robot it is also why the very first replan can stall.
    print("warming up (compiling both sampler variants; the first guided call is slow) ...")
    warm = client.infer({**obs, acb.RTC_ENABLED: True})
    if acb.RTC_RAW_ACTIONS in warm:
        warm_raw = np.asarray(warm[acb.RTC_RAW_ACTIONS])
        client.infer(
            {
                **obs,
                acb.RTC_ENABLED: True,
                acb.RTC_PREV_CHUNK: acb._shift_chunk(warm_raw, args.execute_horizon),  # noqa: SLF001
                acb.RTC_INFERENCE_DELAY: args.inference_delay,
                acb.RTC_PREFIX_ATTENTION_HORIZON: max(0, warm_raw.shape[0] - args.execute_horizon),
                acb.RTC_PREFIX_ATTENTION_SCHEDULE: args.schedule,
                acb.RTC_MAX_GUIDANCE_WEIGHT: args.max_guidance_weight,
            }
        )

    t = time.monotonic()
    first = client.infer({**obs, acb.RTC_ENABLED: True})
    t_unguided = time.monotonic() - t

    if acb.RTC_RAW_ACTIONS not in first:
        print(f"FAIL: server did not echo '{acb.RTC_RAW_ACTIONS}'. It predates RTC support -- restart it.")
        return 1

    raw = np.asarray(first[acb.RTC_RAW_ACTIONS])
    unguided = np.asarray(first["actions"])
    horizon = unguided.shape[0]
    print(f"action horizon = {horizon}   robot action dim = {unguided.shape[1]}   model space = {raw.shape}")

    budget = args.inference_delay / args.action_hz
    print(f"\ninference time: unguided {t_unguided * 1000:.0f} ms")

    shift = args.execute_horizon
    prev = acb._shift_chunk(raw, shift)  # noqa: SLF001
    prefix_horizon = max(0, horizon - shift)

    t = time.monotonic()
    second = client.infer(
        {
            **obs,
            acb.RTC_ENABLED: True,
            acb.RTC_PREV_CHUNK: prev,
            acb.RTC_INFERENCE_DELAY: args.inference_delay,
            acb.RTC_PREFIX_ATTENTION_HORIZON: prefix_horizon,
            acb.RTC_PREFIX_ATTENTION_SCHEDULE: args.schedule,
            acb.RTC_MAX_GUIDANCE_WEIGHT: args.max_guidance_weight,
        }
    )
    t_guided = time.monotonic() - t
    guided = np.asarray(second["actions"])

    print(f"                guided   {t_guided * 1000:.0f} ms  ({t_guided / max(t_unguided, 1e-9):.2f}x)")
    print(f"                budget   {budget * 1000:.0f} ms  (inference_delay={args.inference_delay} @ {args.action_hz} Hz)")
    if t_guided > budget:
        print(f"  -> WARNING: guided inference exceeds the pinned-prefix budget. Raise --inference-delay to "
              f"{int(np.ceil(t_guided * args.action_hz))} or lower the control rate.")

    if not np.isfinite(guided).all():
        print("\nFAIL: guided actions contain NaN/Inf -- guidance is diverging. Lower --max-guidance-weight.")
        return 1

    print("\nchunk magnitude and motion (robot space):")
    _stats("unguided", unguided, args.dof)
    _stats("guided", guided, args.dof)
    ratio = np.abs(np.diff(guided[:, : args.dof], axis=0)).mean() / max(
        np.abs(np.diff(unguided[:, : args.dof], axis=0)).mean(), 1e-9
    )
    print(f"    motion ratio guided/unguided = {ratio:.3f}")
    if ratio < 0.5:
        print("  -> guidance is suppressing motion; this is what a creeping//stalled arm looks like.")

    lo, hi = args.inference_delay, args.inference_delay + args.execute_horizon
    print(f"\nrows the broker would actually execute, indices [{lo}, {hi}):")
    for i in range(lo, min(hi, horizon)):
        d = np.abs(guided[i, : args.dof] - unguided[i, : args.dof]).mean()
        print(f"    [{i}] guided={np.array2string(guided[i, :4], precision=4, floatmode='fixed')}"
              f"  unguided={np.array2string(unguided[i, :4], precision=4, floatmode='fixed')}  |diff|={d:.4f}")

    print("\npinned prefix agreement with the previous chunk (model space, should be near zero):")
    d = args.inference_delay
    print(f"    guided  [0:{d}] vs prev: {np.abs(np.asarray(second[acb.RTC_RAW_ACTIONS])[:d] - prev[:d]).mean():.4f}")
    print(f"    unguided[0:{d}] vs prev: {np.abs(raw[:d] - prev[:d]).mean():.4f}   (baseline)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
