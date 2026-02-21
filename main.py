#!/usr/bin/env python3
import argparse
import queue
import time

import numpy as np
import sounddevice as sd

from parser import LoweringError, ParseError, build_callable, node_to_expanded_str, parse_dsl

def _show_error(src: str, span) -> None:
    if span is None:
        return
    a, b = span
    a = max(0, a)
    b = min(len(src), b)
    print(src)
    print(" " * a + "^" * max(1, b - a))


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="math2osc",
        description="Toy realtime DSL->numpy->audio pipeline. Ctrl-C to stop.",
    )
    ap.add_argument("expr", help="Function of x (phase in radians), e.g. 'sin(x) + 0.2*sin(2*x)'")
    ap.add_argument("--hz", type=float, default=220.0, help="Fundamental frequency in Hz")
    ap.add_argument("--sr", type=float, default=48_000.0, help="Sample rate")
    ap.add_argument("--block", type=int, default=1024, help="Block size")
    ap.add_argument("--gain", type=float, default=0.1, help="Output gain (applied then clipped to [-1,1])")
    ap.add_argument(
        "--no-abort-on-clip",
        action="store_true",
        help="Stop immediately if clipping is detected.",
    )
    ap.add_argument(
        "--report-every",
        type=int,
        default=200,
        help="Report timing every N blocks (printed from main thread)",
    )
    args = ap.parse_args()

    try:
        dsl_ast = parse_dsl(args.expr)
        expanded = node_to_expanded_str(dsl_ast)
        print("expanded:", expanded)
        f = build_callable(dsl_ast, sr=float(args.sr))
    except (ParseError, LoweringError) as e:
        print(type(e).__name__ + ":", e)
        _show_error(args.expr, getattr(e, "span", None))
        raise SystemExit(2)
    except Exception as e:
        print(type(e).__name__ + ":", e)
        raise SystemExit(2)

    sr = float(args.sr)
    block = int(args.block)
    hz = float(args.hz)
    gain = float(args.gain)
    no_abort_on_clip = bool(args.no_abort_on_clip)

    dphase = 2.0 * np.pi * hz / sr
    phase0 = 0.0

    report_every = max(1, int(args.report_every))
    stats_q = queue.SimpleQueue()
    stat_blocks = 0
    stat_sum_compute_ns = 0
    stat_sum_budget_ns = 0
    stat_max_ratio = 0.0

    clip_detected = False

    def callback(outdata, frames, time, status):
        nonlocal phase0
        nonlocal stat_blocks, stat_sum_compute_ns, stat_sum_budget_ns, stat_max_ratio
        nonlocal clip_detected
        if status:
            # Keep callback realtime-safe; ignore transient warnings for now
            pass

        t0 = time_module.perf_counter_ns()
        n = np.arange(frames, dtype=np.float64)
        x = phase0 + dphase * n

        y = f(x) * gain
        if not no_abort_on_clip and np.any((y < -1.0) | (y > 1.0)):
            clip_detected = True
            raise sd.CallbackAbort
        y = np.clip(y, -1.0, 1.0).astype(np.float32, copy=False)
        outdata[:, 0] = y
        t1 = time_module.perf_counter_ns()

        # Keep phase unwrapped so expressions like sin(0.5*x) remain continuous
        phase0 = phase0 + dphase * frames

        budget_ns = int(frames * 1_000_000_000.0 / sr)
        compute_ns = int(t1 - t0)
        ratio = (compute_ns / budget_ns) if budget_ns > 0 else 0.0

        stat_blocks += 1
        stat_sum_compute_ns += compute_ns
        stat_sum_budget_ns += budget_ns
        if ratio > stat_max_ratio:
            stat_max_ratio = ratio

        if stat_blocks >= report_every:
            avg_ratio = (stat_sum_compute_ns / stat_sum_budget_ns) if stat_sum_budget_ns > 0 else 0.0
            stats_q.put((stat_blocks, avg_ratio, stat_max_ratio))
            stat_blocks = 0
            stat_sum_compute_ns = 0
            stat_sum_budget_ns = 0
            stat_max_ratio = 0.0

    try:
        time_module = time  # local alias for faster access in the callback
        with sd.OutputStream(
            samplerate=int(sr),
            channels=1,
            dtype="float32",
            blocksize=block,
            callback=callback,
        ):
            while True:
                sd.sleep(250)
                if clip_detected:
                    print("clipping detected, stopping")
                    break
                while True:
                    try:
                        blocks, avg_ratio, max_ratio = stats_q.get_nowait()
                    except Exception:
                        break
                    print(
                        f"timing over last {blocks} blocks: "
                        f"avg {avg_ratio * 100.0:.1f}% "
                        f"max {max_ratio * 100.0:.1f}%"
                    )
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
