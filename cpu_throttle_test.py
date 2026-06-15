#!/usr/bin/env python3
"""CPU sustained-load benchmark for observing thermal throttling over time."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import multiprocessing as mp
import os
import queue
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path


STATUS_PEAK = "peak"
STATUS_WATCH = "watch"
STATUS_DROP = "drop"


@dataclass(frozen=True)
class Sample:
    bucket: int
    elapsed: float
    units: int
    rate: float
    relative: float
    status: str
    complete: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a sustained multi-process CPU load and print aligned "
            "time-bucket throughput, useful for spotting thermal throttling."
        )
    )
    parser.add_argument(
        "--workload",
        choices=("backtest", "sha256"),
        default="backtest",
        help="CPU workload to run: quant-style backtest or SHA256 hashing (default: backtest)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=45.0,
        help="test duration in minutes after warm-up (default: 45)",
    )
    parser.add_argument(
        "--bucket",
        type=float,
        default=10.0,
        help="sampling bucket length in seconds (default: 10)",
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=5.0,
        help="warm-up seconds before official sampling starts (default: 5)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 8,
        help="worker process count (default: logical CPU count)",
    )
    parser.add_argument(
        "--baseline-buckets",
        type=int,
        default=3,
        help="number of first buckets averaged as 100%% baseline (default: 3)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="optional path to save machine-readable CSV results",
    )
    parser.add_argument(
        "--data-mib",
        type=int,
        default=1,
        help="data size hashed per SHA256 operation, in MiB (default: 1)",
    )
    parser.add_argument(
        "--symbols",
        type=int,
        default=6,
        help="synthetic symbols per backtest unit (default: 6)",
    )
    parser.add_argument(
        "--bars",
        type=int,
        default=8_000,
        help="bars per synthetic symbol for each backtest unit (default: 8000)",
    )
    parser.add_argument(
        "--parameter-sets",
        type=int,
        default=4,
        help="strategy parameter sets swept per backtest unit (default: 4)",
    )
    return parser.parse_args()


def fmt_duration(seconds: float) -> str:
    total = int(round(seconds))
    minutes, sec = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def make_price_series(worker_id: int, symbols: int, bars: int) -> list[list[float]]:
    series: list[list[float]] = []
    for symbol in range(symbols):
        seed = (worker_id + 1) * 1_000_003 + (symbol + 17) * 97_409
        price = 80.0 + ((seed >> 8) % 7_000) / 100.0
        symbol_prices: list[float] = []
        for bar in range(bars):
            seed = (seed * 1_664_525 + 1_013_904_223) & 0xFFFFFFFF
            noise = ((seed >> 9) / 8_388_608.0) - 0.25
            cycle = math.sin((bar + 1) * (0.003 + symbol * 0.00017))
            shock = 0.00055 * noise + 0.00035 * cycle
            price *= 1.0 + shock
            if price < 1.0:
                price = 1.0
            symbol_prices.append(price)
        series.append(symbol_prices)
    return series


def run_backtest_unit(prices_by_symbol: list[list[float]], parameter_sets: int) -> float:
    score = 0.0
    for param in range(parameter_sets):
        fast_alpha = 2.0 / (8.0 + param * 3.0)
        slow_alpha = 2.0 / (34.0 + param * 7.0)
        vol_alpha = 2.0 / (48.0 + param * 5.0)
        threshold = 0.0006 + param * 0.00012
        fee = 0.00012
        slip = 0.00008

        for prices in prices_by_symbol:
            fast = prices[0]
            slow = prices[0]
            vol = 0.0001
            position = 0.0
            pnl = 0.0
            exposure = 0.0
            last_price = prices[0]

            for index in range(1, len(prices)):
                price = prices[index]
                ret = price / last_price - 1.0
                fast += fast_alpha * (price - fast)
                slow += slow_alpha * (price - slow)
                vol += vol_alpha * (abs(ret) - vol)

                signal_strength = (fast / slow - 1.0) / (vol + 0.0001)
                if signal_strength > threshold:
                    target = 1.0
                elif signal_strength < -threshold:
                    target = -1.0
                else:
                    target = 0.0

                turnover = abs(target - position)
                pnl += position * ret - turnover * (fee + slip)
                exposure += abs(target)
                position = target
                last_price = price

            score += pnl - 0.000001 * exposure
    return score


def run_sha256_unit(data: bytes) -> float:
    return float(hashlib.sha256(data).digest()[0])


def worker(
    worker_id: int,
    stop: mp.Event,
    begin: mp.Event,
    ready_q: mp.Queue,
    result_q: mp.Queue,
    start_at: mp.Value,
    bucket_seconds: float,
    workload: str,
    data_mib: int,
    symbols: int,
    bars: int,
    parameter_sets: int,
) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    data = b"x" * data_mib * 1024 * 1024 if workload == "sha256" else b""
    prices_by_symbol = (
        make_price_series(worker_id, symbols=symbols, bars=bars)
        if workload == "backtest"
        else []
    )
    checksum = 0.0
    ready_q.put(worker_id)
    begin.wait()

    official_start = float(start_at.value)

    while not stop.is_set() and time.time() < official_start:
        if workload == "sha256":
            checksum += run_sha256_unit(data)
        else:
            checksum += run_backtest_unit(prices_by_symbol, parameter_sets)

    bucket = 1
    bucket_end = official_start + bucket_seconds
    count = 0

    while not stop.is_set():
        if workload == "sha256":
            checksum += run_sha256_unit(data)
        else:
            checksum += run_backtest_unit(prices_by_symbol, parameter_sets)
        count += 1
        now = time.time()
        if now >= bucket_end:
            result_q.put((bucket, count, checksum))
            count = 0
            bucket += 1
            bucket_end = official_start + bucket * bucket_seconds


def drain_ready(ready_q: mp.Queue, workers: int, timeout: float) -> None:
    ready = set()
    deadline = time.time() + timeout
    while len(ready) < workers:
        remaining = deadline - time.time()
        if remaining <= 0:
            raise TimeoutError(f"only {len(ready)}/{workers} workers became ready")
        ready.add(ready_q.get(timeout=remaining))


def collect_bucket(
    result_q: mp.Queue,
    pending: dict[int, list[int]],
    bucket: int,
    workers: int,
    deadline: float,
) -> tuple[int, bool]:
    values = pending.pop(bucket, [])

    while len(values) < workers and time.time() < deadline:
        timeout = max(0.01, min(0.2, deadline - time.time()))
        try:
            seen = result_q.get(timeout=timeout)
        except queue.Empty:
            continue

        seen_bucket, count = seen[0], seen[1]
        if seen_bucket == bucket:
            values.append(count)
        else:
            pending.setdefault(seen_bucket, []).append(count)

    return sum(values), len(values) == workers


def classify(relative: float) -> str:
    if relative >= 0.95:
        return STATUS_PEAK
    if relative >= 0.90:
        return STATUS_WATCH
    return STATUS_DROP


def print_header(unit_label: str) -> None:
    print()
    print(
        f"{'#':>4} {'Elapsed':>8} {unit_label:>12} {'Rate/s':>12} "
        f"{'Relative':>10} {'Status':>8} {'OK':>4}"
    )
    print("-" * 76)


def print_sample(sample: Sample) -> None:
    ok = "yes" if sample.complete else "no"
    print(
        f"{sample.bucket:>4d} "
        f"{fmt_duration(sample.elapsed):>8} "
        f"{sample.units:>12,d} "
        f"{sample.rate:>12,.1f} "
        f"{sample.relative:>9.1%} "
        f"{sample.status:>8} "
        f"{ok:>4}",
        flush=True,
    )


def open_csv(path: Path | None) -> tuple[object | None, csv.DictWriter | None]:
    if path is None:
        return None, None
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", newline="")
    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "bucket",
            "elapsed_seconds",
            "units",
            "rate_per_second",
            "relative",
            "status",
            "complete",
        ],
    )
    writer.writeheader()
    return handle, writer


def write_csv(writer: csv.DictWriter | None, sample: Sample) -> None:
    if writer is None:
        return
    writer.writerow(
        {
            "bucket": sample.bucket,
            "elapsed_seconds": round(sample.elapsed, 3),
            "units": sample.units,
            "rate_per_second": round(sample.rate, 3),
            "relative": round(sample.relative, 6),
            "status": sample.status,
            "complete": sample.complete,
        }
    )


def main() -> int:
    args = parse_args()
    if args.duration <= 0:
        raise SystemExit("--duration must be positive")
    if args.bucket <= 0:
        raise SystemExit("--bucket must be positive")
    if args.warmup < 0:
        raise SystemExit("--warmup cannot be negative")
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if args.baseline_buckets <= 0:
        raise SystemExit("--baseline-buckets must be positive")
    if args.data_mib <= 0:
        raise SystemExit("--data-mib must be positive")
    if args.symbols <= 0:
        raise SystemExit("--symbols must be positive")
    if args.bars < 2:
        raise SystemExit("--bars must be at least 2")
    if args.parameter_sets <= 0:
        raise SystemExit("--parameter-sets must be positive")

    duration_seconds = args.duration * 60.0
    bucket_count = int(math.ceil(duration_seconds / args.bucket))
    baseline_count = min(args.baseline_buckets, bucket_count)

    stop = mp.Event()
    begin = mp.Event()
    ready_q: mp.Queue = mp.Queue()
    result_q: mp.Queue = mp.Queue()
    start_at = mp.Value("d", 0.0)
    pending: dict[int, list[int]] = {}
    processes: list[mp.Process] = []
    csv_handle = None

    try:
        for worker_id in range(args.workers):
            process = mp.Process(
                target=worker,
                args=(
                    worker_id,
                    stop,
                    begin,
                    ready_q,
                    result_q,
                    start_at,
                    args.bucket,
                    args.workload,
                    args.data_mib,
                    args.symbols,
                    args.bars,
                    args.parameter_sets,
                ),
            )
            process.start()
            processes.append(process)

        print(
            "Workers: "
            f"{args.workers} | Workload: {args.workload} | Bucket: {args.bucket:g}s | "
            f"Warm-up: {args.warmup:g}s | Duration: {fmt_duration(duration_seconds)} | "
            f"Baseline buckets: {baseline_count}"
        )
        if args.workload == "backtest":
            print(
                "Backtest: "
                f"{args.symbols} symbols | {args.bars:,} bars | "
                f"{args.parameter_sets} parameter sets per run"
            )
        else:
            print(f"SHA256: {args.data_mib} MiB per hash")
        print("Starting workers...")
        drain_ready(ready_q, args.workers, timeout=15.0)

        start_at.value = time.time() + args.warmup
        begin.set()
        print(f"Warming up for {args.warmup:g}s...")

        csv_handle, csv_writer = open_csv(args.csv)

        raw_rows: list[tuple[int, float, int, float, bool]] = []
        baseline_rate: float | None = None
        printed_header = False
        unit_label = "Hashes" if args.workload == "sha256" else "Runs"

        for bucket in range(1, bucket_count + 1):
            official_start = float(start_at.value)
            bucket_end = official_start + bucket * args.bucket
            sleep_for = bucket_end - time.time()
            if sleep_for > 0:
                time.sleep(sleep_for)

            total, complete = collect_bucket(
                result_q=result_q,
                pending=pending,
                bucket=bucket,
                workers=args.workers,
                deadline=bucket_end + 2.0,
            )
            elapsed = min(bucket * args.bucket, duration_seconds)
            rate = total / args.bucket
            raw_rows.append((bucket, elapsed, total, rate, complete))

            if len(raw_rows) == baseline_count:
                baseline_rate = sum(row[3] for row in raw_rows) / baseline_count
                if baseline_rate <= 0:
                    baseline_rate = 1.0
                print_header(unit_label)
                printed_header = True
                for row in raw_rows:
                    sample = Sample(
                        bucket=row[0],
                        elapsed=row[1],
                        units=row[2],
                        rate=row[3],
                        relative=row[3] / baseline_rate,
                        status=classify(row[3] / baseline_rate),
                        complete=row[4],
                    )
                    print_sample(sample)
                    write_csv(csv_writer, sample)
            elif baseline_rate is not None:
                sample = Sample(
                    bucket=bucket,
                    elapsed=elapsed,
                    units=total,
                    rate=rate,
                    relative=rate / baseline_rate,
                    status=classify(rate / baseline_rate),
                    complete=complete,
                )
                if not printed_header:
                    print_header(unit_label)
                    printed_header = True
                print_sample(sample)
                write_csv(csv_writer, sample)

        if csv_handle is not None:
            csv_handle.flush()

        print()
        print("Done. Treat sustained 'drop' rows as likely thermal throttling.")
        if args.csv is not None:
            print(f"CSV saved to: {args.csv}")
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted. Stopping workers...", file=sys.stderr)
        return 130
    finally:
        stop.set()
        begin.set()
        for process in processes:
            process.join(timeout=3.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
        if csv_handle is not None:
            csv_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
