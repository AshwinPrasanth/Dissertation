from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import subprocess
import time
import pandas as pd

KERNEL_DIR = Path("../PACE2025-reduced/exact")
UZL = Path("../PACE2025/target/release/uzl_hs")
OUTPUT = Path("data/solver_runs/uzl.csv")

TIMEOUT = 300
WORKERS = 16


def run_instance(path):
    start = time.perf_counter()

    try:
        with open(path, "r") as f:
            result = subprocess.run(
                [str(UZL)],
                stdin=f,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=TIMEOUT,
            )

        runtime = time.perf_counter() - start

        return {
            "instance": path.stem,
            "solver": "uzl",
            "runtime": runtime,
            "timeout": 0,
            "return_code": result.returncode,
        }

    except subprocess.TimeoutExpired:
        runtime = time.perf_counter() - start

        return {
            "instance": path.stem,
            "solver": "uzl",
            "runtime": TIMEOUT,
            "timeout": 1,
            "return_code": 124,
        }


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(KERNEL_DIR.glob("*.hgr"))

    print(f"Found {len(files)} kernels")
    print(f"Workers: {WORKERS}")
    print(f"Timeout per instance: {TIMEOUT}s")

    rows = []

    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        for i, row in enumerate(
            executor.map(run_instance, files),
            start=1,
        ):
            rows.append(row)

            print(
                f"[{i:3d}/{len(files)}] "
                f"{row['instance']} "
                f"{row['runtime']:.3f}s "
                f"{'TIMEOUT' if row['timeout'] else 'SOLVED'}",
                flush=True,
            )

    df = pd.DataFrame(rows)

    df.to_csv(OUTPUT, index=False)

    print()
    print(f"Saved: {OUTPUT}")
    print(f"Solved: {(df['timeout'] == 0).sum()}/{len(df)}")
    print(f"Timeouts: {df['timeout'].sum()}")
    print(f"Mean runtime: {df['runtime'].mean():.3f}s")


if __name__ == "__main__":
    main()