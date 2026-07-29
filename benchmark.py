"""Micro-benchmarks for the hot calculation paths.

Run manually (``python benchmark.py``); not part of the test suite or the app. It
just prints wall-clock timings so changes to the bolt-group, FE-import and VDI
endurance routines can be sanity-checked for speed.
"""
import random
import time

from mechanics import (
    BOLT_MATERIALS_METRIC,
    analyze_bolt_group,
    evaluate_fe_rows,
    rectangular_pattern,
    vdi2230_endurance_amplitude,
)


def benchmark() -> None:
    print("--- BENCHMARK START ---")

    # 1. analyze_bolt_group over a large grid (100 x 100 = 10,000 bolts).
    print("Generating bolt coords...")
    coords = rectangular_pattern(100, 100, 50.0, 50.0)
    print("Benchmarking analyze_bolt_group (10,000 bolts, 100 iterations)...")
    t0 = time.perf_counter()
    for _ in range(100):
        analyze_bolt_group(coords, axial_load=50000.0, moment=100.0, moment_axis="x",
                           shear_load=50000.0, shear_eccentricity=100.0)
    t1 = time.perf_counter()
    print(f"analyze_bolt_group: {(t1 - t0):.4f} seconds")

    # 2. evaluate_fe_rows over 10,000 rows across 500 unique bolts.
    print("Generating FE rows...")
    rows = [{
        "bolt_id": f"B{i % 500}",
        "diameter": 12.0, "pitch": 1.75, "grade": "Grade 10.9",
        "axial_max": random.uniform(20000, 30000),
        "axial_min": random.uniform(5000, 10000),
        "shear_max": random.uniform(1000, 5000),
        "preload": 10000.0,
    } for i in range(10000)]
    print("Benchmarking evaluate_fe_rows (10,000 rows)...")
    t0 = time.perf_counter()
    evaluate_fe_rows(rows, BOLT_MATERIALS_METRIC)
    t1 = time.perf_counter()
    print(f"evaluate_fe_rows: {(t1 - t0):.4f} seconds")

    # 3. vdi2230_endurance_amplitude in a tight loop.
    print("Benchmarking vdi2230_endurance_amplitude (1,000,000 iterations)...")
    t0 = time.perf_counter()
    for _ in range(1000000):
        vdi2230_endurance_amplitude(12.0, False, 20000.0, 45000.0)
    t1 = time.perf_counter()
    print(f"vdi2230_endurance_amplitude: {(t1 - t0):.4f} seconds")
    print("--- BENCHMARK END ---")


if __name__ == "__main__":
    benchmark()
