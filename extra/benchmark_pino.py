import time

import numpy as np
import pandas as pd
import onnxruntime as ort
from pathlib import Path


Model_Path = "conjunction_pino.onnx"
Data_Path = Path(
    "conjunction_dataset.csv"
)
Norm_Path = "normalization_params.txt"

Features = [
    "miss_distance_m",
    "rel_speed_m_s",
    "sigma_primary_m",
    "sigma_secondary_m",
    "hbr_m",
]

Max_Samples = 10000
Warmup_Runs = 5
Benchmark_Runs = 20


def main():

    norm = np.loadtxt(Norm_Path)

    if norm.shape != (10,):
        raise ValueError(
            "Invalid normalization file."
        )

    mean = norm[:5]
    std = norm[5:]

    if np.any(std <= 0):
        raise ValueError(
            "Invalid normalization standard deviations."
        )

    df = pd.read_csv(Data_Path)

    df = df[
        (df["pc_true"] > 0.0)
        & (df["pc_true"] <= 1.0)
        ].copy()

    df = df.dropna(
        subset=Features + ["pc_true"]
    )

    df = df[
        (df[Features] >= 0.0).all(axis=1)
    ]

    df = df.head(Max_Samples)

    if len(df) == 0:
        raise RuntimeError(
            "No valid samples available for benchmarking."
        )

    X_raw = df[
        Features
    ].to_numpy(dtype=np.float64)

    X_norm = (
            (X_raw - mean) / std
    ).astype(np.float32)

    options = ort.SessionOptions()
    options.log_severity_level = 3

    session = ort.InferenceSession(
        Model_Path,
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    # Warm-up.
    for _ in range(Warmup_Runs):

        session.run(
            None,
            {input_name: X_norm}
        )

    timings = []

    for _ in range(Benchmark_Runs):

        start = time.perf_counter()

        output = session.run(
            None,
            {input_name: X_norm}
        )[0]

        elapsed = time.perf_counter() - start

        timings.append(elapsed)

    timings = np.asarray(timings)

    mean_time = timings.mean()
    median_time = np.median(timings)

    samples = len(df)

    log10_pc = output.reshape(-1)

    if not np.all(np.isfinite(log10_pc)):
        raise RuntimeError(
            "Model returned non-finite output."
        )

    if np.any(log10_pc > 0.0):
        raise RuntimeError(
            "Model returned log10(Pc) > 0."
        )

    pc_pred = np.power(10.0, log10_pc)

    pc_true = df[
        "pc_true"
    ].to_numpy(dtype=np.float64)

    log_error = np.abs(
        log10_pc - np.log10(pc_true)
    )

    relative_error = (
            np.abs(pc_pred - pc_true)
            / np.maximum(pc_true, 1e-300)
    )

    print("\nPINO benchmark")

    print(f"Samples: {samples:,}")
    print(f"Benchmark runs: {Benchmark_Runs}")

    print(
        f"\nMean batch inference time: "
        f"{mean_time * 1000:.3f} ms"
    )

    print(
        f"Median batch inference time: "
        f"{median_time * 1000:.3f} ms"
    )

    print(
        f"Mean latency per sample: "
        f"{mean_time * 1e6 / samples:.3f} microseconds"
    )

    print(
        f"Throughput: "
        f"{samples / mean_time:,.0f} samples/second"
    )

    print(
        f"\nMean absolute log10 error: "
        f"{np.mean(log_error):.6f}"
    )

    print(
        f"Median relative error: "
        f"{np.median(relative_error) * 100:.3f}%"
    )


if __name__ == "__main__":
    main()