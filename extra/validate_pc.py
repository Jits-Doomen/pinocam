import numpy as np
import pandas as pd
import onnxruntime as ort

from pathlib import Path


Data_Path = Path("conjunction_dataset.csv")
Model_Path = Path("conjunction_pino.onnx")
Norm_Path = Path("normalization_params.txt")

Seed = 42

Features = [
    "miss_distance_m",
    "rel_speed_m_s",
    "sigma_primary_m",
    "sigma_secondary_m",
    "hbr_m",
]


def load_dataset():

    df = pd.read_csv(Data_Path)

    required = Features + ["pc_true"]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    df = df[required].copy()

    for column in required:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    df = df[
        (df["pc_true"] > 0.0)
        & (df["pc_true"] <= 1.0)
        ]

    for column in Features:
        df = df[df[column] >= 0.0]

    return df.reset_index(drop=True)


def load_normalization():

    values = np.loadtxt(Norm_Path)

    if values.shape != (10,):
        raise ValueError(
            "Normalization file must contain exactly 10 values."
        )

    mean = values[:5]
    std = values[5:]

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "Normalization file contains invalid values."
        )

    if np.any(std <= 0):
        raise ValueError(
            "Normalization standard deviations must be positive."
        )

    return mean, std


def main():

    df = load_dataset()

    np.random.Seed(Seed)

    indices = np.random.permutation(len(df))

    train_end = int(len(df) * 0.70)

    validation_end = train_end + int(
        len(df) * 0.15
    )

    test_indices = indices[validation_end:]

    test_df = df.iloc[
        test_indices
    ].reset_index(drop=True)

    if len(test_df) == 0:
        raise RuntimeError(
            "No held-out test samples available."
        )

    mean, std = load_normalization()

    X_raw = test_df[
        Features
    ].to_numpy(dtype=np.float64)

    X_norm = (
            (X_raw - mean) / std
    ).astype(np.float32)

    pc_true = test_df[
        "pc_true"
    ].to_numpy(dtype=np.float64)

    options = ort.SessionOptions()
    options.log_severity_level = 3

    session = ort.InferenceSession(
        str(Model_Path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    output = session.run(
        None,
        {input_name: X_norm}
    )[0]

    log10_pc_pred = output.reshape(-1)

    if not np.all(np.isfinite(log10_pc_pred)):
        raise RuntimeError(
            "Model returned non-finite predictions."
        )

    if np.any(log10_pc_pred > 0.0):
        raise RuntimeError(
            "Model returned log10(Pc) > 0, "
            "which implies Pc > 1."
        )

    pc_pred = np.power(
        10.0,
        log10_pc_pred
    )

    absolute_log_error = np.abs(
        log10_pc_pred - np.log10(pc_true)
    )

    relative_error = (
            np.abs(pc_pred - pc_true)
            / np.maximum(pc_true, 1e-300)
    )

    print("PINO held-out validation")

    print(f"Test samples: {len(test_df):,}")

    print(
        f"Mean absolute log10 error: "
        f"{np.mean(absolute_log_error):.6f}"
    )

    print(
        f"Median absolute log10 error: "
        f"{np.median(absolute_log_error):.6f}"
    )

    print(
        f"Median relative error: "
        f"{np.median(relative_error) * 100:.3f}%"
    )

    print(
        f"90th percentile relative error: "
        f"{np.percentile(relative_error, 90) * 100:.3f}%"
    )

    results = test_df.copy()

    results["pc_predicted"] = pc_pred
    results["log10_pc_predicted"] = log10_pc_pred
    results["absolute_log10_error"] = absolute_log_error
    results["relative_error"] = relative_error

    results.to_csv(
        "pc_validation_results.csv",
        index=False
    )

    print(
        "\nSaved results to pc_validation_results.csv"
    )

    print(
        "\nThis measures agreement with pc_true. "
        "It does not independently validate the physical "
        "correctness of pc_true."
    )


if __name__ == "__main__":
    main()