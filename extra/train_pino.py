import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

Data_Path = Path(
    "conjunction_dataset.csv"
)
Model_Path = Path("../conjunction_pino.onnx")
Norm_Path = Path("../normalization_params.txt")

Seed = 42

Epochs = 500
Batch_Size = 256
Learning_Rate = 1e-3

Lambda_Phys_Max = 0.15
Phys_Warmup_Epochs = 10

Train_Fraction = 0.70
Validation_Fraction = 0.15

Features = [
    "miss_distance_m",
    "rel_speed_m_s",
    "sigma_primary_m",
    "sigma_secondary_m",
    "hbr_m",
]

Target = "pc_true"

Device = torch.Device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

random.Seed(Seed)
np.random.Seed(Seed)
torch.manual_Seed(Seed)

if torch.cuda.is_available():
    torch.cuda.manual_Seed_all(Seed)

class ConjunctionPINO(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(5, 64),
            nn.GELU(),

            nn.Linear(64, 64),
            nn.GELU(),

            nn.Linear(64, 32),
            nn.GELU(),

            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)

def physics_penalty(model, batch_x):

    x = batch_x.detach().clone().requires_grad_(True)

    predictions = model(x)

    gradients = torch.autograd.grad(
        outputs=predictions,
        inputs=x,
        grad_outputs=torch.ones_like(predictions),
        create_graph=True,
    )[0]

    d_miss_distance = gradients[:, 0]
    d_hbr = gradients[:, 4]

    miss_violation = F.relu(
        d_miss_distance
    ).pow(2).mean()

    hbr_violation = F.relu(
        -d_hbr
    ).pow(2).mean()

    upper_bound_violation = F.relu(
        predictions
    ).pow(2).mean()

    return (
            miss_violation
            + hbr_violation
            + upper_bound_violation
    )

def load_dataset():

    if not Data_Path.exists():
        raise FileNotFoundError(Data_Path)

    df = pd.read_csv(Data_Path)

    required_columns = Features + [Target]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing dataset columns: {missing}"
        )

    df = df[required_columns].copy()

    for column in required_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna()

    df = df[
        (df[Target] >= 1e-20)
        & (df[Target] <= 1.0)
        ]

    for column in Features:
        df = df[df[column] >= 0.0]

    df = df.reset_index(drop=True)

    if len(df) < 100:
        raise ValueError(
            f"Only {len(df)} valid samples remain. "
            "At least 100 are required."
        )

    print(f"Loaded {len(df):,} valid samples.")

    return df

def split_dataset(df):

    indices = np.random.permutation(len(df))

    train_end = int(
        len(df) * Train_Fraction
    )

    validation_end = train_end + int(
        len(df) * Validation_Fraction
    )

    train_indices = indices[:train_end]

    validation_indices = indices[
        train_end:validation_end
    ]

    test_indices = indices[validation_end:]

    train_df = df.iloc[train_indices].copy()
    validation_df = df.iloc[validation_indices].copy()
    test_df = df.iloc[test_indices].copy()

    print(f"Training samples:   {len(train_df):,}")
    print(f"Validation samples: {len(validation_df):,}")
    print(f"Test samples:       {len(test_df):,}")

    return train_df, validation_df, test_df

def normalize_data(
        train_df,
        validation_df,
        test_df
):

    train_raw = train_df[
        Features
    ].to_numpy(dtype=np.float64)

    mean = train_raw.mean(axis=0)
    std = train_raw.std(axis=0)

    std = np.where(
        std < 1e-12,
        1.0,
        std
    )

    def transform(frame):

        raw = frame[
            Features
        ].to_numpy(dtype=np.float64)

        normalized = (raw - mean) / std

        Targets = np.log10(
            frame[Target].to_numpy(dtype=np.float64)
        ).reshape(-1, 1)

        return (
            torch.tensor(
                normalized,
                dtype=torch.float32
            ),
            torch.tensor(
                Targets,
                dtype=torch.float32
            ),
        )

    with open(
            Norm_Path,
            "w",
            encoding="utf-8"
    ) as file:

        file.write(
            " ".join(f"{value:.17g}" for value in mean)
        )

        file.write("\n")

        file.write(
            " ".join(f"{value:.17g}" for value in std)
        )

        file.write("\n")

    print(f"Normalization saved to {Norm_Path}")

    return (
        transform(train_df),
        transform(validation_df),
        transform(test_df),
    )

def evaluate(model, loader):

    model.eval()

    predictions = []
    Targets = []

    with torch.no_grad():

        for x, y in loader:

            x = x.to(Device)

            output = model(x)

            predictions.append(
                output.cpu().numpy()
            )

            Targets.append(
                y.numpy()
            )

    pred = np.concatenate(
        predictions
    ).flatten()

    true = np.concatenate(
        Targets
    ).flatten()

    log_error = np.abs(pred - true)

    pc_pred = np.power(10.0, pred)
    pc_true = np.power(10.0, true)

    relative_error = (
            np.abs(pc_pred - pc_true)
            / np.maximum(pc_true, 1e-300)
    )

    return {
        "log_mae": float(np.mean(log_error)),
        "log_median_error": float(np.median(log_error)),
        "median_relative_error_percent": float(
            np.median(relative_error) * 100.0
        ),
    }

def main():

    print(f"Training Device: {Device}")

    df = load_dataset()

    train_df, validation_df, test_df = split_dataset(df)

    (
        (x_train, y_train),
        (x_validation, y_validation),
        (x_test, y_test),
    ) = normalize_data(
        train_df,
        validation_df,
        test_df,
    )

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        Batch_Size=Batch_Size,
        shuffle=True,
        drop_last=False,
    )

    validation_loader = DataLoader(
        TensorDataset(
            x_validation,
            y_validation
        ),
        Batch_Size=Batch_Size,
        shuffle=False,
    )

    test_loader = DataLoader(
        TensorDataset(x_test, y_test),
        Batch_Size=Batch_Size,
        shuffle=False,
    )

    model = ConjunctionPINO().to(Device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Learning_Rate,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=Epochs,
        eta_min=1e-5,
    )

    criterion = nn.MSELoss()

    best_validation_loss = float("inf")
    best_state = None

    start_time = time.time()

    print("\nStarting physics-regularized training.\n")

    for epoch in range(1, Epochs + 1):

        model.train()

        lambda_phys = Lambda_Phys_Max * min(
            1.0,
            epoch / Phys_Warmup_Epochs
        )

        total_data_loss = 0.0
        total_physics_loss = 0.0
        sample_count = 0

        for batch_x, batch_y in train_loader:

            batch_x = batch_x.to(Device)
            batch_y = batch_y.to(Device)

            optimizer.zero_grad(set_to_none=True)

            predictions = model(batch_x)

            data_loss = criterion(
                predictions,
                batch_y
            )

            phys_loss = physics_penalty(
                model,
                batch_x
            )

            loss = (
                    data_loss
                    + lambda_phys * phys_loss
            )

            loss.backward()
            optimizer.step()

            count = len(batch_x)

            total_data_loss += (
                    data_loss.item() * count
            )

            total_physics_loss += (
                    phys_loss.item() * count
            )

            sample_count += count

        average_data_loss = (
                total_data_loss / sample_count
        )

        average_physics_loss = (
                total_physics_loss / sample_count
        )

        scheduler.step()

        validation_metrics = evaluate(
            model,
            validation_loader
        )

        validation_loss = validation_metrics["log_mae"]

        if validation_loss < best_validation_loss:

            best_validation_loss = validation_loss

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        if epoch == 1 or epoch % 2 == 0:

            print(
                f"Epoch {epoch:02d}/{Epochs} | "
                f"Data MSE: {average_data_loss:.6f} | "
                f"Physics: {average_physics_loss:.6f} | "
                f"Validation log-MAE: {validation_loss:.6f}"
            )

    if best_state is None:
        raise RuntimeError(
            "Training did not produce a valid model."
        )

    model.load_state_dict(best_state)

    print(
        f"\nTraining finished in "
        f"{time.time() - start_time:.2f} seconds."
    )

    test_metrics = evaluate(
        model,
        test_loader
    )

    print("\nHeld-out test-set results")

    for key, value in test_metrics.items():
        print(f"{key}: {value:.6f}")

    model.eval()
    model.cpu()

    dummy_input = torch.zeros(
        1,
        5,
        dtype=torch.float32
    )

    torch.onnx.export(
        model,
        dummy_input,
        str(Model_Path),
        input_names=["conjunction_inputs"],
        output_names=["log10_pc"],
        dynamic_axes={
            "conjunction_inputs": {
                0: "Batch_Size"
            },
            "log10_pc": {
                0: "Batch_Size"
            },
        },
        opset_version=17,
    )

    print(f"\nONNX model exported: {Model_Path}")
    print(f"Normalization file: {Norm_Path}")

    print(
        "\nNote: a random row split does not prevent "
        "scenario-level data leakage."
    )


if __name__ == "__main__":
    main()
