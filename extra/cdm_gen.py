"""Synthetic CDM generator for PINO pipeline development.

Please read whoever is checking this file:
- This creates synthetic, idealized two-body-style scenarios, NOT operational
  tracking data or validated physical conjunction truth.
- The covariance model is an explicit design assumption, not a calibrated OD
  or TLE error model.
- Pc is calculated by the local pc2d_foster.py implementation. It is therefore
  not independent validation ground truth.
- Do not train/evaluate a model as though these labels were real-world truth..
"""

import argparse
import hashlib
import json
import math
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from pc2d_foster import HBRType, calculate_pc_foster, ric_to_eci_transform

MU_Earth = 398600.4418
Earth_Radius_KM = 6378.137
Catalog_Base = 90000
EPS = 1e-12


def normalize(vector):
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < EPS:
        raise ValueError("Cannot normalize zero or non-finite vector.")
    return np.asarray(vector, dtype=np.float64) / norm


def random_unit_vector(rng):
    return normalize(rng.normal(size=3))


def tangent_basis(radial):
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(radial, reference))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    first = normalize(np.cross(reference, radial))
    second = normalize(np.cross(radial, first))
    return first, second


def synthetic_covariance_eci(r, v, age_days):
    age = float(age_days)
    if not np.isfinite(age) or age < 0:
        raise ValueError("age_days must be finite and non-negative.")

    sig_r = 0.050 + 0.010 * age
    sig_t = 0.350 + 0.250 * age
    sig_n = 0.050 + 0.015 * age

    factor = np.array([
        [sig_r, 0.0, 0.0],
        [0.20 * sig_t, sig_t * math.sqrt(1.0 - 0.20**2), 0.0],
        [0.08 * sig_n, -0.12 * sig_n, sig_n * 0.985],
    ])
    cov_rtn_pos = factor @ factor.T
    cov_rtn_vel = np.diag([
        (2e-5 + 5e-6 * age) ** 2,
        (5e-5 + 1e-5 * age) ** 2,
        (2e-5 + 5e-6 * age) ** 2,
    ])

    rotation = ric_to_eci_transform(r, v)
    cov = np.zeros((6, 6), dtype=np.float64)
    cov[:3, :3] = rotation @ cov_rtn_pos @ rotation.T
    cov[3:, 3:] = rotation @ cov_rtn_vel @ rotation.T
    cov = 0.5 * (cov + cov.T)

    if np.min(np.linalg.eigvalsh(cov)) <= 0:
        raise RuntimeError("Synthetic covariance is not positive definite.")
    return cov


def sample_primary_state(rng, altitude_min=350.0, altitude_max=1500.0):
    altitude = float(rng.uniform(altitude_min, altitude_max))
    radius = Earth_Radius_KM + altitude
    radial = random_unit_vector(rng)
    tangent_a, tangent_b = tangent_basis(radial)
    heading = float(rng.uniform(0.0, 2.0 * math.pi))
    direction = math.cos(heading) * tangent_a + math.sin(heading) * tangent_b
    speed = math.sqrt(MU_Earth / radius)
    return radius * radial, speed * direction


def validate_scenario(s):
    vectors = ("r1", "v1", "r2", "v2")
    for key in vectors:
        value = np.asarray(s[key], dtype=float)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError(f"{key} must be a finite 3-vector.")

    r1, v1, r2, v2 = (s[k] for k in vectors)
    for key in ("cov1", "cov2"):
        cov = np.asarray(s[key], dtype=float)
        if cov.shape != (6, 6) or not np.all(np.isfinite(cov)):
            raise ValueError(f"{key} must be a finite 6x6 matrix.")
        if not np.allclose(cov, cov.T, atol=1e-12, rtol=1e-10):
            raise ValueError(f"{key} must be symmetric.")
        if np.min(np.linalg.eigvalsh(0.5 * (cov + cov.T))) <= 0:
            raise ValueError(f"{key} must be positive definite.")

    radius1, radius2 = np.linalg.norm(r1), np.linalg.norm(r2)
    if min(radius1, radius2) <= Earth_Radius_KM:
        raise ValueError("An object is at or below Earth's surface.")

    tangency = max(abs(float(np.dot(r1, v1))), abs(float(np.dot(r2, v2))))
    speed_error = max(
        abs(float(np.linalg.norm(v1)) - math.sqrt(MU_Earth / radius1)),
        abs(float(np.linalg.norm(v2)) - math.sqrt(MU_Earth / radius2)),
    )
    rel_r, rel_v = r1 - r2, v1 - v2
    rel_speed = float(np.linalg.norm(rel_v))
    tca_residual = abs(float(np.dot(rel_r, rel_v)))

    if rel_speed < 0.05:
        raise ValueError("Relative speed below configured minimum.")
    if tca_residual > 1e-8:
        raise ValueError("State does not satisfy linear TCA orthogonality.")
    if tangency > 1e-8 or speed_error > 1e-10:
        raise ValueError("State violates the circular-orbit construction.")

    return {
        "relative_speed_km_s": rel_speed,
        "miss_distance_km": float(np.linalg.norm(rel_r)),
        "linear_tca_residual_km2_s": tca_residual,
        "circular_speed_residual_km_s": speed_error,
    }


def sample_conjunction(rng, max_attempts=1000):
    for _ in range(max_attempts):
        r1, v1 = sample_primary_state(rng)
        radius = float(np.linalg.norm(r1))
        speed = float(np.linalg.norm(v1))
        radial1 = normalize(r1)

        age1, age2 = rng.uniform(0.0, 7.0, size=2)
        combined_sigma = math.hypot(0.350 + 0.250 * age1, 0.350 + 0.250 * age2)
        target_miss = combined_sigma * float(rng.uniform(0.02, 4.0))
        hbr = float(rng.uniform(0.001, 0.020))
        central_angle = 2.0 * math.asin(min(0.999999, target_miss / (2.0 * radius)))

        ta, tb = tangent_basis(radial1)
        angle = float(rng.uniform(0.0, 2.0 * math.pi))
        separation = math.cos(angle) * ta + math.sin(angle) * tb
        radial2 = normalize(
            math.cos(central_angle) * radial1
            + math.sin(central_angle) * separation
        )
        r2 = radius * radial2
        relative_position = r1 - r2

        ea, eb = tangent_basis(radial2)
        ca = float(np.dot(relative_position, ea))
        cb = float(np.dot(relative_position, eb))
        amplitude = math.hypot(ca, cb)
        target = float(np.dot(relative_position, v1)) / speed
        if amplitude < EPS or abs(target) > amplitude:
            continue

        phase = math.atan2(cb, ca)
        delta = math.acos(float(np.clip(target / amplitude, -1.0, 1.0)))
        velocity_angle = phase + (delta if rng.random() < 0.5 else -delta)
        v2 = speed * (
            math.cos(velocity_angle) * ea + math.sin(velocity_angle) * eb
        )

        cov1 = synthetic_covariance_eci(r1, v1, age1)
        cov2 = synthetic_covariance_eci(r2, v2, age2)
        pc = calculate_pc_foster(
            r1, v1, cov1, r2, v2, cov2,
            HBR=hbr, hbr_type=HBRType.CIRCLE,
        )

        if (
            not pc.get("IsPosDef", False)
            or not pc.get("IntegrationConverged", False)
            or not np.isfinite(pc.get("Pc", np.nan))
            or not np.isfinite(pc.get("LogPc", np.nan))
        ):
            continue

        scenario = {
            "r1": r1, "v1": v1, "cov1": cov1, "age1": float(age1),
            "r2": r2, "v2": v2, "cov2": cov2, "age2": float(age2),
            "hbr_km": hbr, "pc_result": pc,
        }
        try:
            scenario["quality"] = validate_scenario(scenario)
            return scenario
        except ValueError:
            pass

    raise RuntimeError(f"Could not generate a valid scenario in {max_attempts} attempts.")


def covariance_position_eci_to_rtn_m2(r, v, covariance):
    rotation = ric_to_eci_transform(r, v)
    block = rotation.T @ covariance[:3, :3] @ rotation
    return 0.5 * (block + block.T) * 1e6


def kvn(key, value, unit=None):
    if unit is None:
        return f"{key:<36}= {value}"
    return f"{key:<36}= {value:<26}[{unit}]"


def build_cdm_text(s, message_id, creation_dt, tca_dt):
    def stamp(dt):
        if dt.tzinfo is None:
            raise ValueError("CDM dates must be timezone-aware.")
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

    r1, v1, cov1 = s["r1"], s["v1"], s["cov1"]
    r2, v2, cov2 = s["r2"], s["v2"], s["cov2"]
    rel_r, rel_v = r1 - r2, v1 - v2
    ric = ric_to_eci_transform(r1, v1)
    rel_r_rtn = ric.T @ rel_r * 1000.0
    rel_v_rtn = ric.T @ rel_v * 1000.0

    index = int(message_id.rsplit("_", 1)[1])
    oid1 = Catalog_Base + 2 * index
    oid2 = oid1 + 1

    lines = [
        kvn("CCSDS_CDM_VERS", "1.0"),
        kvn("CREATION_DATE", stamp(creation_dt)),
        kvn("ORIGINATOR", "SYNTH-GEN"),
        kvn("MESSAGE_ID", message_id),
        kvn("TCA", stamp(tca_dt)),
        kvn("MISS_DISTANCE", f"{np.linalg.norm(rel_r) * 1000:.6f}", "m"),
        kvn("RELATIVE_SPEED", f"{np.linalg.norm(rel_v) * 1000:.9f}", "m/s"),
        kvn("RELATIVE_POSITION_R", f"{rel_r_rtn[0]:.6f}", "m"),
        kvn("RELATIVE_POSITION_T", f"{rel_r_rtn[1]:.6f}", "m"),
        kvn("RELATIVE_POSITION_N", f"{rel_r_rtn[2]:.6f}", "m"),
        kvn("RELATIVE_VELOCITY_R", f"{rel_v_rtn[0]:.9f}", "m/s"),
        kvn("RELATIVE_VELOCITY_T", f"{rel_v_rtn[1]:.9f}", "m/s"),
        kvn("RELATIVE_VELOCITY_N", f"{rel_v_rtn[2]:.9f}", "m/s"),
        kvn("COMMENT HBR_METRES", f"{s['hbr_km'] * 1000:.3f}"),
        kvn("COMMENT SYNTHETIC_PC_FOSTER", f"{s['pc_result']['Pc']:.16e}"),
        kvn("COMMENT SYNTHETIC_ONLY", "NOT OPERATIONAL TRACKING DATA"),
        kvn("COMMENT TCA_MODEL", "LINEAR RELATIVE RANGE RATE ZERO; NONLINEAR TCA UNVERIFIED"),
    ]

    def object_block(label, oid, r, v, cov, age):
        return [
            kvn("OBJECT", label),
            kvn("OBJECT_DESIGNATOR", str(oid)),
            kvn("CATALOG_NAME", "SYNTH_SATCAT"),
            kvn("OBJECT_NAME", f"SYNTH-{oid}"),
            kvn("INTERNATIONAL_DESIGNATOR", f"SYNTH-{oid}"),
            kvn("EPHEMERIS_NAME", "TWO_BODY_SYNTHETIC"),
            kvn("COVARIANCE_METHOD", "CALCULATED"),
            kvn("MANEUVERABLE", "N/A"),
            kvn("REF_FRAME", "EME2000"),
            kvn("EPOCH", stamp(tca_dt)),
            kvn("COMMENT SYNTHETIC_COVARIANCE_AGE_DAYS", f"{age:.6f}"),
            kvn("X", f"{r[0]:.12f}", "km"),
            kvn("Y", f"{r[1]:.12f}", "km"),
            kvn("Z", f"{r[2]:.12f}", "km"),
            kvn("X_DOT", f"{v[0]:.14f}", "km/s"),
            kvn("Y_DOT", f"{v[1]:.14f}", "km/s"),
            kvn("Z_DOT", f"{v[2]:.14f}", "km/s"),
            kvn("CR_R", f"{cov[0, 0]:.15e}", "m**2"),
            kvn("CT_R", f"{cov[1, 0]:.15e}", "m**2"),
            kvn("CT_T", f"{cov[1, 1]:.15e}", "m**2"),
            kvn("CN_R", f"{cov[2, 0]:.15e}", "m**2"),
            kvn("CN_T", f"{cov[2, 1]:.15e}", "m**2"),
            kvn("CN_N", f"{cov[2, 2]:.15e}", "m**2"),
        ]

    lines.extend(object_block(
        "OBJECT1", oid1, r1, v1,
        covariance_position_eci_to_rtn_m2(r1, v1, cov1), s["age1"]
    ))
    lines.extend(object_block(
        "OBJECT2", oid2, r2, v2,
        covariance_position_eci_to_rtn_m2(r2, v2, cov2), s["age2"]
    ))
    return "\r\n".join(lines) + "\r\n"


def atomic_write(path, content):
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(content, encoding="utf-8", newline="")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_utc(value):
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if dt.tzinfo is None:
        raise argparse.ArgumentTypeError("Timestamp must include a UTC offset.")
    return dt.astimezone(timezone.utc)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic research CDMs.")
    parser.add_argument("--out-dir", default="cdms_gen")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--creation-date", type=parse_utc, default=None)
    parser.add_argument("--interval-seconds", type=float, default=0.0)
    args = parser.parse_args()

    if args.count < 1:
        parser.error("--count must be at least one.")
    if args.interval_seconds < 0:
        parser.error("--interval-seconds cannot be negative.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        parser.error(f"{manifest_path} already exists; choose a new output directory.")

    creation_dt = args.creation_date or datetime.now(timezone.utc)
    run_stamp = creation_dt.strftime("%Y%m%dT%H%M%S")
    rng = np.random.default_rng(args.seed)
    manifest = []

    for index in range(args.count):
        scenario = sample_conjunction(rng)
        tca_dt = creation_dt + timedelta(
            seconds=float(rng.uniform(3600.0, 72.0 * 3600.0))
        )
        message_id = f"SYNTH_{run_stamp}_{index:06d}"
        output = out_dir / f"{message_id}.cdm"
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite {output}")

        atomic_write(output, build_cdm_text(
            scenario, message_id, creation_dt, tca_dt
        ))
        pc = scenario["pc_result"]
        manifest.append({
            "file": output.name,
            "sha256": sha256(output),
            "message_id": message_id,
            "creation_date_utc": creation_dt.isoformat(),
            "tca_utc": tca_dt.isoformat(),
            "state_epoch": "TCA",
            "synthetic": True,
            "validation_status": "NOT_INDEPENDENTLY_VALIDATED",
            "pc_method": "local_pc2d_foster.calculate_pc_foster",
            "pc_foster": pc["Pc"],
            "log_pc_foster": pc["LogPc"],
            "hbr_km": scenario["hbr_km"],
            "is_remediated": pc["IsRemediated"],
            "integration_converged": pc["IntegrationConverged"],
            "integration_order": pc["IntegrationOrder"],
            "v3_features": pc["v3_features"],
            "synthetic_covariance_age_1_days": scenario["age1"],
            "synthetic_covariance_age_2_days": scenario["age2"],
            **scenario["quality"],
        })
        atomic_write(
            manifest_path,
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        )
        print(
            f"[{index + 1}/{args.count}] {output.name} "
            f"Pc={pc['Pc']:.6e} converged={pc['IntegrationConverged']} "
            f"order={pc['IntegrationOrder']}"
        )
        if args.interval_seconds and index + 1 < args.count:
            time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
