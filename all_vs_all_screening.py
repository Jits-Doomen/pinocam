import time
import requests
import numpy as np
import onnxruntime as ort

from pathlib import Path
from sgp4.api import Satrec, jday
from scipy.spatial import cKDTree
from datetime import datetime, timezone, timedelta

TLE_URL = (
    "https://celestrak.org/NORAD/elements/"
    "gp.php?GROUP=active&FORMAT=tle"
)

ONNX_Model_Path = "conjunction_pino.onnx"
TLE_Cache_File = Path("tle_cache.txt")
TLE_Cache_Max_Age = 3.5 * 60 * 60 

Screening_Hours = 24
Screening_Step_Min = 5
Screening_Radius_KM = 50.0

Sweep_Interval_Secs = 300

Earth_MU = 398600.4418 # in km^3/s^2
DT_Maneuver = 3600.0 # seconds

Pc_Threshold = 1e-5

SMD_Bar = 9.0

Base_Covariance = np.diag(
    [1.5, 5.0, 1.5, 0.01, 0.02, 0.01]
) ** 2

Matrix_Tolerance = 1e-12
Root_Imaginary_Tolerance = 1e-10

def get_tle_data(url=TLE_URL):

    print("Checking CelesTrak TLE cache...")

    tle_data = None

    # Check whether a previous successful download exists.
    if TLE_Cache_File.exists():

        cache_age = time.time() - TLE_Cache_File.stat().st_mtime

        if cache_age < TLE_Cache_Max_Age:

            print(
                f"Using cached TLE data "
                f"({cache_age / 3600:.2f} hours old)."
            )

            tle_data = TLE_Cache_File.read_text(
                encoding="utf-8"
            )

        else:

            print(
                f"TLE cache expired "
                f"({cache_age / 3600:.2f} hours old)."
            )

    # Download fresh data if the cache is missing or expired.
    if tle_data is None:

        print("Fetching TLE data from CelesTrak...")

        try:

            response = requests.get(
                url,
                timeout=30,
                headers={
                    "User-Agent": "PINO-Conjunction-Research/1.0"
                }
            )

            response.raise_for_status()

            tle_data = response.text

            # Save only after a successful download.
            TLE_Cache_File.write_text(
                tle_data,
                encoding="utf-8"
            )

            print("TLE cache successfully updated.")

        except requests.RequestException as exc:

            # If downloading fails, fall back to existing data.
            if TLE_Cache_File.exists():

                print(
                    f"Download failed: {exc}\n"
                    "Falling back to previously cached TLE data."
                )

                tle_data = TLE_Cache_File.read_text(
                    encoding="utf-8"
                )

            else:
                raise RuntimeError(
                    "CelesTrak download failed and no cache exists."
                ) from exc

    # Parse TLE data.
    lines = [
        line.strip()
        for line in tle_data.splitlines()
        if line.strip()
    ]

    satellites = []
    names = []

    i = 0

    while i < len(lines):

        if (
                i + 2 < len(lines)
                and lines[i + 1].startswith("1 ")
                and lines[i + 2].startswith("2 ")
        ):

            name = lines[i]
            line1 = lines[i + 1]
            line2 = lines[i + 2]

            try:

                satellite = Satrec.twoline2rv(
                    line1,
                    line2
                )

                satellites.append(satellite)
                names.append(name)

            except Exception as exc:

                print(
                    f"Skipping invalid TLE for {name}: {exc}"
                )

            i += 3

        else:
            i += 1

    if not satellites:

        raise RuntimeError(
            "No valid satellites were loaded from CelesTrak."
        )

    print(
        f"Loaded {len(satellites)} active satellites."
    )

    return satellites, names



def compute_lvlh_to_eci(r, v):

## Now I construct the LVLH-to-ECI rotation matrix.
## LVLH convention:
## X = Radial
## Y = Along-track
## Z = Cross-track
## Columns of the returned matrix are the LVLH
## basis vectors expressed in ECI coordinates.


## Turns 1-d r and 1-d v into 1-d arrays with 3 floats
    r = np.asarray(r, dtype=float).reshape(3)
    v = np.asarray(v, dtype=float).reshape(3)

    r_mag = np.linalg.norm(r)

    if r_mag < Matrix_Tolerance:
        raise ValueError("Invalid position vector.")

    r_hat = r / r_mag

    h = np.cross(r, v)
    h_mag = np.linalg.norm(h)

    if h_mag < Matrix_Tolerance:
        raise ValueError("Invalid angular momentum vector.")

    h_hat = h / h_mag

    t_hat = np.cross(h_hat, r_hat)
    t_mag = np.linalg.norm(t_hat)

    if t_mag < Matrix_Tolerance:
        raise ValueError("Invalid along-track vector.")

    t_hat /= t_mag

    R_lvlh_to_eci = np.column_stack(
        (r_hat, t_hat, h_hat)
    )

    return R_lvlh_to_eci



def compute_cw_stm(n, dt):
    """
    Calculate the 6x6 Clohessy-Wiltshire STM.

    State convention:
        [x, y, z, vx, vy, vz]

    LVLH convention:
        x = radial
        y = along-track
        z = cross-track
    """

    if n <= 0 or not np.isfinite(n):
        raise ValueError("Mean motion must be positive.")

    nt = n * dt

    cos_nt = np.cos(nt)
    sin_nt = np.sin(nt)

    Phi = np.zeros((6, 6), dtype=float)

    Phi[0, 0] = 4.0 - 3.0 * cos_nt
    Phi[0, 3] = sin_nt / n
    Phi[0, 4] = 2.0 * (1.0 - cos_nt) / n

    Phi[1, 0] = 6.0 * (sin_nt - nt)
    Phi[1, 1] = 1.0
    Phi[1, 3] = 2.0 * (cos_nt - 1.0) / n
    Phi[1, 4] = (4.0 * sin_nt - 3.0 * nt) / n

    Phi[2, 2] = cos_nt
    Phi[2, 5] = sin_nt / n

    Phi[3, 0] = 3.0 * n * sin_nt
    Phi[3, 3] = cos_nt
    Phi[3, 4] = 2.0 * sin_nt

    Phi[4, 0] = 6.0 * n * (cos_nt - 1.0)
    Phi[4, 3] = -2.0 * sin_nt
    Phi[4, 4] = 4.0 * cos_nt - 3.0

    Phi[5, 2] = -n * sin_nt
    Phi[5, 5] = cos_nt

    return Phi


def calculate_bplane_parameters(
        r_rel,
        v_rel,
        cov_lvlh,
        R_lvlh_to_eci
):
    """
    Calculate the relative B-plane position and covariance.

    Assumptions:
        - Relative covariance is supplied in the primary
          satellite's LVLH frame.
        - The covariance is transformed into ECI.
        - The B-plane is perpendicular to relative velocity.

    Returns:
        bp: B-plane position (2x1)
        C: B-plane covariance (2x2)
        M_rot: ECI-to-B-plane projection (2x3)
    """

    r_rel = np.asarray(r_rel, dtype=float).reshape(3)
    v_rel = np.asarray(v_rel, dtype=float).reshape(3)

    cov_lvlh = np.asarray(
        cov_lvlh,
        dtype=float
    ).reshape(6, 6)

    R_lvlh_to_eci = np.asarray(
        R_lvlh_to_eci,
        dtype=float
    ).reshape(3, 3)

    v_mag = np.linalg.norm(v_rel)

    if v_mag < Matrix_Tolerance:
        raise ValueError("Relative velocity is too small.")

    h_vec = np.cross(r_rel, v_rel)
    h_mag = np.linalg.norm(h_vec)

    if h_mag < Matrix_Tolerance:
        raise ValueError("Invalid relative angular momentum.")

    t_hat = v_rel / v_mag
    n_hat = h_vec / h_mag
    r_hat = np.cross(t_hat, n_hat)

    M_rot = np.vstack(
        (n_hat, r_hat)
    )

    bp = (M_rot @ r_rel).reshape(2, 1)

    C_position_lvlh = cov_lvlh[:3, :3]

    C_position_eci = (
            R_lvlh_to_eci
            @ C_position_lvlh
            @ R_lvlh_to_eci.T
    )

    C_bplane = (
            M_rot
            @ C_position_eci
            @ M_rot.T
    )

    C_bplane = 0.5 * (
            C_bplane + C_bplane.T
    )

    if not np.all(np.isfinite(C_bplane)):
        raise ValueError("Non-finite B-plane covariance.")

    return bp, C_bplane, M_rot


def compute_eop_bplane_cam(
        bp,
        C,
        Phi12,
        SMD_Bar
):
    """
    Evaluate the existing EOP B-plane CAM polynomial.

    Inputs:
        bp       : B-plane position (2x1)
        C        : B-plane covariance (2x2)
        Phi12    : B-plane position/velocity transition (2x2)
        SMD_Bar  : EOP design parameter

    Returns:
        List of (root, bf, lambda_0) solutions.

    IMPORTANT!:
        This implementation preserves the supplied EOP
        polynomial. Its derivation, units, and optimality
        conditions require independent verification.
    """

    bp = np.asarray(
        bp,
        dtype=float
    ).reshape(2, 1)

    C = np.asarray(
        C,
        dtype=float
    ).reshape(2, 2)

    Phi12 = np.asarray(
        Phi12,
        dtype=float
    ).reshape(2, 2)

    if not np.all(np.isfinite(bp)):
        return []

    if not np.all(np.isfinite(C)):
        return []

    if not np.all(np.isfinite(Phi12)):
        return []

    try:

        Q = np.linalg.inv(C)

        F = 2.0 * (Phi12 @ Q)

        det_F = np.linalg.det(F)

        F_inv = np.linalg.inv(F)

        F_tilde = det_F * F_inv

    except np.linalg.LinAlgError:
        return []

    Fbp = F_tilde @ bp

    A = (Fbp.T @ Q @ Fbp).item()

    B = (
            bp.T @ Q @ Fbp
            +
            Fbp.T @ Q @ bp
    ).item()

    C_term = (bp.T @ Q @ bp).item()

    T = np.trace(F)
    D = det_F

    c4 = -SMD_Bar * (D ** 2)

    c3 = (
            SMD_Bar
            * 2.0
            * T
            * D
    )

    c2 = (
            A
            - SMD_Bar * (T ** 2 + 2.0 * D)
    )

    c1 = (
            -B
            + SMD_Bar * 2.0 * T
    )

    if C_term >= SMD_Bar:
        return []

    c0 = C_term - SMD_Bar

    coefficients = np.array(
        [c4, c3, c2, c1, c0],
        dtype=float
    )

    if not np.all(np.isfinite(coefficients)):
        return []

    first_nonzero = next(
        (
            i
            for i, value in enumerate(coefficients)
            if abs(value) > Matrix_Tolerance
        ),
        None
    )

    if first_nonzero is None:
        return []

    reduced_coefficients = coefficients[first_nonzero:]

    if len(reduced_coefficients) < 2:
        return []

    roots = np.roots(reduced_coefficients)

    valid_roots = [
        float(root.real)
        for root in roots
        if abs(root.imag) < Root_Imaginary_Tolerance
    ]

    solutions = []

    I = np.eye(2)

    for value in valid_roots:

        I_v_F = I - value * F

        try:

            bf = np.linalg.solve(
                I_v_F,
                bp
            )

            lambda_0 = (
                    value * 2.0 * Q @ bf
            )

            if (
                    np.all(np.isfinite(bf))
                    and np.all(np.isfinite(lambda_0))
            ):

                solutions.append(
                    (value, bf, lambda_0)
                )

        except np.linalg.LinAlgError:
            continue

    return solutions

def load_pino_model(model_path):

    print("Loading PINO ONNX model...")

    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"]
    )

    inputs = session.get_inputs()

    if len(inputs) != 1:
        raise RuntimeError(
            "Expected exactly one ONNX model input."
        )

    model_input = inputs[0]

    input_name = model_input.name

    input_shape = []

    for dimension in model_input.shape:

        if (
                isinstance(dimension, (int, np.integer))
                and dimension > 0
        ):
            input_shape.append(int(dimension))

        else:
            input_shape.append(1)

    input_size = int(np.prod(input_shape))

    if input_size < 5:
        raise RuntimeError(
            f"Model input has only {input_size} values. "
            "The screening pipeline requires at least five."
        )

    print(f"Model loaded: {model_path}")
    print(f"Input name: {input_name}")
    print(f"Input shape: {input_shape}")

    for output in session.get_outputs():

        print(
            f"Output: {output.name}, "
            f"shape: {output.shape}"
        )

    return session, input_name, input_shape


NORM_MEAN = np.array([
    3160.5038798938508,
    8000.9837136139968,
    2267.5591676062541,
    2268.3006677756493,
    14.475273831078239,
], dtype=np.float32)

NORM_STD = np.array([
    7844.3946831551939,
    4041.8483111051305,
    4196.3265286960677,
    4197.5587272294888,
    22.703363683868911,
], dtype=np.float32)

DEFAULT_HBR_M = 10.0


def predict_collision_probability(
        session,
        input_name,
        input_shape,
        distance,
        relative_speed,
        relative_position,
        norm_mean=NORM_MEAN,
        norm_std=NORM_STD,
        hbr_m=DEFAULT_HBR_M
):
    miss_distance_m = float(distance) * 1000.0
    rel_speed_m_s = float(relative_speed) * 1000.0

    pos_var_km2 = np.diag(Base_Covariance)[:3]
    sigma_km = float(np.sqrt(np.mean(pos_var_km2)))
    sigma_primary_m = sigma_km * 1000.0
    sigma_secondary_m = sigma_km * 1000.0

    features = np.array(
        [
            miss_distance_m,
            rel_speed_m_s,
            sigma_primary_m,
            sigma_secondary_m,
            float(hbr_m),
        ],
        dtype=np.float32
    )

    features = (features - norm_mean) / norm_std

    input_data = np.zeros(
        input_shape,
        dtype=np.float32
    )
    input_data.flat[:5] = features

    outputs = session.run(
        None,
        {
            input_name: input_data
        }
    )

    if not outputs:
        raise ValueError("ONNX model returned no outputs.")

    prediction = np.asarray(
        outputs[0],
        dtype=float
    ).squeeze()

    if prediction.size != 1:
        raise ValueError(
            f"Expected one scalar output, got shape "
            f"{np.asarray(outputs[0]).shape}"
        )

    log10_pc = float(prediction)

    if not np.isfinite(log10_pc):
        raise ValueError(
            "Model returned a non-finite log10 probability."
        )

    pc = float(np.clip(10.0 ** log10_pc, 0.0, 1.0))

    return pc

def calculate_semi_major_axis(r, v):

    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)

    if r_mag < Matrix_Tolerance:
        raise ValueError("Invalid orbital position.")

    specific_energy = (
            0.5 * v_mag ** 2
            - Earth_MU / r_mag
    )

    if abs(specific_energy) < Matrix_Tolerance:
        raise ValueError(
            "Orbital energy is too close to zero."
        )

    if specific_energy >= 0:
        raise ValueError(
            "Non-elliptical orbit; CW model is not applicable."
        )

    return -Earth_MU / (2.0 * specific_energy)

def main():

    session, input_name, input_shape = load_pino_model(
        ONNX_Model_Path
    
    )

    print("\nStarting conjunction screening pipeline.")

    while True:

        try:

            sats, names = get_tle_data()

        except Exception as exc:

            print(
                f"Unable to retrieve TLE data: {exc}"
            )

            print(
                f"Retrying in {Sweep_Interval_Secs} seconds."
            )

            time.sleep(Sweep_Interval_Secs)
            continue

        now = datetime.now(timezone.utc)

        print(
            f"\n[{now.strftime('%Y-%m-%d %H:%M:%S UTC')}] "
            f"Starting {Screening_Hours}-hour conjunction sweep..."
        )

        alerts_found = 0
        inference_errors = 0
        cam_errors = 0

        total_minutes = Screening_Hours * 60

        for minutes_ahead in range(
                0,
                total_minutes,
                Screening_Step_Min
        ):

            eval_time = now + timedelta(
                minutes=minutes_ahead
            )

            jd, fr = jday(
                eval_time.year,
                eval_time.month,
                eval_time.day,
                eval_time.hour,
                eval_time.minute,
                eval_time.second
            )

            positions = []
            velocities = []
            valid_idx = []

            for idx, sat in enumerate(sats):

                error, r, v = sat.sgp4(jd, fr)

                if error == 0:

                    positions.append(r)
                    velocities.append(v)

                    valid_idx.append(idx)

            if not positions:
                continue

            positions = np.asarray(
                positions,
                dtype=float
            )

            velocities = np.asarray(
                velocities,
                dtype=float
            )

            tree = cKDTree(positions)

            pairs = tree.query_pairs(
                r=Screening_Radius_KM,
                output_type="ndarray"
            )

            if len(pairs) == 0:
                continue

            for i, j in pairs:

                p1 = positions[i]
                p2 = positions[j]

                v1 = velocities[i]
                v2 = velocities[j]

                rel_pos = p1 - p2
                rel_vel = v1 - v2

                dist = np.linalg.norm(rel_pos)

                relative_speed = np.linalg.norm(
                    rel_vel
                )

                if relative_speed < 0.05:
                    continue

                try:

                    pc = predict_collision_probability(
                        session,
                        input_name,
                        input_shape,
                        dist,
                        relative_speed,
                        rel_pos
                    )

                except Exception as exc:

                    inference_errors += 1

                    if inference_errors <= 10:

                        print(
                            f"ONNX inference error: {exc}"
                        )

                    continue

                if pc <= Pc_Threshold and dist >= 5.0:
                    continue

                alerts_found += 1

                n1 = names[valid_idx[i]]
                n2 = names[valid_idx[j]]

                t_str = eval_time.strftime(
                    "%Y-%m-%d %H:%M:%S UTC"
                )

                print(
                    f"\n[{t_str}] Conjunction Alert: "
                    f"{n1} vs {n2}"
                )

                print(
                    f"   Miss Distance: {dist:.3f} km"
                )

                print(
                    f"   Relative Speed: "
                    f"{relative_speed:.6f} km/s"
                )

                print(
                    f"   PINO Output: {pc:.6e}"
                )

                try:

                    R_lvlh_to_eci = compute_lvlh_to_eci(
                        p1,
                        v1
                    )

                    semi_major_axis = (
                        calculate_semi_major_axis(
                            p1,
                            v1
                        )
                    )

                    n_mean = np.sqrt(
                        Earth_MU /
                        semi_major_axis ** 3
                    )

                    Phi = compute_cw_stm(
                        n_mean,
                        DT_Maneuver
                    )

                    Phi12 = Phi[:3, 3:]

                    Phi12_eci = (
                            R_lvlh_to_eci
                            @ Phi12
                            @ R_lvlh_to_eci.T
                    )

                    bp, C_bplane, M_rot = (
                        calculate_bplane_parameters(
                            rel_pos,
                            rel_vel,
                            Base_Covariance,
                            R_lvlh_to_eci
                        )
                    )
                    M_sens = M_rot @ R_lvlh_to_eci @ Phi12
                    Phi12_bplane = M_sens @ M_sens.T

                    solutions = compute_eop_bplane_cam(
                        bp,
                        C_bplane,
                        Phi12_bplane,
                        SMD_Bar
                    )

                    best_dv = float("inf")
                    best_lambda = None

                    for solution in solutions:

                        nu, bf, lambda_0 = solution

                        dv_proxy = (
                                np.linalg.norm(lambda_0) / 2.0
                        )

                        if dv_proxy < best_dv:

                            best_dv = dv_proxy
                            best_lambda = lambda_0

                    if best_lambda is not None:

                        print(
                            "   EOP CAM candidate found."
                        )

                        print(
                            "   Nominal delta-v proxy: "
                            f"{best_dv:.8e}"
                        )

                        print(
                            "   Initial Costate Lambda_0: "
                            f"[{best_lambda[0, 0]:.6e}, "
                            f"{best_lambda[1, 0]:.6e}]"
                        )

                    else:

                        print(
                            "   EOP CAM: No maneuver required (SMD >= threshold)."
                        )

                except Exception as exc:

                    cam_errors += 1

                    print(
                        f"   B-plane/CAM calculation failed: {exc}"
                    )

                print("-" * 65)

        print("\n" + "=" * 65)

        print("Sweep Sweep")

        print(
            f"Potential conjunction alerts: {alerts_found}"
        )

        print(
            f"ONNX inference errors: {inference_errors}"
        )

        print(
            f"B-plane/CAM errors: {cam_errors}"
        )

        print("=" * 65)

        print(
            f"Next sweep in {Sweep_Interval_Secs} seconds."
        )

        time.sleep(Sweep_Interval_Secs)

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print("\nScreening stopped by user.")

    except Exception as exc:

        print(
            f"\nFatal screening error: {exc}"
        )

        raise
