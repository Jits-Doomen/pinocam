"""Planar Gaussian collision probability (Pc) using numerical quadrature.

Research implementation; not certified operational software.

Units:
    position: km
    velocity: km/s
    position covariance: km^2
    HBR: km

The implementation projects the combined position covariance into the
encounter plane and integrates the resulting 2-D Gaussian over the hard-body
region. The supplied covariance must already describe uncertainty at TCA.
"""

from enum import Enum
from functools import lru_cache

import numpy as np


class HBRType(Enum):
    CIRCLE = 1
    SQUARE = 2
    SQUARE_EQU_AREA = 3


_EPS = 1e-12


@lru_cache(maxsize=12)
def _legendre_rule(order: int):
    return np.polynomial.legendre.leggauss(order)


def _vector(name, value):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite 3-vector.")
    return value


def _covariance(name, value):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (6, 6) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite 6x6 covariance.")
    if not np.allclose(value, value.T, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{name} must be symmetric.")
    return 0.5 * (value + value.T)


def ric_to_eci_transform(r_eci, v_eci):
    """Return a matrix with RIC unit vectors as columns."""
    r = _vector("r_eci", r_eci)
    v = _vector("v_eci", v_eci)
    r_norm = np.linalg.norm(r)
    h = np.cross(r, v)
    h_norm = np.linalg.norm(h)
    if r_norm < _EPS or h_norm < _EPS:
        raise ValueError("Degenerate state in RIC transformation.")
    radial = r / r_norm
    cross_track = h / h_norm
    in_track = np.cross(cross_track, radial)
    return np.column_stack((radial, in_track, cross_track))


def estimate_sgp4_covariance(r_eci, v_eci, age_days=0.0):
    """Illustrative covariance heuristic; NOT an empirically calibrated SGP4 model."""
    age_days = float(age_days)
    if not np.isfinite(age_days) or age_days < 0:
        raise ValueError("age_days must be finite and non-negative.")

    sigmas = np.array([0.1, 0.8 + 0.4 * age_days, 0.1])
    cov_pos_ric = np.diag(sigmas ** 2)
    cov_vel_ric = np.diag([1e-6, 1e-5, 1e-6])
    rotation = ric_to_eci_transform(r_eci, v_eci)

    cov = np.zeros((6, 6), dtype=np.float64)
    cov[:3, :3] = rotation @ cov_pos_ric @ rotation.T
    cov[3:, 3:] = rotation @ cov_vel_ric @ rotation.T
    return cov


def _log_weighted_sum(log_values, weights):
    valid = (weights > 0.0) & np.isfinite(log_values)
    if not np.any(valid):
        return -np.inf
    values = log_values[valid]
    w = weights[valid]
    peak = float(np.max(values))
    total = float(np.sum(w * np.exp(values - peak)))
    if total <= 0.0 or not np.isfinite(total):
        return -np.inf
    return peak + np.log(total)


def _integrate_log_pc(x0, z0, sx2, sz2, hbr, hbr_type, order):
    nodes, weights = _legendre_rule(int(order))

    if hbr_type == HBRType.CIRCLE:
        radii = 0.5 * hbr * (nodes + 1.0)
        angles = np.pi * (nodes + 1.0)
        rr, pp = np.meshgrid(radii, angles, indexing="ij")
        x = x0 + rr * np.cos(pp)
        z = z0 + rr * np.sin(pp)
        log_density = -0.5 * (x * x / sx2 + z * z / sz2)
        q_weights = (
            (0.5 * hbr * weights)[:, None]
            * (np.pi * weights)[None, :]
            * rr
        )
    else:
        half = hbr if hbr_type == HBRType.SQUARE else np.sqrt(np.pi) * hbr / 2.0
        x = x0 + half * nodes
        z = z0 + half * nodes
        xx, zz = np.meshgrid(x, z, indexing="ij")
        log_density = -0.5 * (xx * xx / sx2 + zz * zz / sz2)
        q_weights = (half * weights)[:, None] * (half * weights)[None, :]

    log_integral = _log_weighted_sum(log_density, q_weights)
    if not np.isfinite(log_integral):
        return -np.inf
    log_norm = np.log(2.0 * np.pi) + 0.5 * (np.log(sx2) + np.log(sz2))
    return float(log_integral - log_norm)


def calculate_pc_foster(
    r1, v1, cov1, r2, v2, cov2,
    HBR=0.005,
    hbr_type=HBRType.CIRCLE,
    *,
    eigenvalue_floor_ratio=1e-4,
    integration_orders=(32, 64, 128, 256, 512),
    integration_log_tolerance=1e-7,
):
    """Calculate 2-D Gaussian Pc in the encounter plane.

    Position covariance blocks are assumed to be expressed at the encounter
    epoch. Velocity covariance and position-velocity cross terms are not
    propagated by this function.
    """
    try:
        r1, v1, r2, v2 = (
            _vector("r1", r1), _vector("v1", v1),
            _vector("r2", r2), _vector("v2", v2),
        )
        cov1 = _covariance("cov1", cov1)
        cov2 = _covariance("cov2", cov2)
    except (TypeError, ValueError) as exc:
        return {
            "Pc": np.nan, "LogPc": np.nan, "IsRemediated": False,
            "IsPosDef": False, "IntegrationConverged": False,
            "Reason": f"INVALID_INPUT: {exc}",
        }

    HBR = float(HBR)
    floor_ratio = float(eigenvalue_floor_ratio)
    if not np.isfinite(HBR) or HBR <= 0:
        reason = "HBR_MUST_BE_FINITE_AND_POSITIVE"
    elif not isinstance(hbr_type, HBRType):
        reason = "INVALID_HBR_TYPE"
    elif not np.isfinite(floor_ratio) or floor_ratio < 0:
        reason = "INVALID_EIGENVALUE_FLOOR_RATIO"
    elif not integration_orders or any(
        not isinstance(n, (int, np.integer)) or n < 8 for n in integration_orders
    ):
        reason = "INVALID_INTEGRATION_ORDERS"
    else:
        reason = None

    if reason:
        return {
            "Pc": np.nan, "LogPc": np.nan, "IsRemediated": False,
            "IsPosDef": False, "IntegrationConverged": False,
            "Reason": reason,
        }

    rel_r = r1 - r2
    rel_v = v1 - v2
    speed = float(np.linalg.norm(rel_v))
    h = np.cross(rel_r, rel_v)
    h_norm = float(np.linalg.norm(h))
    if speed < _EPS or h_norm < _EPS:
        reason = "ZERO_RELATIVE_VELOCITY" if speed < _EPS else "COLINEAR_STATE_VECTORS"
        return {
            "Pc": np.nan, "LogPc": np.nan, "IsRemediated": False,
            "IsPosDef": False, "IntegrationConverged": False,
            "Reason": reason,
        }

    y_hat = rel_v / speed
    z_hat = h / h_norm
    x_hat = np.cross(y_hat, z_hat)
    encounter = np.vstack((x_hat, y_hat, z_hat))

    combined = cov1[:3, :3] + cov2[:3, :3]
    projected = encounter @ combined @ encounter.T
    plane = projected[np.ix_([0, 2], [0, 2])]
    plane = 0.5 * (plane + plane.T)

    try:
        eigenvalues, eigenvectors = np.linalg.eigh(plane)
    except np.linalg.LinAlgError:
        return {
            "Pc": np.nan, "LogPc": np.nan, "IsRemediated": False,
            "IsPosDef": False, "IntegrationConverged": False,
            "Reason": "EIGENDECOMP_FAILED",
        }

    scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)
    if float(np.min(eigenvalues)) < -1e-12 * scale:
        return {
            "Pc": np.nan, "LogPc": np.nan, "IsRemediated": False,
            "IsPosDef": False, "IntegrationConverged": False,
            "Reason": "NEGATIVE_ENCOUNTER_COVARIANCE_EIGENVALUE",
        }

    floor = (floor_ratio * HBR) ** 2
    fixed_eigenvalues = np.maximum(eigenvalues, floor)
    remediated = bool(np.any(fixed_eigenvalues > eigenvalues))
    if np.any(fixed_eigenvalues <= 0):
        return {
            "Pc": np.nan, "LogPc": np.nan, "IsRemediated": remediated,
            "IsPosDef": False, "IntegrationConverged": False,
            "Reason": "NON_POSITIVE_DEFINITE_AFTER_REMEDIATION",
        }

    mean_encounter = np.array([np.dot(rel_r, x_hat), np.dot(rel_r, z_hat)])
    mean_principal = eigenvectors.T @ mean_encounter
    x0, z0 = map(float, mean_principal)
    sx2, sz2 = map(float, fixed_eigenvalues)

    previous = None
    log_pc = -np.inf
    converged = False
    used_order = None
    for order in integration_orders:
        log_pc = _integrate_log_pc(x0, z0, sx2, sz2, HBR, hbr_type, int(order))
        used_order = int(order)
        if previous is not None:
            if np.isneginf(log_pc) and np.isneginf(previous):
                converged = True
                break
            if np.isfinite(log_pc) and np.isfinite(previous):
                if abs(log_pc - previous) <= integration_log_tolerance:
                    converged = True
                    break
        previous = log_pc

    if np.isnan(log_pc):
        pc = np.nan
    elif log_pc < np.log(np.finfo(float).tiny):
        pc = 0.0
    else:
        pc = float(np.exp(log_pc))

    clipped = bool(np.isfinite(pc) and pc > 1.0)
    if clipped:
        pc = 1.0

    return {
        "Pc": pc,
        "LogPc": float(log_pc),
        "IsRemediated": remediated,
        "IsPosDef": True,
        "IntegrationConverged": converged,
        "IntegrationOrder": used_order,
        "ProbabilityClipped": clipped,
        "EncounterPlaneMissDistance_km": float(np.linalg.norm(mean_encounter)),
        "v3_features": [x0, z0, float(np.sqrt(sx2)), float(np.sqrt(sz2)), HBR],
        "PrincipalAxisEigenvalues_km2": [sx2, sz2],
        "Reason": None if converged else "INTEGRATION_NOT_CONVERGED",
    }
