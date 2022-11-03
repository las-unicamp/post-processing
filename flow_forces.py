from typing import List
import numpy as np
from metrics import Metrics
from properties import compute_strain_rate_tensor
from vector_operations import gradient_of_scalar, divergence_of_tensor


def compute_coriolis_force():
    raise NotImplementedError("need to implement this...")


def compute_centrifugal_force():
    raise NotImplementedError("need to implement this...")


def compute_euler_force():
    raise NotImplementedError("need to implement this...")


def compute_pressure_force(pressure: np.ndarray, metrics: Metrics):
    dpdx, dpdy = gradient_of_scalar(pressure, metrics)
    return dpdx, dpdy


def compute_relative_convective_force():
    raise NotImplementedError("need to implement this...")


def compute_shear_force(
    relative_velocity_x: np.ndarray,
    relative_velocity_y: np.ndarray,
    reference_viscosity: float,
    metrics: Metrics,
) -> List[np.ndarray]:
    txx, txy, tyy = compute_strain_rate_tensor(
        relative_velocity_x, relative_velocity_y, reference_viscosity, metrics
    )
    dtx, dty = divergence_of_tensor(txx, txy, txy, tyy, metrics)
    return dtx, dty


def compute_wall_pressure_forces(
    pressure: np.ndarray, reference_pressure: float, metrics: Metrics
) -> List[np.ndarray]:
    p = pressure - reference_pressure
    force_x = -0.5 * (p[:-1, 0] + p[1:, 0]) * metrics.normal_x[:-1, 0]
    force_y = -0.5 * (p[:-1, 0] + p[1:, 0]) * metrics.normal_y[:-1, 0]

    return force_x, force_y


def compute_wall_viscous_forces(
    velocity_x: np.ndarray,
    velocity_y: np.ndarray,
    reference_viscosity: float,
    metrics: Metrics,
) -> List[np.ndarray]:
    tauxx, tauxy, tauyy = compute_strain_rate_tensor(
        velocity_x, velocity_y, reference_viscosity, metrics
    )

    force_x = (
        0.5 * (tauxx[1:, 0] + tauxx[:-1, 0]) * metrics.normal_x[:-1, 0]
        + 0.5 * (tauxy[1:, 0] + tauxy[:-1, 0]) * metrics.normal_y[:-1, 0]
    )
    force_y = (
        0.5 * (tauyy[1:, 0] + tauyy[:-1, 0]) * metrics.normal_y[:-1, 0]
        + 0.5 * (tauxy[1:, 0] + tauxy[:-1, 0]) * metrics.normal_x[:-1, 0]
    )

    return force_x, force_y
