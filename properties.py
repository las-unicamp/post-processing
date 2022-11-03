from typing import List
import numpy as np
from metrics import Metrics
from vector_operations import (
    gradient_of_scalar,
    curl_of_vector,
    divergence_of_vector,
    divergence_of_tensor,
)


def compute_entropy_measure(
    pressure: np.ndarray,
    static_pressure: float,
    density: np.ndarray,
    reference_density: float,
    gamma: float,
) -> np.ndarray:
    return (pressure / static_pressure) / (density / reference_density) ** gamma - 1.0


def compute_strain_rate_tensor(
    velocity_x: np.ndarray,
    velocity_y: np.ndarray,
    reference_viscosity: float,
    metrics: Metrics,
) -> List[np.ndarray]:
    dudx, dudy = gradient_of_scalar(velocity_x, metrics)
    dvdx, dvdy = gradient_of_scalar(velocity_y, metrics)

    div = dudx + dvdy
    tauxx = reference_viscosity * (dudx + dudx - 2.0 / 3.0 * div)
    tauxy = reference_viscosity * (dudy + dvdx)
    tauyy = reference_viscosity * (dvdy + dvdy - 2.0 / 3.0 * div)

    return tauxx, tauxy, tauyy


def compute_shear_stress(
    velocity_x: np.ndarray,
    velocity_y: np.ndarray,
    reference_viscosity: float,
    metrics: Metrics,
) -> np.ndarray:
    _, txy, _ = compute_strain_rate_tensor(
        velocity_x, velocity_y, reference_viscosity, metrics
    )
    return txy


def compute_volumetric_dilatation(
    velocity_x: np.ndarray, velocity_y: np.ndarray, metrics: Metrics
) -> np.ndarray:
    return divergence_of_vector(velocity_x, velocity_y, metrics)


def compute_z_vorticity(
    velocity_x: np.ndarray, velocity_y: np.ndarray, metrics: Metrics
) -> np.ndarray:
    return curl_of_vector(velocity_x, velocity_y, metrics)


def compute_z_vorticity_volumetric_expansion(
    velocity_x: np.ndarray, velocity_y: np.ndarray, metrics: Metrics
) -> np.ndarray:
    return compute_z_vorticity(
        velocity_x, velocity_y, metrics
    ) * compute_volumetric_dilatation(velocity_x, velocity_y, metrics)


def compute_z_vorticity_diffusion(
    velocity_x: np.ndarray,
    velocity_y: np.ndarray,
    density: np.ndarray,
    reference_viscosity: float,
    metrics: Metrics,
) -> np.ndarray:
    dwdx, dwdy = gradient_of_scalar(
        compute_z_vorticity(velocity_x, velocity_y, metrics), metrics
    )
    dwdx, _ = gradient_of_scalar(dwdx, metrics)
    _, dwdy = gradient_of_scalar(dwdy, metrics)

    return reference_viscosity / density * (dwdx + dwdy)


def compute_z_vortex_transport(
    velocity_x: np.ndarray, velocity_y: np.ndarray, metrics: Metrics
) -> np.ndarray:
    dwzdx, dwzdy = gradient_of_scalar(
        compute_z_vorticity(velocity_x, velocity_y, metrics), metrics
    )

    return -(velocity_x * dwzdx + velocity_y * dwzdy)


def compute_baroclinicity(pressure: np.ndarray, density: np.ndarray, metrics: Metrics):
    drdx, drdy = gradient_of_scalar(1.0 / density, metrics)
    dpdx, dpdy = gradient_of_scalar(pressure, metrics)

    return -(drdx * dpdy - drdy * dpdx)


def compute_shear_stress_density_gradient(
    velocity_x: np.ndarray,
    velocity_y: np.ndarray,
    density: np.ndarray,
    reference_viscosity: float,
    metrics: Metrics,
) -> np.ndarray:
    txx, txy, tyy = compute_strain_rate_tensor(
        velocity_x, velocity_y, reference_viscosity, metrics
    )
    dtx, dty = divergence_of_tensor(txx, txy, txy, tyy, metrics)
    drdx, drdy = gradient_of_scalar(1.0 / density, metrics)

    return drdx * dty - drdy * dtx
