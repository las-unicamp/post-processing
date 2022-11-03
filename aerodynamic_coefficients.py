import numpy as np
from numpy.typing import NDArray
from metrics import Metrics
from vector_operations import (
    tangential_component_of_vector,
    tangential_and_normal_derivatives_of_scalar,
)


def compute_circulation(
    velocity_x: NDArray, velocity_y: NDArray, j: int, metrics: Metrics
):
    """
    Compute the airfoil circulation.

    Remember that due to the non-slip boundary condition, one needs to specify
    a region away from the airfoil surface to define the line integral. Since
    we are using a O-type mesh, it suffices to specify a j index for that.
    """

    dl_vector = (metrics.dx[:, j], metrics.dy[:, j])
    velocity_vector = (velocity_x[:, j], velocity_y[:, j])
    circulation = np.einsum("ij,ij", velocity_vector, dl_vector)
    return circulation


# def test_circulation():
#     vel_x = [1, 1, 2]
#     vel_y = [0, 2, 1]

#     dx = [1, 0.5, 0.5]
#     dy = [0.5, 1, 1]

#     dl = (dx, dy)
#     vel = (vel_x, vel_y)

#     circulation = np.einsum("ij,ij", vel, dl)

#     print(circulation)


# if __name__ == "__main__":
#     test_circulation()


def compute_lift_and_drag_coefficients(
    wall_pressure_forces: NDArray,
    wall_viscous_forces: NDArray,
    reference_density: float,
    reference_velocity: float,
) -> float:
    """
    Compute lift and drag coefficients

    Args:
        wall_pressure_forces (NDArray):
    """
    total_force_contribution = wall_pressure_forces + wall_viscous_forces

    coefficients = np.sum(total_force_contribution, axis=1) / (
        0.5 * reference_density * reference_velocity**2
    )

    lift = coefficients[1]
    drag = coefficients[0]

    return lift, drag


def compute_pitch_moment_coefficient(
    wall_pressure_forces: NDArray,
    wall_viscous_forces: NDArray,
    reference_density: float,
    reference_velocity: float,
    metrics: Metrics,
    pivot_point: NDArray,
) -> float:
    x_lever = metrics.x[:-1, 0] - pivot_point[0]
    y_lever = metrics.y[:-1, 0] - pivot_point[1]

    forces_x = wall_pressure_forces[0] + wall_viscous_forces[0]
    forces_y = wall_pressure_forces[1] + wall_viscous_forces[1]

    return sum(forces_y * x_lever + forces_x * y_lever) / (
        0.5 * reference_density * reference_velocity**2
    )


def compute_skin_friction_coefficient(
    velocity_x: NDArray,
    velocity_y: NDArray,
    reference_density: float,
    reference_viscosity: float,
    reference_velocity: float,
    metrics: Metrics,
) -> NDArray:
    tangential_velocity = tangential_component_of_vector(
        velocity_x, velocity_y, metrics
    )

    _, normal_derivative = tangential_and_normal_derivatives_of_scalar(
        tangential_velocity, metrics
    )

    normal_derivative_at_the_wall = normal_derivative[:, 0]

    return (
        normal_derivative_at_the_wall
        * reference_viscosity
        / (0.5 * reference_density * reference_velocity**2)
    )


def compute_pressure_coefficient(
    pressure, static_pressure, reference_density, reference_velocity
):
    """Evaluate pressure coefficient"""
    return (pressure - static_pressure) / (
        0.5 * reference_density * reference_velocity**2
    )
