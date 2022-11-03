# pylint: disable=invalid-name

from typing import Tuple
import numpy as np
from numpy.typing import NDArray
from metrics import Metrics


def contravariant_from_cartesian(
    vector_x: NDArray, vector_y: NDArray, metrics: Metrics
) -> Tuple[NDArray, NDArray]:
    component_1 = vector_x * metrics.dAdx + vector_y * metrics.dAdy
    component_2 = vector_x * metrics.dBdx + vector_y * metrics.dBdy

    return component_1, component_2


def tangential_component_of_vector(
    vector_x: NDArray, vector_y: NDArray, metrics: Metrics
) -> NDArray:
    """
    Compute the physical tangential component of a Cartesian vector.
    Remember that the covariant basis is tangent to the curvilinear coordinates.
    """
    covariant_component_in_direction_A = (
        vector_x * metrics.dxdA + vector_y * metrics.dydA
    )
    return covariant_component_in_direction_A / metrics.h1


def normal_component_of_vector(
    vector_x: NDArray, vector_y: NDArray, metrics: Metrics
) -> NDArray:
    """
    Compute the physical normal component of a Cartesian vector
    Remember that the contravariant basis is normal to the curvilinear coordinates.
    """
    contravariant_component_in_direction_B = (
        vector_x * metrics.dBdx + vector_y * metrics.dBdy
    )
    length_of_basis_vector = np.sqrt(metrics.dBdx**2 + metrics.dBdy**2)
    return contravariant_component_in_direction_B / length_of_basis_vector


def gradient_of_scalar(scalar: NDArray, metrics: Metrics) -> Tuple[NDArray, NDArray]:
    dsdA, dsdB = np.gradient(scalar, edge_order=2)

    dsdx = dsdA * metrics.dAdx + dsdB * metrics.dBdx
    dsdy = dsdA * metrics.dAdy + dsdB * metrics.dBdy

    return dsdx, dsdy


def gradient_of_vector(
    vector_x: NDArray, vector_y: NDArray, metrics: Metrics
) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    dvxdx, dvxdy = gradient_of_scalar(vector_x, metrics)
    dvydx, dvydy = gradient_of_scalar(vector_y, metrics)

    return dvxdx, dvxdy, dvydx, dvydy


def divergence_of_vector(
    vector_x: NDArray, vector_y: NDArray, metrics: Metrics
) -> NDArray:
    """
    Evaluates the divergence of a vector using the Voss-Weyl formula.
    The components of the input vector must be in the Cartesian system.
    """
    component_1, component_2 = contravariant_from_cartesian(vector_x, vector_y, metrics)

    part1 = np.gradient(component_1 * metrics.jacobian, edge_order=2)
    part2 = np.gradient(component_2 * metrics.jacobian, edge_order=2)
    return (part1 + part2) / metrics.jacobian


def divergence_of_tensor(
    tensor_xx: NDArray,
    tensor_xy: NDArray,
    tensor_yx: NDArray,
    tensor_yy: NDArray,
    metrics: Metrics,
) -> Tuple[NDArray, NDArray]:
    component_x = divergence_of_vector(tensor_xx, tensor_xy, metrics)
    component_y = divergence_of_vector(tensor_yx, tensor_yy, metrics)

    return component_x, component_y


def tangential_and_normal_derivatives_of_scalar(
    scalar: NDArray, metrics: Metrics
) -> Tuple[NDArray, NDArray]:
    """
    Compute the (physical) tangential and normal derivatives of scalar
    See formula in Aris book: page 155, exercise 7.41.2
    """
    dsdA, dsdB = np.gradient(scalar, edge_order=2)
    dsdA_physical = dsdA / metrics.h1
    dsdB_physical = dsdB / metrics.h2

    return dsdA_physical, dsdB_physical


def curl_of_vector(vector_x: NDArray, vector_y: NDArray, metrics: Metrics) -> NDArray:
    _, dvxdy = gradient_of_scalar(vector_x, metrics)
    dvydx, _ = gradient_of_scalar(vector_y, metrics)

    return dvydx - dvxdy
