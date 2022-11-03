from typing import List, Tuple
import numpy as np
from metrics import Metrics


def get_indices_of_airfoil_limits(
    x: np.ndarray, y: np.ndarray, angle: float
) -> Tuple[int]:
    """
    Given the airfoil surface points x and y, this function returns the indices
    of the leading- and trailing-edges.

    Args:
        x (list[float]): x-components of the surface nodes;
        y (list[float]): y-components of the surface nodes;
        angle (float): Airfoil static angle with respect to the x-axis;

    Return:
        Indexes of the points over the airfoil suction side and pressure side.
    """
    straight_airfoil_x = np.cos(np.deg2rad(angle)) * x - np.sin(np.deg2rad(angle)) * y

    leading_edge_index = np.argmin(straight_airfoil_x)
    trailing_edge_index = np.argmax(straight_airfoil_x)

    return leading_edge_index, trailing_edge_index


def get_airfoil_suction_and_pressure_side_indices(
    x: np.ndarray, y: np.ndarray, angle: float, metrics: Metrics
) -> Tuple[List[int], List[int]]:
    """
    Given the x and y coordinate points of the airfoil surface, and its angle
    of attack (with respect to the x-axis) this method returns the indices of
    the points located over the the airfoil suction side and pressure side.

    Note that depending on the orientation of the point distribution, one can
    have different indices along the airfoil suction side. Hence, some metric
    terms are needed to check the orientation.

    OBS: Here it is considered that the flow points towards the
    positive x-direction. ---> (from left to right)

    Args:
        x (list[float]): x-components of the surface nodes;
        y (list[float]): y-components of the surface nodes;
        angle (float): Airfoil static angle with respect to the x-axis;

    Return:
        Indexes of the points over the airfoil suction side and pressure side.
    """
    leading_edge_index, trailing_edge_index = get_indices_of_airfoil_limits(x, y, angle)

    leading_edge_index_comes_first = leading_edge_index < trailing_edge_index

    if metrics.orientation == "counterclockwise":
        if leading_edge_index_comes_first:
            suction_side_indices = list(
                range(leading_edge_index, trailing_edge_index + 1)
            )
        else:
            suction_side_indices = list(range(leading_edge_index, len(x) - 1))
            suction_side_indices += list(range(trailing_edge_index + 1))

    if metrics.orientation == "clockwise":
        if leading_edge_index_comes_first:
            suction_side_indices = list(range(leading_edge_index))
            suction_side_indices += list(range(trailing_edge_index, len(x - 1)))
        else:
            suction_side_indices = list(
                range(trailing_edge_index, leading_edge_index + 1)
            )

    pressure_side_indices = [
        index for index in range(len(x)) if index not in suction_side_indices
    ]

    return suction_side_indices, pressure_side_indices
