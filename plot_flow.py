from typing import Any
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


IMAGE_WIDTH_IN_PIXELS = 600
IMAGE_HEIGHT_IN_PIXELS = 600
DPI = 96

WIDTH = IMAGE_WIDTH_IN_PIXELS / DPI


def plot_flow(
    axis: matplotlib.axes.Axes,
    variable: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    levels: np.ndarray,
    norm: matplotlib.cm.colors.Normalize,
    cmap: Any,
    xyranges: np.ndarray,
    outfile_name: str,
) -> None:
    # plot flow field
    axis.contourf(x, y, variable, levels, norm=norm, cmap=cmap, extend="both")

    # plot the airfoil
    airfoil = [[x[_, 0], y[_, 0]] for _ in range(x.shape[0])]
    airfoil = np.array(airfoil)
    hull = ConvexHull(airfoil)
    axis.fill(
        airfoil[hull.vertices, 0],
        airfoil[hull.vertices, 1],
        color="#bfbfbf",
        linewidth=0.2,
    )

    # adjust axis
    axis.set_xlim(*xyranges[0])
    axis.set_ylim(*xyranges[1])
    axis.axis("off")
    axis.set_aspect("equal")

    # save image
    plt.savefig(outfile_name, dpi=DPI)
