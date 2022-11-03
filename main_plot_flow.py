import re
import os
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.gridspec import GridSpec
from scipy.spatial import ConvexHull

import texfig  # this import must be on top to configure Matplotlib's backend
from metrics import Metrics
from file_searching import search_for_files
from read_flow_cgns import read_flow_in_cgns
from read_grid_cgns import read_grid_in_cgns
from aerodynamic_coefficients import compute_pressure_coefficient
from properties import compute_entropy_measure
from cmaps import customMagma, transparentCmap


# PATH_TO_DIR = "/home/pclab/Desktop/M02_Re60k_span01_g480/proc_pitch_ramp_k010/output"
PATH_TO_DIR = "/media/miotto/3B712DB11C683E49/SD7003/M01_Re60k_span01_g480/proc_pitch_ramp_k005/output"


SELECTED_TIME = 4.0

X_LIMITS = [-0.01, 1.05]
Y_LIMITS = [-0.2, 0.35]

PIVOT_POINT = [0.25 * np.cos(np.deg2rad(8)), -0.25 * np.sin(np.deg2rad(8))]


MACH_NUMBER = 0.1
REYNOLDS_NUMBER = 6e4

GAMMA = 1.4
REFERENCE_DENSITY = 1.0
REFERENCE_VISCOSITY = 1.0 / REYNOLDS_NUMBER
STATIC_PRESSURE = 1.0 / GAMMA
SIMULATION_INITIAL_TIME = 0.0

OUTPUT_FIGURE_RATIO = 0.34
NUM_ROWS = 2
NUM_COLS = 4


def get_closest_value_in_array(array: np.ndarray, target: float) -> int:
    closest = min(array, key=lambda x: abs(x - target))
    return np.where(array == closest)[0][0]


def get_simulation_dt(simulation_files):
    _, time1 = read_flow_in_cgns(simulation_files[0])
    _, time2 = read_flow_in_cgns(simulation_files[1])
    return time2 - time1


def main():
    qout_files = search_for_files(PATH_TO_DIR, pattern="qout2Davg*.cgns")
    grid_file = search_for_files(PATH_TO_DIR, pattern="grid2D.cgns")[0]

    x, y = read_grid_in_cgns(grid_file)

    # metrics = Metrics(x, y)

    simulation_dt = get_simulation_dt(qout_files)
    timestamps = SIMULATION_INITIAL_TIME + np.arange(len(qout_files)) * simulation_dt

    closest = get_closest_value_in_array(timestamps, SELECTED_TIME)

    selected_qout = qout_files[closest]
    print(f"Opening file {os.path.split(selected_qout)[1]}")

    fig = texfig.figure(ratio=OUTPUT_FIGURE_RATIO)
    grid = GridSpec(NUM_ROWS, NUM_COLS, left=0.0, right=1.0, wspace=0.01, hspace=0.01)

    cmap_solid = customMagma()
    cmap_transparent = transparentCmap(
        color="#63599e", alpha=0.35
    )  # Nice colors to choose from: #63599e #6c62a4 #63599e #6272a4

    vmin, vmax = -4.0, 0.0
    norm = matplotlib.cm.colors.Normalize(vmax=vmax, vmin=vmin)
    levels = np.linspace(vmin, vmax, 512)

    vmin_entropy, vmax_entropy = 0.05, 0.25
    norm_entropy = matplotlib.cm.colors.Normalize(vmax=vmax_entropy, vmin=vmin_entropy)
    levels_entropy = np.linspace(vmin_entropy, vmax_entropy, 13)

    q_vector, time = read_flow_in_cgns(selected_qout)

    print(f"target time: {SELECTED_TIME}")
    print(f"Simulation time: {time}")

    pressure = q_vector[3]
    pressure_coeff = compute_pressure_coefficient(
        pressure, STATIC_PRESSURE, REFERENCE_DENSITY, MACH_NUMBER
    )

    entropy = compute_entropy_measure(
        pressure, STATIC_PRESSURE, q_vector[0], REFERENCE_DENSITY, GAMMA
    )

    for index_row in range(NUM_ROWS):
        for index_col in range(NUM_COLS):

            ax = plt.subplot(grid[index_row, index_col])

            contourf = ax.contourf(
                x,
                y,
                pressure_coeff,
                levels,
                norm=norm,
                cmap=cmap_solid,
                extend="both",
                zorder=-20,
            )

            ax.contourf(
                x,
                y,
                entropy,
                levels_entropy,
                norm=norm_entropy,
                cmap=cmap_transparent,
                extend="both",
                zorder=-20,
            )

            # plot the airfoil
            airfoil = [[x[_, 0], y[_, 0]] for _ in range(x.shape[0])]
            airfoil = np.array(airfoil)
            hull = ConvexHull(airfoil)
            ax.fill(
                airfoil[hull.vertices, 0],
                airfoil[hull.vertices, 1],
                color="#bfbfbf",
                linewidth=0.2,
            )

            ax.set_xlim(*X_LIMITS)
            ax.set_ylim(*Y_LIMITS)
            ax.axis("off")
            ax.set_aspect("equal")
            ax.set_rasterization_zorder(-10)

            # --------------------------------------------------
            #### Label subplots ####

            if index_row == 0:
                ax.text(
                    0.5,
                    1.07,
                    r"$\alpha_{eff} = $" + f" {SELECTED_TIME:.1f}",
                    transform=ax.transAxes,
                    ha="center",
                    size=10,
                )

            if index_col == 0 and index_row == 0:
                ax.text(
                    0.02,
                    0.8,
                    r"$s/c = 0.1$",
                    transform=ax.transAxes,
                    ha="left",
                    size=10,
                )

            if index_col == 0 and index_row == 1:
                ax.text(
                    0.02,
                    0.8,
                    r"$s/c = 0.4$",
                    transform=ax.transAxes,
                    ha="left",
                    size=10,
                )

    # --------------------------------------------------
    #### Colorbar ####

    numscale = 3  # Amount of numbers appearing in the colorbar

    # Use scientific notation
    fmt = matplotlib.ticker.ScalarFormatter(useMathText=True, useOffset=False)
    fmt.set_powerlimits((0, 0))

    colorbar_axis = inset_axes(
        ax,
        width="100%",  # width = % of parent_bbox width
        height="6%",  # height = % of parent_bbox width
        loc="lower left",
        bbox_to_anchor=(-2.01, -0.1, 2.0, 1),  # (x, y, width, height)
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    colorbar_axis.tick_params(direction="in", width=0.5)
    cbar = fig.colorbar(
        contourf,
        cax=colorbar_axis,
        # format=fmt, # comment this if you don't want scientific notation
        ticks=np.linspace(vmin, vmax, numscale),
        extendfrac=0,
        orientation="horizontal",
    )
    # Get rid of the original label
    colorbar_axis.get_xaxis().get_offset_text().set_visible(False)
    exponent_text = colorbar_axis.get_xaxis().get_major_formatter().get_offset()
    # Recreate the label specifying its coordinate
    cbar.set_label(
        r"$C_p$ " + exponent_text, rotation="horizontal", labelpad=0, ha="left"
    )
    colorbar_axis.xaxis.set_label_coords(1.05, 1.0, transform=colorbar_axis.transAxes)

    # --------------------------------------------------
    #### Save file ####

    output_file_name = "Fig_Cp_times_60k_pitch_k010_M01_span01_vs_04"
    texfig.savefig(output_file_name, dpi=1000, bbox_inches="tight", pad_inches=0)
    # plt.show()
    plt.close("all")

    # Fix bug in the PGF imports.
    # New versions of matplotlib PGF backend is using includegraphics instead of
    # pgfimage, but the latex driver seems to doesn't find the file when it is in
    # another directory, even using the `import` package. This fix aims to force
    # the latex to use pgfimage.
    try:
        with open(output_file_name + ".pgf", "r", encoding="UTF-8") as fid:
            lines = fid.readlines()

        with open(output_file_name + ".pgf", "w", encoding="UTF-8") as fid:
            for line in lines:
                fid.write(re.sub(r"\\includegraphics", r"\\pgfimage", line))
    except OSError:
        pass


if __name__ == "__main__":
    main()
