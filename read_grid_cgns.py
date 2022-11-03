import CGNS
import numpy as np
import matplotlib.pyplot as plt


class _CGNSzones:
    def __init__(self):
        self.x = None
        self.y = None
        self.z = None
        self.q = None


def read_grid_in_cgns(file):

    ifile, nbases = CGNS.open_file_read(file)

    nzones = CGNS.nzones_read(ifile, nbases)

    zone = []

    for ibase in range(1, nbases + 1):
        for izone in range(1, nzones + 1):

            zone.append(_CGNSzones())

            idim = CGNS.zonedim_read(ifile, ibase, izone)

            isize, nx, ny, nz = CGNS.zone_size_read(ifile, ibase, izone, idim)

            if idim == 2:
                zone[izone - 1].x = CGNS.read_2d_coord(
                    "CoordinateX", ifile, ibase, izone, [1, 1], isize, nx, ny
                )
                zone[izone - 1].y = CGNS.read_2d_coord(
                    "CoordinateY", ifile, ibase, izone, [1, 1], isize, nx, ny
                )

            else:
                raise NotImplementedError("Does not support 3D just yet")

    x = np.concatenate((zone[0].x, zone[1].x[:, 1:]), axis=1)
    y = np.concatenate((zone[0].y, zone[1].y[:, 1:]), axis=1)

    CGNS.close_file(ifile)

    return x, y


if __name__ == "__main__":
    x, y = read_grid_in_cgns("../output_examples/grid2D.cgns")

    print(x.shape, y.shape)

    plt.plot(x[:, :10], y[:, :10], "o")
    plt.show()
