import numpy as np
import CGNS


class _CGNSzones:
    def __init__(self):
        self.density = None
        self.momentum_x = None
        self.momentum_y = None
        self.pressure = None


def read_flow_in_cgns(file):

    ifile, nbases = CGNS.open_file_read(file)

    nzones = CGNS.nzones_read(ifile, nbases)

    zone = []
    q_vector = [None] * 4

    for ibase in range(1, nbases + 1):
        for izone in range(1, nzones + 1):

            zone.append(_CGNSzones())

            idim = CGNS.zonedim_read(ifile, ibase, izone)

            isize, nx, ny, nz = CGNS.zone_size_read(ifile, ibase, izone, idim)

            if idim == 2:
                zone[izone - 1].density = CGNS.read_2d_flow(
                    "Density", ifile, ibase, izone, [1, 1], isize, nx, ny
                )
                zone[izone - 1].momentum_x = CGNS.read_2d_flow(
                    "MomentumX", ifile, ibase, izone, [1, 1], isize, nx, ny
                )
                zone[izone - 1].momentum_y = CGNS.read_2d_flow(
                    "MomentumY", ifile, ibase, izone, [1, 1], isize, nx, ny
                )
                zone[izone - 1].pressure = CGNS.read_2d_flow(
                    "Pressure", ifile, ibase, izone, [1, 1], isize, nx, ny
                )
            else:
                raise NotImplementedError("Does not support 3D just yet")

        q_vector[0] = np.concatenate((zone[0].density, zone[1].density[:, 1:]), axis=1)
        q_vector[1] = np.concatenate(
            (zone[0].momentum_x, zone[1].momentum_x[:, 1:]), axis=1
        )
        q_vector[2] = np.concatenate(
            (zone[0].momentum_y, zone[1].momentum_y[:, 1:]), axis=1
        )
        q_vector[3] = np.concatenate(
            (zone[0].pressure, zone[1].pressure[:, 1:]), axis=1
        )

    time = CGNS.descriptors_read(ifile, ibase)

    q_vector = np.array(q_vector)

    CGNS.close_file(ifile)

    return q_vector, time


if __name__ == "__main__":
    q = read_flow_in_cgns("../output_examples/qout2Davg001170.cgns")

    print(q.shape)
