'''
unfold 
systems in Cartesian
'''
import numpy as np
from vasp_io import readPROCAR_phase, readCONTCAR, readKPOINTS, getNELECT
# from structure import Structure, Lattice
import matplotlib.pyplot as plt
# from ploter import grid, interpol2d
# import bz_plot
import argparse

np.set_printoptions(precision=3, suppress=True)

def draw_spectral_wt(ax, path):

    x = np.load(path+'/x.npy')
    y = np.load(path+'/y.npy')
    z = np.load(path+'/z.npy')
    Z = np.load(path+'/Z.npy')

    extent = (min(x), max(x), min(y), max(y))
    ax.imshow(Z.T, extent=extent, aspect='auto', vmax=2, cmap='YlGnBu')
    ax.set_ylim((min(y), max(y)))


def draw_bulk_band(ax, path, k_list=None):
    def get_kpt_length(kpt_vec, rec_mat, labels):
        """kpt vectors to one line
        """
        # Cartesian coord
        for index, kpt in enumerate(kpt_vec):
            kpt_vec[index] = np.dot(kpt, rec_mat)

        kpts = np.append([kpt_vec[0]], kpt_vec, axis=0)
        x = [np.linalg.norm(kpts[i] - kpts[i-1]) for i, k in enumerate(kpts)]
        x = x[1:]

        for indx in range(len(x)):
            # if indx % nkpt_line == 0:
            if not labels[indx - 1] == '' and not labels[indx] == '':
                x[indx] = 0
        x = np.cumsum(x)
        return x
    def get_efermi(PROCAR):
        CUTOFF_OCC = 0.5
        occs = PROCAR[5].flatten()
        eigs = PROCAR[2].flatten()
        eigs = [eig for idx, eig in enumerate(eigs)
                if occs[idx] > CUTOFF_OCC]
        return max(eigs)

    procar_path = '{}/PROCAR'.format(path)
    kpoints_path = '{}/KPOINTS'.format(path)
    contcar_path = '{}/CONTCAR'.format(path)
    PROCAR = readPROCAR_phase(procar_path)

    latConst, latticeVecs, atomSetDirect, dynamics_list = \
        readCONTCAR(contcar_path)

    rec_mat = np.linalg.inv(latticeVecs).T / latConst

    e_fermi = get_efermi(PROCAR)
    eigs = PROCAR[2] - e_fermi
    
    kpts, labels = readKPOINTS(fileName=kpoints_path)

    k_list = k_list or [0, eigs.shape[0]]
    x = get_kpt_length(kpts, rec_mat, labels)
    x0 = 0

    for k_i in range(len(k_list) - 1):
        k_st, k_end = k_list[k_i], k_list[k_i+1]
        if k_st > k_end:
            indx_x = np.linspace(k_end, k_st, k_st-k_end+1, dtype=int)
            indx_e = np.linspace(k_st, k_end, k_st-k_end+1, dtype=int)
        else:
            indx_x = np.linspace(k_st, k_end, k_end-k_st+1, dtype=int)
            indx_e = np.linspace(k_st, k_end, k_end-k_st+1, dtype=int)

        for band in eigs.T:
            ax.plot(x0 + x[indx_x] - x[indx_x[0]], band[indx_e], color='#ca0000', lw=2)
        x0 = x[indx_e[-1]]


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='draw unfolded nanowire'
        'band structure and bulk band structure.')
    parser.add_argument('path', metavar='dir', type=str,
                       help='path where [x,y,z,Z].npy are', default='./')
    parser.add_argument("-k", "--kpts", type=int, nargs='+', 
                        help='k index list in bulk band calculation')
    BULK_PATH = '/home/users/nwan/02Project/14_DIELC_NW/01_bulk/' \
                '00_pristine/unfold/00_111/01_BAND'
    parser.add_argument("-b", "--bulkpath", type=str,
                        help='path where [x,y,z,Z].npy are', default=BULK_PATH)


    args = parser.parse_args()
    path = args.path
    band_kpt = args.kpts
    bulk_path = args.bulkpath

    ax = plt.subplot(111)

    draw_spectral_wt(ax, path)
    draw_bulk_band(ax, bulk_path, band_kpt)
    plt.show()
