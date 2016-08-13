"""
unfold 
systems in Cartesian
"""
import numpy as np
from vasp_io import readPROCAR_phase, readCONTCAR, getNELECT
from structure import Structure, Lattice
import matplotlib.pyplot as plt
import bz_plot
import argparse

np.set_printoptions(precision=3, suppress=True)


def get_kpt_path(st_kpt, end_kpt, rec_lat_mat, n_points):
    """ kpoints from st_kpt to end_kpt
    """
    path = [st_kpt * ((n_points - i) / float(n_points)) \
            + end_kpt * (i / float(n_points))
            for i in range(n_points+1)]
    path = [np.dot(kpt, rec_lat_mat) for kpt in path]
    return path


def get_proj_coord(dist_vec, lat_mat_sc, lat_mat_prim, axis=None):
    """ return dist_vec 
        in fractional coordinates
        with respect to lat_mat_prim
    """
    SPAN = [-1., 0., 1.]
    SPAN = [0.]
    axis = axis or -1
    coord_list = []
    for x in SPAN:
        for y in SPAN:
            for z in SPAN:
                dit_vec_disp = dist_vec + np.array([x, y, z])
                a = np.dot(np.dot(dit_vec_disp, lat_mat_sc), np.linalg.inv(lat_mat_prim))
                coord_list.append(a)

    residue_list = []
    for c_ind, coord in enumerate(coord_list):
        residue = np.abs(np.rint(coord) - coord)
        residue_list.append(residue[:, :, axis])

    a = np.zeros(dist_vec.shape)
    for i in xrange(a.shape[0]):
        for j in xrange(a.shape[1]):
            residue = [r[i, j] for r in residue_list]
            ind = np.argmin(residue)
            # print residue, ind
            a[i, j] = coord_list[ind][i, j]
            # a[j, i] = coord_list[ind][i, j]
    return a


def get_overlap(dist_mat_vec, prim_atom_index, lat_mat_sc, lat_mat_prim, tol):
    """ if atoms overlap
        S_0N,RM
    """
    proj_coord = get_proj_coord(dist_mat_vec, lat_mat_sc, lat_mat_prim)
    overlap = np.abs(proj_coord - np.rint(proj_coord)) < tol
    overlap = np.prod(overlap, axis=2)

    return overlap


def get_lat_mat_prim(lat_mat_sc, dist_mat_vec, atoms, axis=2):
    """ get primitive lattice
        Args:
          atoms: list of index in dist_mat_vec
          first atom is origin 
          rest point primitive lattice vectors
    """
    length = np.linalg.norm(lat_mat_sc, axis=-1)
    small_ax = np.argmin(length)
  
    assert len(atoms) == 4, 'len(atoms){} is not 4 '.format(len(atoms))

    lat_mat_prim = np.zeros((3,3))

    for i, atom in enumerate(atoms[1:]):
        if atom == atoms[0]:
            lat_mat_prim[i, small_ax] = 1
        else:
            lat_mat_prim[i, :] = dist_mat_vec[atom, atoms[0]]

    for ind, lat_vec in enumerate(lat_mat_prim):
        for lat_vec_other in lat_mat_prim[ind+1:, :]:
            if (lat_vec == lat_vec_other).all():
                for ax_ind in [small_ax, (small_ax+1)%2]:
                    shift_vec = np.zeros(3)
                    shift_vec[ax_ind] = 1.
                    lat_mat_prim[ind] = lat_vec - shift_vec * np.sign(lat_vec[ax_ind])
                    if np.linalg.det(lat_mat_prim) != 0:
                        break

    return lat_mat_prim


def get_lat_mat_sc_refine(lat_mat_sc, lat_mat_prim, axis=-1):
    """
    """    
    AXIS_LENGTH = 1E2
    lat_mat_sc_refine = np.copy(lat_mat_sc)
    lat_mat_sc_refine[axis] = np.array([0, 0, AXIS_LENGTH])

    return lat_mat_sc_refine


def get_spectral_wt(kpt_sc, kpt_prims, proj_cmplx, overlap, layer_ind, dist_mat_vec,
                    lat_mat_sc, lat_mat_prim, rec_axis):
    """ sum_MNr exp(ik_prim (r-r'(M))) C_M^KJ, C_N^KJ* overlap
        ->      exp(ik_prim_xy (r-r'(M))) * sin(k_z z) * C_M^KJ C_N^KJ * overlap
        for all kpt_prims
    """
    v_axis = np.array([0, 0, 1.])
    v_axis_rec = np.dot(rec_axis, rec_lat_mat_prim)

    n_layer = max(layer_ind)

    # n_atom = len(dist_mat_vec)
    n_orbit = proj_cmplx.shape[-1]
    W_kpt = np.zeros((len(kpt_prims)))

    proj_cmplx_ = proj_cmplx.flatten('C')
    CMCN = proj_cmplx_.reshape([proj_cmplx_.size, 1]) *\
           np.conj(proj_cmplx_.reshape([1, proj_cmplx_.size]))
    
    CMCN *= np.kron(overlap, np.eye(n_orbit))
    dist_cart_vec = np.dot(dist_mat_vec, lat_mat_sc)

    for kpt_i, kpt_prim in enumerate(kpt_prims):
        W = 0
        # important
        # Bloch factor
        # perpendicular to the axis of the superlattice
        del_kpt = (kpt_prim - np.dot(kpt_prim, v_axis) * v_axis
                   - kpt_sc)
        phase = np.dot(dist_cart_vec, del_kpt)
        bloch_phase = np.exp(-2j * np.pi * phase)
        W =  np.kron(bloch_phase, np.eye(n_orbit)) * CMCN

        # huckel factor
        # parallel to the axis of the superlattice
        del_kpt = (np.dot(kpt_prim, v_axis) * v_axis - kpt_sc)
        k_huckel = np.dot(del_kpt, v_axis_rec) / np.linalg.norm(v_axis_rec) ** 2
        k_huckel = k_huckel * 2 * (n_layer + 1)

        phase = np.dot(dist_cart_vec, del_kpt)

        huckel_coeff = np.sin(np.pi * k_huckel * layer_ind / (n_layer + 1.))
        basis = np.where(layer_ind == 1)
        # print np.where(overlap[basis][0] == 1)
        huckel_proj = proj_cmplx * np.tile(huckel_coeff, (n_orbit, 1)).T
        # summation
        sum_huckel = 0
        for overlap_basis in overlap[basis]:
            ind = np.where(overlap_basis == 1)
            basis_sum = np.sum([np.linalg.norm(item) for item in np.sum(huckel_proj[ind], axis=0)])
            sum_huckel += basis_sum

        W_kpt[kpt_i] = np.sum(W.real) * sum_huckel

    return W_kpt


def is_in_WS_cell(kpt, rec_lat_mat, tol=0.05):
    """ kpt and rec_lat_mat are Cartesian
        Args:
            kpt: 3X1 array in Cartesian coords.
            rec_lat_mat: 3X3 array in Cartesian coords.
            tol: buffer
    """
    from bz_plot import get_dd_matrix
    A, b = get_dd_matrix(rec_lat_mat)

    return all(np.dot(A, kpt) <= b * (1 + tol))


def get_G_sc_set(rec_lat_mat_refine, rec_lat_mat_prim, n_layer, rec_axis, tol=0.05):
    """ return list G_sc within 1st BZ_prim
        Args:
            rec_lat_mat_refine: 
            rec_lat_mat_prim: 
            tol: 
    """
    def get_G_huckel(v_axis_rec, n_layer):
        """ return list G_sc within 1st BZ_prim
            Args:
                rec_lat_mat_prim: 
                tol: 
        """
        G_sc_set = [v_axis_rec*i/(n_layer+1.)/2. for i in range(1, n_layer+1)]
        G_sc_set += [-v_axis_rec*i/(n_layer+1.)/2. for i in range(1, n_layer+1)]
        # G_sc_set = [v_axis_rec*i/(n_layer-1.)/2. for i in range(0, n_layer+1)]
        # G_sc_set += [-v_axis_rec*i/(n_layer-1.)/2. for i in range(0, n_layer+1)]

        return G_sc_set

    v_axis_rec = np.dot(rec_axis, rec_lat_mat_prim)
    # print 'v_axis_rec', v_axis_rec
    N_EXP_X = 7
    G_sc_set = []
    # along axis, no expansion
    for G_k in get_G_huckel(v_axis_rec, n_layer):
        for i in range(-N_EXP_X, N_EXP_X):
            for j in range(-N_EXP_X, N_EXP_X):
                G = np.dot(np.array([i, j, 0], dtype=float), rec_lat_mat_refine)
                G = G + G_k
                if is_in_WS_cell(G, rec_lat_mat_prim * 2):
                    G_sc_set.append(G)
    return G_sc_set


def get_unfold_kpts_sl(kpt_sc, rec_lat_mat_refine, rec_lat_mat_prim, n_layer, rec_axis):
    """ return list of kpt_prim = kpt_sc + G_sc
        G_sc expand only perpendicular to supper latice axis
    """
    kpt_prim_list = []
    G_sc_set = get_G_sc_set(rec_lat_mat_refine, rec_lat_mat_prim, n_layer, rec_axis)

    for G_sc in G_sc_set:
        kpt = G_sc + kpt_sc
        if is_in_WS_cell(kpt, rec_lat_mat_prim):
            kpt_prim_list.append(kpt)

    return kpt_prim_list


def get_dist_to_line(kpt, st_kpt, end_kpt):
    """ return:
        dist_para, dist_perp: distance parallel and perpendicular to 
        line, st_kpt  ----- end_kpt
    """
    dist = st_kpt - kpt
    
    para_vec = st_kpt - end_kpt
    para_vec /= np.linalg.norm(para_vec)
    # print np.linalg.norm(para_vec)
    dist_para = np.dot(dist, para_vec)
    dist_perp = dist - dist_para * para_vec
    dist_perp = np.linalg.norm(dist_perp)

    return dist_para, dist_perp


def get_layer_index(struct, proj_coord, overlap, indice, lat_mat_prim, axis, atom_basis):
    """ from structure return n_atom list of layer index
    """    
    temp = proj_coord * np.array([overlap, overlap, overlap]).T
    temp = temp[atom_basis, :, :]
    # print temp 
    temp = [[np.dot(p, axis) for p in temp_basis]
            for temp_basis in temp]

    height = np.rint(np.array(temp))
    # print height
    height = height.sum(axis=0)
    height = height - height.min() + 1
    # print height
    return np.array(height,  dtype=int)


def read_atom_structure(path, atoms_prim, atom_basis, defect_atom_index=None, 
                        axis=None, center=None):
    """ read atomic structure of super lattice
        Returns
        Args
    """
    if center is None:
        center = np.array([0, 0, 0])
    # axis = axis or 2
    # span = [1, 1, 0]
    # span[axis] = 0
    tol = 0.11
    struct = Structure.read_contcar(path, 'POSCAR')
    dist_mat_vec = struct.frac_dist_mat(span=[1, 1, 0], offset=center)

    if defect_atom_index is not None:
        # we deal with only one element! 
        # First element! O.N.L.Y.!
        # TODO(Sunghyun Kim): treat n elements.
        indice = range(struct.get_n_elements()[0])
        for ind in defect_atom_index:
            indice.remove(ind)
        indice = np.array(indice)
        dist_mat_vec = dist_mat_vec[indice, :, :]
        dist_mat_vec = dist_mat_vec[:, indice, :]
        n_Si = len(indice)
    else:
        n_Si = struct.get_n_elements()[0]
        dist_mat_vec = dist_mat_vec[:n_Si, :n_Si]

    atoms_prim_new = []

    for i in atoms_prim:
        atoms_prim_new.append(i - np.sum(np.array(defect_atom_index) < i))

    atoms_prim = atoms_prim_new
    atom_basis_new = []
    for i in atom_basis:
        atom_basis_new.append(i - np.sum(np.array(defect_atom_index) < i))
    atom_basis = atom_basis_new

    lat_mat_sc = struct.lattice.get_matrix()
    lat_mat_prim = get_lat_mat_prim(lat_mat_sc, dist_mat_vec, atoms_prim)
    lat_mat_prim = np.dot(lat_mat_prim, lat_mat_sc) 

    overlap = get_overlap(dist_mat_vec, atom_basis, lat_mat_sc, lat_mat_prim, tol)

    proj_coord = get_proj_coord(dist_mat_vec, lat_mat_sc, lat_mat_prim)

    lat_mat_sc_refine = get_lat_mat_sc_refine(lat_mat_sc, lat_mat_prim)

    layer_ind = get_layer_index(struct, proj_coord, overlap, indice, lat_mat_prim, axis, atom_basis)

    return lat_mat_sc, lat_mat_prim, lat_mat_sc_refine, proj_coord, dist_mat_vec, overlap, indice, layer_ind


def calc_w(rec_lat_mat_sc, rec_lat_mat_sc_refine, rec_lat_mat_prim, lat_mat_sc, lat_mat_prim,\
           dist_mat_vec, overlap, proj_coord, Kpts, Eigs, Proj_cmplx, layer_ind,\
           rec_axis, kpts=None, bands=None):
    """ return spectral weight
        Args:
            kpts: list of indice for supercell kpts
            bands: list of indice for supercell bands
    """

    # l_sel_kpt_path = True
    n_layer = max(layer_ind)

    kpts_sc = kpts or range(Eigs.shape[0])
    bands = bands or range(Eigs.shape[1])

    x = []
    y = np.array([])
    z = np.array([])

    for kpt_i in kpts_sc:
        kpt_sc = Kpts[kpt_i]
        kpt_sc = np.dot(kpt_sc, rec_lat_mat_sc)

        kpt_prims = get_unfold_kpts_sl(kpt_sc, rec_lat_mat_sc_refine, rec_lat_mat_prim, n_layer, rec_axis)

        for band_i in bands:
            print 'kpt_i, band_i',kpt_i, band_i
            proj_cmplx = Proj_cmplx[kpt_i, band_i, :, :4]

            W_kpt = get_spectral_wt(kpt_sc, kpt_prims, proj_cmplx, overlap, layer_ind,\
                                    dist_mat_vec, lat_mat_sc, lat_mat_prim, rec_axis)
            x = x + kpt_prims
            y = np.append(y, np.ones(len(kpt_prims)) * Eigs[kpt_i, band_i])
            z = np.append(z,  W_kpt)

    kpt_prims = x
    eigs = y
    w = z
    return kpt_prims, eigs, w


def draw_BZ_proj(kpt_prims, eigs, w, rec_lat_mat_sc_refine, rec_lat_mat_prim, name=None, norm_factor=None):
    PREC = 1E4
    norm_factor = norm_factor or 1.
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.grid(False)
    ax.set_axis_off()
    ax.set_aspect('equal')
    rec_lat_mat_prim = np.array(rec_lat_mat_prim*PREC, dtype=int)/PREC

    w_norm = w #/ w.sum()
    print 'w.max()', w.max()

    bz_plot.draw_arrow(ax, [0, 0, 0], 1.2*(np.array([0, 0, np.sqrt(2)/2.5])), label='')
    bz_plot.draw_arrow(ax, [0, 0, 0], 1.2*(np.array([1/2.5, -1/2.5, 0])), label='')
    bz_plot.draw_arrow(ax, [0, 0, 0], 1.2*(np.array([-1/2.5, -1/2.5, 0])), label='')

    # ls_list = [0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 
    #            0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1]
    # ls_list = [1] * len(ls_list)
    # ls_list = ['-' if item else '--' for item in ls_list]
    bz_plot.draw_BZ_edge(ax, rec_lat_mat_sc_refine, color='#0571b0', diff=0)
    # ls_list = [1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 
    #            1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1]
    # ls_list = ['-' if item else '--' for item in ls_list]
    bz_plot.draw_BZ_edge(ax, rec_lat_mat_prim, color='k', diff=1000)

    bz_plot.draw_BZ_points(ax, kpt_prims, w_norm, color='#ca0020', norm_factor=norm_factor)


    if name is not None:
        plt.savefig('{}.eps'.format(name))
    else:
        plt.show()
    plt.clf()
    plt.close()


if __name__ == '__main__':
    '''
    '''
    parser = argparse.ArgumentParser(description='Unfold crystal momentum in supper lattice structure.')
    parser.add_argument('path', metavar='dir', type=str,
                       help='path where PROCAR, OUTCAR, POSCAR is')
    parser.add_argument("-p", "--prim", type=int, nargs=4, 
                        help='atoms_prim: pointing primitive lattice vectors')
    parser.add_argument("-b", "--basis", type=int, nargs='+',
                        help='atoms_basis: basis atoms in primitive cell')
    parser.add_argument("-a", "--axis", type=int, nargs=3,
                        help='superlattice axis n1*a1+n2*a2+n3*a3', default=[0, 1, 1])
    parser.add_argument("-r", "--rec_axis", type=int, nargs=3,
                        help='reciprocal axis n1*b1+n2*b2+n3*b3', default=[0, 1, 1])
    parser.add_argument("-d", "--defect", type=int, nargs='+',
                        help='atoms_defect: defect atom', default=[])
    parser.add_argument("-n", "--bands", type=int, nargs='+',
                        help='band indices', default=[])
    parser.add_argument("-k", "--kpt", type=int, nargs=1,
                        help='kpt_i: supercell kpt index', default=0)
    
    args = parser.parse_args()
    path = args.path
    atoms_prim = args.prim
    atom_basis = args.basis
    defect_atom_index = args.defect
    kpt_i = args.kpt
    axis = args.axis
    rec_axis = args.rec_axis
    bands = args.bands

    band_cbm = getNELECT('{}/OUTCAR'.format(path)) / 2
    if len(bands) < 1:
        bands = range(0, band_cbm)
    # print band_cbm

    lat_mat_sc, lat_mat_prim, lat_mat_sc_refine, proj_coord, dist_mat_vec, overlap, indice, layer_ind = \
        read_atom_structure(path, atoms_prim, atom_basis, defect_atom_index, axis)
    
    rec_lat_mat_sc_refine = np.linalg.inv(lat_mat_sc_refine).T
    rec_lat_mat_prim = np.linalg.inv(lat_mat_prim).T
    rec_lat_mat_sc = np.linalg.inv(lat_mat_sc).T
    print path
    print 'np.dot(axis, lat_mat_prim)'
    print np.dot(axis, lat_mat_prim)
    print 'lat_mat_prim'
    print lat_mat_prim
    print 'lat_mat_sc_refine'
    print lat_mat_sc_refine
    print 'rec_lat_mat_prim'
    print rec_lat_mat_prim
    Kpts, K_wt, Eigs, Proj, Proj_cmplx, Occs = readPROCAR_phase(path + '/PROCAR')

    ratio = np.sum(np.sum(Proj[:, :, indice, :], axis=2), axis=-1) \
            / (np.sum(np.sum(Proj, axis=2), axis=-1) + 1E-10)

    Proj_cmplx = Proj_cmplx[:, :, indice, :]
    # for band_i in range(band_cbm -1, band_cbm + 4):
    # for band_i in range(0, band_cbm + 10):
    for band_i in bands:
        kpt_prims, eigs, w = \
            calc_w(rec_lat_mat_sc, rec_lat_mat_sc_refine, rec_lat_mat_prim, lat_mat_sc, lat_mat_prim,\
                           dist_mat_vec, overlap, proj_coord, Kpts, Eigs, Proj_cmplx, layer_ind,\
                           rec_axis, kpts=[kpt_i], bands=[band_i],)

        print 'bulk ratio', ratio[kpt_i, band_i]
        print 'eig', eigs[0]
        # name = '{}.{}.{}'.format(path.split('/')[:], band_i, kpt_i)
        print 'max', np.cumsum(np.sort(w)[::-1])/w.sum()
        name = None
        # kpt_prims = eigs = w = None
        draw_BZ_proj(kpt_prims, eigs, w, rec_lat_mat_sc_refine, rec_lat_mat_prim, name=name, norm_factor=ratio[kpt_i, band_i])
