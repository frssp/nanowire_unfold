'''
unfold 
systems in Cartesian
'''
import numpy as np
from vasp_io import readPROCAR_phase, readCONTCAR
from structure import Structure, Lattice
import matplotlib.pyplot as plt
from ploter import grid, interpol2d
import warnings
import bz_plot
import argparse

np.set_printoptions(precision=3, suppress=True)


def get_kpt_path(st_kpt, end_kpt, rec_lat_mat, n_points):
    '''
    kpoints from st_kpt to end_kpt
    '''
    path = [st_kpt * ((n_points - i) / float(n_points)) \
            + end_kpt * (i / float(n_points))
            for i in range(n_points+1)]
    path = [np.dot(kpt, rec_lat_mat) for kpt in path]
    return path


def get_proj_coord(dist_vec, lat_mat_sc, lat_mat_prim):
    '''
    return dist_vec 
    in fractional coordinates
    with respect to lat_mat_prim
    '''
    a = np.dot(np.dot(dist_vec, lat_mat_sc),\
               np.linalg.inv(lat_mat_prim))

    for i in xrange(a.shape[0]):
        for j in xrange(a.shape[1]):
            if i > j:
                a[i, j] = a[j, i]
    return a


def get_overlap(dist_mat_vec, prim_atom_index, lat_mat_sc, lat_mat_prim, tol=0.2):
    '''
    if atoms overlap
    S_0N,RM
    '''
    # TODO(Sunghyun Kim): tol should be adjusted??
    proj_coord = get_proj_coord(dist_mat_vec, lat_mat_sc, lat_mat_prim)
    overlap = np.abs(proj_coord - np.rint(proj_coord)) < tol
    overlap = np.prod(overlap, axis=2)

    return overlap


def get_lat_mat_prim(dist_mat_vec, atoms, axis=2):
    '''
    atoms: list of index in dist_mat_vec
    first atom is origin 
    rest point primitive lattice vectors
    '''
    assert len(atoms) == 4, \
        'len(atoms){} is not 4 '.format(len(atoms))

    lat_mat_prim = np.zeros((3,3))
    for i, atom in enumerate(atoms[1:]):
        if atom == atoms[0]:
            lat_mat_prim[i, axis] = 1
        else:
            lat_mat_prim[i, :] = dist_mat_vec[atom, atoms[0]]

    return lat_mat_prim


def get_lat_mat_sc_refine(lat_mat_sc, lat_mat_prim, n_cells, axis=-1, tol=0.5):
    """ return supercell lattice w/o vacuum
        2 * (n + 2) translations can fit in this supercell. 
        n + 2 is for the boundary condition that the standing wave is 0 at the boundary.
        2 is for the G/2.
        This supercell is (in some sense) imaginary.
        Only reciprocal lattice of this supercell is used.
        tol is unit of /AA
    """
    import itertools

    n_cells_ = np.array(n_cells) + 2
    # AXIS_LENGTH = 1E2
    # original supercell lattice vector for axial direction
    lat_mat_sc_refine = np.copy(lat_mat_sc)
    trial_planar_vecs = []
    for coord in itertools.product(range(3), repeat=3):
        if np.linalg.norm(coord) == 0:
            continue
        vec = np.array([[0, 0, 0], n_cells_, -n_cells_])[coord, [0, 1, 2]]
        vec = np.dot(vec, lat_mat_prim)

        # For other lattice vectors axial component should be small enough
        # tol should depends on th n_cells
        if np.abs(vec[axis]) < tol:
            trial_planar_vecs.append(vec)

    indx = np.argsort(np.linalg.norm(np.array(trial_planar_vecs), axis=-1))
    lat_mat_sc_refine[axis + 1] = trial_planar_vecs[indx[-1]] * 2

    i=0
    while True:
        lat_mat_sc_refine[axis + 2] = trial_planar_vecs[indx[1+i]] * 2
        if np.linalg.det(lat_mat_sc_refine) > np.linalg.det(lat_mat_prim) * 4 :
            break
        i+=1
    print lat_mat_sc_refine
    print lat_mat_prim
    return lat_mat_sc_refine


def get_spectral_wt(kpt_sc, kpt_prims, proj_cmplx, overlap, dist_mat_vec, lat_mat_sc, lat_mat_prim):
    """ return spectral weight
        sum_MNr exp(ik_prim(r-r'(M))) C_M^KJ, C_N^KJ* overlap
        for all kpt_prims
    """
    n_orbit = proj_cmplx.shape[-1]
    kpt_prims = np.array(kpt_prims)

    proj_cmplx_ = proj_cmplx.flatten('C')
    CMCN = proj_cmplx_.reshape([proj_cmplx_.size, 1]) *\
           np.conj(proj_cmplx_.reshape([1, proj_cmplx_.size])) *\
           np.kron(overlap, np.eye(n_orbit))
    
    dist_cart_vec = np.dot(dist_mat_vec, lat_mat_sc)
    
    del_kpt = kpt_prims - np.tile(kpt_sc, (len(kpt_prims), 1))
    phase = np.dot(dist_cart_vec, del_kpt.T)

    # NOT! -2J! Why?
    del_g = np.exp(+2j * np.pi * phase)

    # The major load occurs here!
    W_kpt = np.kron(del_g, np.eye(n_orbit)[:,:, np.newaxis]) *\
            np.tile(CMCN[:,:, np.newaxis], (1, 1, len(kpt_prims)))

    W_kpt = np.sum(W_kpt.real, axis=(0, 1))
    
    return W_kpt


def is_in_WS_cell_array(kpts, rec_lat_mat, tol=0.05):
    """ kpt and rec_lat_mat are Cartesian
        Args:
            kpts: nx3 array in Cartesian coords.
            rec_lat_mat: 3X3 array in Cartesian coords.
            tol: buffer
    """
    from bz_plot import get_dd_matrix
    kpts = np.array(kpts)
    A, b = get_dd_matrix(rec_lat_mat)
    l_in = np.dot(A, kpts.T) <= b[:, np.newaxis] * (1 + tol)
    return np.all(l_in, axis=0)


def get_G_sc_set(rec_lat_mat_sc, n_cells, nw_axis=2):
    """ return list G_sc within FBZ_prim
    """ 
    expand_1 = np.arange((-n_cells[0] - 2) * 2, (n_cells[0] + 2) * 2 + 1)
    expand_2 = np.arange((-n_cells[1] - 2) * 2, (n_cells[1] + 2) * 2 + 1)
    expand_3 = np.arange((-n_cells[2] - 2) * 2, (n_cells[2] + 2) * 2 + 1)

    vec = np.array(np.meshgrid(expand_1, expand_2, expand_3)).T
    vec = vec.reshape(vec.size/3, 3)
    G_sc_set = np.dot(vec, rec_lat_mat_sc)
    return G_sc_set


def get_unfold_kpts_nw(kpt_sc, rec_lat_mat_sc, rec_lat_mat_prim, n_cells, axis=2):
    '''
    return list of kpt_prim = kpt_sc + G_sc
    G_sc expand only along axis
    '''
    kpt_prim_list = []

    G_sc_set = get_G_sc_set(rec_lat_mat_sc, n_cells)

    # remove zone center of G_sc
    # consider here!!
    # TODO(Sunghyun kim): not only gamma 111 
    idx_Gamma = np.all(G_sc_set == 0, axis=1)
    G_sc_set = G_sc_set[np.logical_not(idx_Gamma)]

    kpt_prim_list = G_sc_set + kpt_sc 
    l_in = is_in_WS_cell_array(kpt_prim_list, rec_lat_mat_prim, tol=1E-2)
    kpt_prim_list = kpt_prim_list[l_in]

    # remove BZ boundary perpendicular to the axis
    # but keep BZ boundary along the axis
    multi = np.ones(3) - 0.2 / (np.array(n_cells) + 2)
    multi[axis] = 1. + 1E-2
    # only in WS cell
    l_in = is_in_WS_cell_array(kpt_prim_list, rec_lat_mat_prim * multi, tol=0)
    kpt_prim_list = kpt_prim_list[l_in]

    # remove duplicated
    kpt_prim_list = np.vstack({tuple(row) for row in kpt_prim_list})

    return kpt_prim_list


def l_kpt_on_line(kpts, st_kpt, end_kpt, rec_lat_mat_prim, rec_lat_mat_sc):
    ''' return: line, st_kpt  ----- end_kpt
        l_on: distance parallel and perpendicular to 
    '''
    min_kpt_length = np.linalg.norm(rec_lat_mat_sc, axis=1).min() / 5.

    dir_vec = end_kpt - st_kpt
    dir_vec /= np.linalg.norm(dir_vec)

    dist = kpts - st_kpt

    dist_para = np.dot(dist, dir_vec)
    
    dist_perp = dist - dist_para[:, np.newaxis] * dir_vec[np.newaxis, :]
    l_on = np.linalg.norm(dist_perp, axis=1) < min_kpt_length

    return l_on


def read_elec_structure(path, atoms_prim, atom_basis):
    def get_efermi(Occs, Eigs):
        CUTOFF_OCC = 0.5
        eigs = [eig for idx, eig in enumerate(Eigs.flatten())
                if Occs.flatten()[idx] > CUTOFF_OCC]
        return max(eigs)

    struct = Structure.read_contcar(path, 'POSCAR')
    n_Si = struct.get_n_elements()[0]
    dist_mat_vec = struct.frac_dist_mat()[:n_Si, :n_Si]

    Kpts, K_wt, Eigs, Proj, Proj_cmplx, Occs = readPROCAR_phase(path + '/PROCAR')
    
    e_fermi = get_efermi(Occs, Eigs)
    Eigs = Eigs - e_fermi

    lat_mat_sc = struct.lattice.get_matrix()
    rec_lat_mat_sc = np.linalg.inv(lat_mat_sc).T
    Kpts = [np.dot(kpt, rec_lat_mat_sc) for kpt in Kpts]

    lat_mat_prim = get_lat_mat_prim(dist_mat_vec, atoms_prim)
    lat_mat_prim = np.dot(lat_mat_prim, lat_mat_sc) 
    rec_lat_mat_prim = np.linalg.inv(lat_mat_prim).T

    overlap = get_overlap(dist_mat_vec, atom_basis, lat_mat_sc, lat_mat_prim)

    proj_coord = get_proj_coord(dist_mat_vec, lat_mat_sc, lat_mat_prim)

    n_cell_x = int(np.max(np.abs(np.rint(proj_coord)[:,:, 0] * overlap)))
    n_cell_y = int(np.max(np.abs(np.rint(proj_coord)[:,:, 1] * overlap)))
    n_cell_z = int(np.max(np.abs(np.rint(proj_coord)[:,:, 2] * overlap)))
    n_cells = [n_cell_x, n_cell_y, n_cell_z]
    # Temporary fix
    n_cells = [np.max(n_cells), np.max(n_cells), np.max(n_cells)]

    lat_mat_sc_refine = get_lat_mat_sc_refine(lat_mat_sc, lat_mat_prim, n_cells, axis=-1)
    rec_lat_mat_sc_refine = np.linalg.inv(lat_mat_sc_refine).T
    return rec_lat_mat_sc_refine, rec_lat_mat_prim, lat_mat_sc, lat_mat_prim, n_cells, n_Si,\
           dist_mat_vec, overlap, proj_coord, Kpts, Eigs, Proj_cmplx


def calc_w(rec_lat_mat_sc, rec_lat_mat_prim, lat_mat_sc, lat_mat_prim, n_cells, n_Si,\
           dist_mat_vec, overlap, proj_coord, Kpts, Eigs, Proj_cmplx,\
           kpts=None, bands=None, band_kpt=None):
    '''
    arguments:
        kpts: list of indice for supercell kpts
        bands: list of indice for supercell bands
    '''
    kpts_sc = kpts or range(Eigs.shape[0])
    bands = bands or range(Eigs.shape[1])

    x = np.empty((0,3))
    y = np.array([])
    z = np.array([])

    for kpt_i in kpts_sc:
        kpt_sc = Kpts[kpt_i]


        kpt_prims = get_unfold_kpts_nw(kpt_sc, rec_lat_mat_sc, rec_lat_mat_prim, n_cells)

        if band_kpt is not None:
            print 'along band line'
            kpt_prim_sel = np.empty((0,3))
            kpts = np.dot(band_kpt, rec_lat_mat_prim)
            l_dup = np.empty((0))
            for i in range(len(kpts) - 1):
                st_kpt, end_kpt = kpts[i], kpts[i+1]
                l_on = l_kpt_on_line(kpt_prims, st_kpt, end_kpt, rec_lat_mat_prim, rec_lat_mat_sc)
                kpt_line_add = kpt_prims[l_on]

                if len(kpt_line_add) == 0:
                    continue

                kpt_prim_sel = np.vstack((kpt_prim_sel, kpt_line_add))

                # remove spectral weight doubling
                dir_vec = end_kpt - st_kpt
                dir_vec /= np.linalg.norm(dir_vec)

                # time reversal symmetry
                dist_para_1 = np.dot(kpt_prim_sel - st_kpt, dir_vec)
                dist_para_2 = np.dot(-kpt_prim_sel - st_kpt, dir_vec)
                dist_para = np.amax([dist_para_1, dist_para_2], axis=0)

                tol_dup = np.linalg.norm(end_kpt - st_kpt) * 0.01
                diff = dist_para[np.newaxis, :] - dist_para[:, np.newaxis] + np.eye(len(dist_para)) * 100
                l_dup_line = np.amin(np.abs(diff), axis=1) < tol_dup
                l_dup = np.hstack((l_dup, l_dup_line))

            if len(kpt_prim_sel) == 0:
                continue
        else:
            kpt_prim_sel = kpt_prims
            l_dup = 1

        for band_i in bands:
            # print 'eig, kpt_i, band_i', Eigs[kpt_i, band_i], kpt_i, band_i
            proj_cmplx = Proj_cmplx[kpt_i, band_i, :n_Si, :4]
            # print 'calc spectral w'
            W_kpt = get_spectral_wt(kpt_sc, kpt_prim_sel, proj_cmplx, overlap, \
                                    dist_mat_vec, lat_mat_sc, lat_mat_prim)

            x = np.vstack((x, kpt_prim_sel))
            y = np.append(y, np.ones(len(kpt_prim_sel)) * Eigs[kpt_i, band_i])
            z = np.append(z, W_kpt * (1 - 0.5 * l_dup))

    kpt_prims = x
    eigs = y
    w = z
    return kpt_prims, eigs, w


def draw_band(x, y, z, show=True):
    x, y, z = np.array(x), np.array(y), np.array(z)
    y = y
    z = z
    # Z = interpol2d(x, y, z, 100, 700, 0.003, 0.025)
    Z = interpol2d(x, y, z, 100, 700, 0.013/np.sqrt(2)/2, 0.025)
    Z /= 100 * 300

    np.save('x.npy', x)
    np.save('y.npy', y)
    np.save('z.npy', z)
    np.save('Z.npy', Z)

    if show:
        extent = (min(x), max(x), min(y), max(y))
        ax = plt.subplot(111)
        ax.imshow(Z.T, extent=extent,
                   aspect='auto', vmax=1, cmap='terrain')
        ax.set_ylim((min(y), max(y)))
        plt.show()


def build_band(kpt_prims, eigs, w, rec_lat_mat_prim, rec_lat_mat_sc, kpt_lines):
    ''' build band structure for draw_band 
        x: kpts along kpt_line
        y: eigs 
        z: w 
    '''
    ENERGY_BUFF = 1 # eV
    kpt_prims = np.array(kpt_prims)
    kpt_lines = np.dot(kpt_lines, rec_lat_mat_prim)

    x = np.array([0])
    y = np.array([np.min(eigs) - ENERGY_BUFF])
    weight = np.array([0])

    for kpt_line_i in range(len(kpt_lines) - 1):
        st_kpt, end_kpt = kpt_lines[kpt_line_i], kpt_lines[kpt_line_i+1]
        x = np.hstack((x, x[-1] + np.linalg.norm(end_kpt - st_kpt)))
        y = np.hstack((y, [np.max(eigs) + ENERGY_BUFF]))
        weight = np.hstack((weight, [0]))

    for kpt_line_i in range(len(kpt_lines) - 1):
        x0 = x[kpt_line_i]
        st_kpt, end_kpt = kpt_lines[kpt_line_i], kpt_lines[kpt_line_i+1]

        dir_vec = end_kpt - st_kpt
        dir_vec /= np.linalg.norm(dir_vec)

        l_on = l_kpt_on_line(kpt_prims, st_kpt, end_kpt, rec_lat_mat_prim, rec_lat_mat_sc)

        # time reversal symmetry
        dist_para_1 = np.dot(kpt_prims - st_kpt, dir_vec)
        dist_para_2 = np.dot(-kpt_prims - st_kpt, dir_vec)
        dist_para = np.amax([dist_para_1, dist_para_2], axis=0)

        x = np.hstack((x, x0 + dist_para))
        y = np.hstack((y, eigs[l_on]))
        weight = np.hstack((weight, w[l_on]))

    return x, y, weight


def draw_BZ_proj(kpt_prims, eigs, w, rec_lat_mat_prim, kpt_lines, l_show=True, name='none.png'):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.grid(False)
        ax.set_axis_off()
        ax.set_aspect('equal')

        #normalize
        w_norm = w
        bz_plot.draw_BZ_edge(ax, rec_lat_mat_prim)
        bz_plot.draw_BZ_points(ax, kpt_prims, w_norm, color='#ca0020', norm_factor=None)
        if kpt_lines is not None:
            st_kpt, end_kpt = np.dot(kpt_lines, rec_lat_mat_prim)
            bz_plot.draw_arrow(ax, st_kpt, end_kpt, color='#ca0020')


        ax.view_init(elev=10, azim=72) # for <111>
        if l_show:
            plt.show()
        else:
            plt.savefig(name)


def draw_BZ(rec_lat_mat_prim, rec_lat_mat_sc, kpt_lines=None):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.grid(False)
        ax.set_axis_off()
        ax.set_aspect('equal')

        bz_plot.draw_BZ_edge(ax, rec_lat_mat_prim)
        bz_plot.draw_BZ_edge(ax, rec_lat_mat_sc)

        if kpt_lines is not None:
            st_kpt, end_kpt = np.dot(kpt_lines, rec_lat_mat_prim)
            bz_plot.draw_arrow(ax, st_kpt, end_kpt, color='#ca0020')

        plt.show()


def parse_band_kpt(kpt_list):
    if kpt_list is not None:
        assert len(kpt_list) % 3 == 0,  "length of bandkpt should be 3 X n:{}".format(len(kpt_list))
        return np.array(kpt_list).reshape((len(kpt_list) / 3, 3))
    else:
        return kpt_list

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Unfold crystal momentum in nanowire structure.')
    parser.add_argument('path', metavar='dir', type=str,
                       help='path where PROCAR, OUTCAR, POSCAR is')
    parser.add_argument("-p", "--prim", type=int, nargs=4, 
                        help='atoms_prim: pointing primitive lattice vectors')
    parser.add_argument("-b", "--basis", type=int, nargs='+',
                        help='atoms_basis: basis atoms in primitive cell')
    # parser.add_argument("-a", "--axis", type=int, nargs=3,
    #                     help='superlattice axis n1*a1+n2*a2+n3*a3', default=[0, 1, 1])
    parser.add_argument("-n", "--bands", type=int, nargs='+',
                        help='band indices', default=[])
    parser.add_argument("-k", "--kpt", type=int, nargs='+',
                        help='kpt_i: supercell kpt index', default=None)
    parser.add_argument("--band", dest='l_draw_band', action='store_true',
                        help='draw unfolded band structure')
    parser.add_argument("--bandkpt", type=float, nargs='+',
                        help='kpt_i: supercell kpt index', default=None)
    parser.set_defaults(l_draw_band=False)
    parser.add_argument("--show", dest='l_show', action='store_true',
                        help='show unfolded band structure')
    parser.set_defaults(l_show=False)


    args = parser.parse_args()
    path = args.path
    atoms_prim = args.prim
    atom_basis = args.basis
    bands = args.bands
    kpt_list = args.kpt
    band_kpt = parse_band_kpt(args.bandkpt)
    l_draw_band = args.l_draw_band
    l_show = args.l_show

    if len(bands) == 2 and bands[-1] < 0:
        bands = range(bands[0], -bands[1])

    print 'reading PROCAR:',  path
    rec_lat_mat_sc, rec_lat_mat_prim, lat_mat_sc, lat_mat_prim, n_cells, n_Si, \
    dist_mat_vec, overlap, proj_coord, Kpts, Eigs, Proj_cmplx = \
        read_elec_structure(path, atoms_prim, atom_basis)

    print '\nnumber Si atoms (w/o passivating hydrogen atoms)'
    print n_Si
    print '\nprimitive lattice vectors'
    print lat_mat_prim
    print '\nnumber of translations in the wire (w.r.t primitive lattice vectors)'
    print n_cells
    print '\nartificial supercell containing nanowire (w/ B.C.)'
    print np.linalg.inv(rec_lat_mat_sc.T)
    print '\nprimitive reciprocal lattice vector '
    print rec_lat_mat_prim
    print '\nsupercell reciprocal lattice vector '
    print rec_lat_mat_sc
    
    if l_draw_band:
        print '========================== Draw band =========================='
        # draw_BZ(rec_lat_mat_prim, rec_lat_mat_sc, band_kpt)
        kpt_prims, eigs, w = \
            calc_w(rec_lat_mat_sc, rec_lat_mat_prim, lat_mat_sc, lat_mat_prim,\
                   n_cells, n_Si, dist_mat_vec, overlap, proj_coord, Kpts, Eigs, Proj_cmplx,\
                   kpts=kpt_list, bands=bands, band_kpt=band_kpt)
        kpt_rel, eigs, weight = build_band(kpt_prims, eigs, w, rec_lat_mat_prim, rec_lat_mat_sc, band_kpt)
        draw_band(kpt_rel, eigs, weight, l_show)
        # draw_BZ_proj(kpt_prims, eigs, w, rec_lat_mat_prim, band_kpt)

    else:
        print '========================== Draw BZ =========================='
        # draw_BZ(rec_lat_mat_prim, rec_lat_mat_sc)
        for kpt_i in kpt_list:
            for band_i in bands:
                kpt_prims, eigs, w = \
                    calc_w(rec_lat_mat_sc, rec_lat_mat_prim, lat_mat_sc, lat_mat_prim,\
                           n_cells, n_Si, dist_mat_vec, overlap, proj_coord, Kpts, Eigs, Proj_cmplx,\
                           kpts=[kpt_i], bands=[band_i])
                draw_BZ_proj(kpt_prims, eigs, w, rec_lat_mat_prim, band_kpt, l_show, name='bz_proj_{}_{}.png'.format(band_i, kpt_i))

                np.save('kpt_prims_{}_{}.npy'.format(kpt_i, band_i), kpt_prims)
                np.save('eigs{}_{}.npy'.format(kpt_i, band_i), eigs)
                np.save('w_{}_{}.npy'.format(kpt_i, band_i), w)

