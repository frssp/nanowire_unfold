#!/usr/local/bin/python
"""
band structure for nanowires
with irreducible representation
"""
import numpy as np
from vasp_io import readPROCAR_phase
# from vasp_io import readCONTCAR
from structure import Structure
#import matplotlib.pyplot as plt
#from ploter import grid, interpol2d

def read_elec_structure(path, frac=True):
    """
    read electronic structure
    """
    struct = Structure.read_contcar(path, 'POSCAR')
    n_Si = struct.get_n_elements()[0]
    # dist_mat_vec = struct.frac_dist_mat()[:n_Si, :n_Si]
    print n_Si

    Kpts, K_wt, Eigs, Proj, Proj_cmplx, Occs = \
            readPROCAR_phase(path + '/PROCAR')
    lat_mat = struct.lattice.get_matrix()
    rec_lat_mat = np.linalg.inv(lat_mat).T
    if not frac:
        Kpts = [np.dot(kpt, rec_lat_mat) for kpt in Kpts]

    return struct, Kpts, Eigs, Proj_cmplx


def get_bloch_phase(kpt, pos, n_orbit, axis=None):
    """returns matrix contains bloch phases for atoms
       It is useful to remove bloch phase in wave fucntions with n_orbit / atom
    """
    axis = axis or -1
    pos = np.array(pos, dtype=np.cfloat)
    phase = np.diag(np.exp(-2.j * np.pi * kpt * pos[:, axis]))
    phase = np.kron(phase, np.eye(n_orbit))

    return phase


class SymOper(object):
    """
    collection of symmetric operators
    """
    def __init__(self, pos, lat_mat, tol=1E-4, n_orbit=1):
        """
        constructor
        """
        self.pos = pos
        self.lat_mat = lat_mat
        self.tol = tol
        self.n_orbit = n_orbit
        self.operators = []

    def chk_perm(self, overlap):
        """check permuation is valid or not 
        column sum and row sum should be list of all elements are 1
        """
        col_sum = np.sum(overlap, axis=0)
        if not (np.abs(col_sum - 1) < self.tol).all():
            print col_sum
            return False
        row_sum = np.sum(overlap, axis=1)
        if not (np.abs(row_sum - 1) < self.tol).all():
            print row_sum
            return False
        return True

    @staticmethod
    def reflection_matrix(norm_vec):
        """Return the reflection matrix associated with reflection through
        a plane of ax+by+cz=0, norm_vec = [a, b, c]

        see http://en.wikipedia.org/wiki/Transformation_matrix
        """

        norm_vec = norm_vec / np.linalg.norm(norm_vec)
        a, b, c = norm_vec
        aa, bb, cc = a*a, b*b, c*c
        bc, ac, ab, = b*c, a*c, a*b
        return np.array([[1-2*aa, -2*ab, -2*ac],
                         [-2*ab, 1-2*bb, -2*bc],
                         [-2*ac, -2*bc, 1-2*cc]])

    @staticmethod
    def rotation_matrix(theta, axis):
        """Return the rotation matrix
        associated with counterclockwise rotation about
        the given axis by theta radians.
        """
        import math

        axis = np.asarray(axis)
        theta = np.asarray(theta)
        axis = axis/math.sqrt(np.dot(axis, axis))
        a = math.cos(theta/2)
        b, c, d = -axis*math.sin(theta/2)
        aa, bb, cc, dd = a*a, b*b, c*c, d*d
        bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
        return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                         [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                         [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

    @staticmethod
    def orbit_transform(trans_mat, n_orbit):
        """
        orbital transform matrix
        s is not transformed.
        p is transformed like x, y, z
        """
        orb_mat = np.zeros((n_orbit, n_orbit))
        # s orbital
        orb_mat[0, 0] = 1

        if n_orbit == 4:
            # s + p orbital
            orb_mat[1:4, 1:4] = trans_mat
        # print np.array(np.round(orb_mat * 100), dtype=float)/100.
        return orb_mat

    def build_iden_operator(self):
        """identity operator
        """
        overlap = np.eye(len(pos))
        # self.operators.append(overlap)
        # return overlap
        op = np.kron(overlap, np.eye(self.n_orbit))
        self.operators.append(op)

    def build_rot_operator(self, center, theta, axis=None):
        """C_N^M
        rotational operator mapping pos_i -> pos_j
        return rot_ij = 1 if pos_i -> pos_j
                        0 otherwise
        """

        tol = self.tol
        pos = self.pos
        lat_mat = self.lat_mat
        axis = axis or [0, 0, 1]
        n_atom = len(pos)

        # 0 ~ 1 -> -0.5 ~ 0.5
        tran_vec = np.array([0.5, 0.5, 0]) + center
        pos = np.array(pos) + tran_vec
        pos = pos % 1 - tran_vec

        # Cartesian
        pos = np.dot(pos, lat_mat)

        # rotate positions
        rot_mat = self.rotation_matrix(theta, axis)
        pos_rot = np.dot(rot_mat, pos.T).T

        # Fractional coordinates
        pos = np.dot(pos, np.linalg.inv(lat_mat))
        pos_rot = np.dot(pos_rot, np.linalg.inv(lat_mat))

        overlap = np.zeros((n_atom, n_atom), dtype=int)
        for i, p_old in enumerate(pos):
            diff = pos_rot - p_old
            overlap[i, :] = np.linalg.norm(diff - np.rint(diff), axis=1) < tol
        
        if not self.chk_perm(overlap):
            print 'not valid op', overlap
        # self.operators.append(overlap)
        # return overlap
        orb_mat = self.orbit_transform(rot_mat, self.n_orbit)
        
        op = np.kron(overlap, orb_mat)
        self.operators.append(op)

    def build_refl_operator(self, center, norm_vec):
        """
        Sigma_(x/y/z)
        reflection operator mapping pos_i -> pos_j
        return rot_ij = 1 if pos_i -> pos_j
                        0 otherwise
        """

        tol = self.tol
        pos = self.pos
        lat_mat = self.lat_mat
        n_atom = len(pos)

        # 0 ~ 1 -> -0.5 ~ 0.5
        tran_vec = np.array([0.5, 0.5, 0]) + center
        pos = np.array(pos) + tran_vec
        pos = pos % 1 - tran_vec

        # Cartesian
        pos = np.dot(pos, lat_mat)

        # reflect positions
        refl_mat = self.reflection_matrix(norm_vec)
        pos_refl = np.dot(refl_mat, pos.T).T

        # Fractional coordinates
        pos = np.dot(pos, np.linalg.inv(lat_mat))
        pos_refl = np.dot(pos_refl, np.linalg.inv(lat_mat))

        overlap = np.zeros((n_atom, n_atom), dtype=int)
        for i, p_old in enumerate(pos):
            diff = pos_refl - p_old
            overlap[i, :] = np.linalg.norm(diff - np.rint(diff), axis=1) < tol

        if not self.chk_perm(overlap):
            print 'not valid op', overlap
        # self.operators.append(overlap)
        # return overlap

        orb_mat = self.orbit_transform(refl_mat, self.n_orbit)
        
        op = np.kron(overlap, orb_mat)
        self.operators.append(op)

    def build_rot_refl_operator(self, center, norm_vec, theta, axis=None):
        """S(x/y/z)
        reflection operator mapping pos_i -> pos_j
        return rot_ij = 1 if pos_i -> pos_j
                        0 otherwise
        """
        tol = self.tol
        pos = self.pos
        lat_mat = self.lat_mat
        axis = axis or [0, 0, 1]
        n_atom = len(pos)

        # 0 ~ 1 -> -0.5 ~ 0.5
        tran_vec = np.array([0.5, 0.5, 0]) + center
        pos = np.array(pos) + tran_vec
        pos = pos % 1 - tran_vec

        # Cartesian
        pos = np.dot(pos, lat_mat)

        # rotate positions
        rot_mat = self.rotation_matrix(theta, axis)
        pos_rot = np.dot(rot_mat, pos.T).T

        # reflect positions
        refl_mat = self.reflection_matrix(norm_vec)
        pos_rot_refl = np.dot(refl_mat, pos_rot.T).T

        # Fractional coordinates
        pos = np.dot(pos, np.linalg.inv(lat_mat))
        pos_rot_refl = np.dot(pos_rot_refl, np.linalg.inv(lat_mat))

        overlap = np.zeros((n_atom, n_atom), dtype=int)
        for i, p_old in enumerate(pos):
            diff = pos_rot_refl - p_old
            overlap[i, :] = np.linalg.norm(diff - np.rint(diff), axis=1) < tol

        if not self.chk_perm(overlap):
            print 'not valid op', overlap
        # self.operators.append(overlap)
        # return overlap

        orb_mat = self.orbit_transform(np.dot(refl_mat, rot_mat), self.n_orbit)
        # orb_mat[-1, -1] = -1
        # orb_mat = self.orbit_transform(np.dot(rot_mat, refl_mat), self.n_orbit)
        # print orb_mat
        
        op = np.kron(overlap, orb_mat)
        self.operators.append(op)

    def build_inv_operator(self, center):
        """
        i
        inversion operator mapping pos_i -> pos_j
        return inv_ij = 1 if pos_i -> pos_j
                        0 otherwise
        """
        tol = self.tol
        pos = self.pos
        lat_mat = self.lat_mat
        n_atom = len(pos)

        # 0 ~ 1 -> -0.5 ~ 0.5
        tran_vec = np.array([0.5, 0.5, 0]) + center
        pos = np.array(pos) + tran_vec
        pos = pos % 1 - tran_vec

        # Cartesian
        pos = np.dot(pos, lat_mat)

        # invert positions
        pos_inv = pos * -1

        # Fractional coordinates
        pos = np.dot(pos, np.linalg.inv(lat_mat))
        pos_inv = np.dot(pos_inv, np.linalg.inv(lat_mat))

        overlap = np.zeros((n_atom, n_atom), dtype=int)
        for i, p_old in enumerate(pos):
            diff = pos_inv - p_old
            overlap[i, :] = np.linalg.norm(diff - np.rint(diff), axis=1) < tol

        # self.operators.append(overlap)
        # return overlap
        op = np.kron(overlap, -np.eye(self.n_orbit))
        self.operators.append(op)

    def build_frac_trans(self, trans, dir_vec=[0, 0, 1]):
        """t
        translation operator mapping pos_i -> pos_j
        return rot_ij = 1 if pos_i -> pos_j
                        0 otherwise
        """

        tol = self.tol
        pos = self.pos
        # lat_mat = self.lat_mat
        # axis = axis or [0, 0, 1]
        n_atom = len(pos)

        # 0 ~ 1 -> -0.5 ~ 0.5
        tran_vec = np.array([0.5, 0.5, 0]) + self.center
        pos = np.array(pos) + tran_vec
        pos = pos % 1 - tran_vec

        # translation
        pos_trans = pos + trans * np.array(dir_vec)
        pos_trans %= 1

        overlap = np.zeros((n_atom, n_atom), dtype=int)
        for i, p_old in enumerate(pos):
            diff = pos_trans - p_old
            overlap[i, :] = np.linalg.norm(diff - np.rint(diff), axis=1) < tol
        
        op = np.kron(overlap, np.eye(self.n_orbit))
        self.operators.append(op)

    def build_glide_reflection(self, center, trans, norm_vec, dir_vec=[0, 0, 1]):
        """Ud
        tran-reflection operator mapping pos_i -> pos_j
        return rot_ij = 1 if pos_i -> pos_j
                        0 otherwise
        """

        tol = self.tol
        pos = self.pos
        lat_mat = self.lat_mat
        n_atom = len(pos)

        # 0 ~ 1 -> -0.5 ~ 0.5
        tran_vec = np.array([0.5, 0.5, 0]) + center
        pos = np.array(pos) + tran_vec
        pos = pos % 1 - tran_vec

        # translation
        pos_trans = pos + trans * np.array(dir_vec)
        pos_trans %= 1

        # Cartesian
        pos_trans = np.dot(pos_trans, lat_mat)

        # reflect positions
        refl_mat = self.reflection_matrix(norm_vec)
        pos_trans_refl = np.dot(refl_mat, pos_trans.T).T

        # Fractional coordinates
        pos_trans_refl = np.dot(pos_trans_refl, np.linalg.inv(lat_mat))

        overlap = np.zeros((n_atom, n_atom), dtype=int)
        for i, p_old in enumerate(pos):
            diff = pos_trans_refl - p_old
            overlap[i, :] = np.linalg.norm(diff - np.rint(diff), axis=1) < tol
        
        if not self.chk_perm(overlap):
            print 'not valid op', overlap
        # self.operators.append(overlap)
        # return overlap
        orb_mat = self.orbit_transform(refl_mat, self.n_orbit)
        
        op = np.kron(overlap, orb_mat)
        self.operators.append(op)

    def build_screw(self, center, trans, theta, dir_vec=[0, 0, 1]):
        """Sn
        tran-reflection operator mapping pos_i -> pos_j
        return rot_ij = 1 if pos_i -> pos_j
                        0 otherwise
        """

        tol = self.tol
        pos = self.pos
        lat_mat = self.lat_mat
        n_atom = len(pos)

        # 0 ~ 1 -> -0.5 ~ 0.5
        tran_vec = np.array([0.5, 0.5, 0]) + center
        pos = np.array(pos) + tran_vec
        pos = pos % 1 - tran_vec

        # translation
        pos_trans = pos + trans * np.array(dir_vec)
        pos_trans %= 1

        # Cartesian
        pos_trans = np.dot(pos_trans, lat_mat)

        # rotate positions
        rot_mat = self.rotation_matrix(theta, dir_vec)
        pos_trans_rot = np.dot(rot_mat, pos_trans.T).T

        # Fractional coordinates
        pos_trans_rot = np.dot(pos_trans_rot, np.linalg.inv(lat_mat))

        overlap = np.zeros((n_atom, n_atom), dtype=int)
        for i, p_old in enumerate(pos):
            diff = pos_trans_rot - p_old
            overlap[i, :] = np.linalg.norm(diff - np.rint(diff), axis=1) < tol
        
        if not self.chk_perm(overlap):
            print 'not valid op', overlap
        # self.operators.append(overlap)
        # return overlap
        orb_mat = self.orbit_transform(rot_mat, self.n_orbit)
        
        op = np.kron(overlap, orb_mat)
        self.operators.append(op)


class PointGroup(object):
    """
    abstract class for point group
    """
    def __init__(self):
        pass

    def proj_operator(self, irrep):
        """
        projection operator to irreducible representation
        """
        char_table = self.char_table
        sym_ops = self.sym_ops

        proj_mat = np.zeros(sym_ops.operators[0].shape)
        for i, op in enumerate(sym_ops.operators):
            proj_mat += char_table[irrep, i] * op
        return proj_mat


class PG_D2d(PointGroup):
    """
    point group D_2d
    """
    def __init__(self):
        """
        make symmetry operators
        and charecter table
        """
        self.sym_ops = None
        self.char_table = np.array([[1, 1, 1, 1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, -1, -1, -1, -1],
                                    [1, 1, -1, -1, 1, 1, -1, -1],
                                    [1, 1, -1, -1, -1, -1, 1, 1],
                                    [2, -2, 0, 0, 0, 0, 0, 0]])

    def build_sym_opers(self, pos, lat_mat, center=None, n_orbit=1):
        """
        build symmetry operators
        """
        center = center or [0, 0, 0]
        sym_ops = SymOper(pos, lat_mat, n_orbit=n_orbit)

        sym_ops.build_iden_operator()
        sym_ops.build_rot_operator(center, np.pi)
        sym_ops.build_rot_refl_operator(center, [1, 0, 0], np.pi/2)
        sym_ops.build_rot_refl_operator(center, [0, 1, 0], np.pi/2)
        sym_ops.build_rot_operator(center, np.pi, axis=[1, 0, 0])
        sym_ops.build_rot_operator(center, np.pi, axis=[0, 1, 0])
        sym_ops.build_refl_operator(center, [1, 1, 0])
        sym_ops.build_refl_operator(center, [1, -1, 0])
        self.sym_ops = sym_ops


class PG_C2v(PointGroup):
    """
    point group C_2v
    """
    def __init__(self):
        """
        make symmetry operators
        and charecter table
        """
        self.sym_ops = None
        self.char_table = np.array([[1, 1, 1, 1],
                                    [1, 1, -1, -1],
                                    [1, -1, 1, -1],
                                    [1, -1, -1, 1]])

    def build_sym_opers(self, pos, lat_mat, center=None, n_orbit=1):
        """
        build symmetry operators
        """
        center = center or [0, 0, 0]
        sym_ops = SymOper(pos, lat_mat, n_orbit=n_orbit)

        sym_ops.build_iden_operator()
        sym_ops.build_rot_operator(center, np.pi, axis=[0, 1, 0])
        # sym_ops.build_rot_operator(center, np.pi, axis=[0, 0, 1])
        sym_ops.build_refl_operator(center, [0, 0, 1])
        # sym_ops.build_refl_operator(center, [0, 1, 0])
        sym_ops.build_refl_operator(center, [1, 0, 0])
        self.sym_ops = sym_ops


class LineGroup(object):
    """ Line group class
    """
    def __init__(self):
        """ tool tip missing
        """
        self.generators = []
        self.orders = []
        self.operators = []
        self.indice = []
        self.irreps = []

    def _set_generator(self, gen_list):
        """generators of line group
        """
        TOL = 1E-5

        for gen in gen_list:
            self.generators.append(gen)
            op = np.eye(len(gen))
            order = 1
            while True:
                op = np.dot(op, gen)
                diff = op - np.eye(len(op))
                if (np.abs(diff) < TOL).all():
                    self.orders.append(order)
                    break
                else:
                    order +=1

    def _get_operator(self, indice):
        """return operator 
        prod gen[i]^indice[i]
        """
        def _power_oper(op, power):
            if power == 0:
                return np.eye(len(op))
            else:
                return np.dot(op, _power_oper(op, power - 1))

        gens = self.generators

        op = np.eye(len(gens[0]))
        for i, gen in enumerate(gens):
            op = np.dot(op, _power_oper(gen, indice[i]))
        return op

    def _build_operators(self):
        """build whole operators using generator
        """
        indice = [range(order) for order in self.orders]
        # print indice
        from itertools import product

        for ind in product(*indice):
            gen_list = self.generators
            op = np.eye(len(gen_list[0]))

            op = self._get_operator(ind)
            criterion = np.array([np.linalg.norm(op - op_old) 
                                  for op_old in self.operators]) < 1E-5

            if not any(criterion):
                self.operators.append(op)
                self.indice.append(ind)

    def _get_irrep(self, indice):
        """return irrep for operator corresponding indice
        you should override this method
        """
        return 1

    def get_proj_op(self, irrep_i, char_table):
        """projection operator to irreducible representation
        """
        # char_table = self.char_table
        sym_ops = self.operators

        proj_mat = np.zeros(sym_ops[0].shape, dtype=np.cfloat)

        for i, op in enumerate(sym_ops):
            proj_mat += char_table[irrep_i, i] * op
        return proj_mat


class L_4m2m(LineGroup):
    """Line group L-4m2m for n_fold=2
    Args:
        norm_vec: 3-length list indicating norm_vec of delta_v

    Prams:
        indice: list of powers or generators that generates operators
                [C_2, S_2n, sigma_v]
    """
    def __init__(self, pos, lat_mat, n_orbit,
                 cv_norm_vec, s2n_norm_vec, center=None):
        """ tool tip missing
        """
        self.generators = []
        self.orders = []
        self.operators = []
        self.irreps = []
        self.indice = []
        self.char_table = None
        self.dim_irrep = []

        center = center or [0, 0, 0]

        sym_ops = SymOper(pos, lat_mat, n_orbit=n_orbit)

        sym_ops.build_rot_refl_operator(center, s2n_norm_vec, np.pi / 2)
        sym_ops.build_refl_operator(center, cv_norm_vec)
        
        self._set_generator(sym_ops.operators)
        self._build_operators()
        self.order = len(self.operators)

    def _build_operators(self):
        """build whole operators using generator
        """
        def _power_oper(op, power):
            if power == 0:
                return np.eye(len(op))
            else:
                return np.dot(op, _power_oper(op, power - 1))

        generators = self.generators

        # identity
        self.operators.append(_power_oper(generators[0], 0))
        self.indice.append([0, 0, 0])
        # C_2
        self.operators.append(_power_oper(generators[0], 2))
        self.indice.append([1, 0, 0])

        # sigma_v
        self.operators.append(_power_oper(generators[1], 1))
        self.indice.append([0, 0, 1])
        # sigma_v'
        self.operators.append(np.dot(_power_oper(generators[1], 1), 
                                     _power_oper(generators[0], 2)))
        self.indice.append([1, 0, 1])

        # S_2n
        self.operators.append(_power_oper(generators[0], 1))
        self.indice.append([0, 1, 0])
        # S_2n'
        self.operators.append(_power_oper(generators[0], 3))
        self.indice.append([0, 3, 0])

        # sigma_v S_2n
        self.operators.append(np.dot(_power_oper(generators[1], 1), 
                                     _power_oper(generators[0], 1)))
        self.indice.append([0, 1, 1])
        # sigma_v' S_2n
        self.operators.append(np.dot(_power_oper(generators[1], 1), 
                                     _power_oper(generators[0], 3)))
        self.indice.append([1, 1, 1])

    def _get_operator_extended(self, op, trans, trans_gen, ind):
        # size = len(op)
        # print size
        # op_trans = np.zeros((size * trans_gen, size * trans_gen))

        # for trans_i in range(trans_gen):
        #     st_i, end_i = trans_i * size, (trans_i + 1) * size
        #     op_trans[st_i: end_i, st_i: end_i] = op.copy()

        iden = np.eye(trans_gen)
        
        iden = np.roll(iden, -trans, axis=1)
        if ind[1] % 2 == 1:
            iden = np.fliplr(iden)

        op_trans = np.kron(iden, op)

        return op_trans

    def _get_operators_kpt(self, trans, kpt):
        """return the list of operators for selected k-points, and translation
        """
        def get_trans_op(trans, ind_op):
            """new translation for the operator
            trans_new = (trans + 1) * -1, for Ud
                        trans           , otherwise
            """
            # return trans
            if ind_op[1] == 0:
                return trans
            else:
                return ((trans + 1) * -1 ** ind_op[1]) #% tran_gen

        from fractions import Fraction
        if kpt > 0:
            tran_gen = Fraction(1 / kpt).limit_denominator(100).numerator
        else:
            tran_gen= 1

        op_extend_list = []
        operators = self.operators
        indice = self.indice

        for op_i, op in enumerate(operators):
            trans_op = get_trans_op(trans, indice[op_i])
            # print trans_op
            op_extend = self._get_operator_extended(op, trans_op, tran_gen, indice[op_i])
            op_extend_list.append(op_extend)

        return op_extend_list

    def get_wavefunc_extend(self, wavefunc, kpt):
        """return extended wave function with translations
        for kpt = m/n * 2 * pi:
            exp(i kpt * 0a) * wf, exp(i kpt * 1a) * wf, ... exp(i kpt * na) * wf
        """
        from fractions import Fraction
        if kpt > 0:
            tran_gen = Fraction(1 / kpt).limit_denominator(100).numerator
        else:
            tran_gen= 1

        wavefunc_extend = np.empty(0)
        for tran in range(tran_gen):
            wavefunc_extend = np.append(wavefunc_extend, 
                                        wavefunc * np.exp(2j*np.pi*kpt*tran))
        return wavefunc_extend

    def _get_irrep_gens(self, k, m, p_u, p_h, p_v):
        """return generators of irrep
        see 
            M. Damnjonovic and I. Milosevic, 
            "Line Groups in Physics: Theory and Applications to Nanotures and 
            Polymers", Lect. Notes Phys. 801 (Springer Berlin Heidelberg 2010)
        Args:
            k: k vector [0 ~ 0.5] pi is ommited
            m: angular momentum, [0, 1]
            p_[u, h, v]: parities
        """
        def _E(k, m, p_u, p_h, p_v):
            expik = np.exp(2j*np.pi*k)
            chars = [np.array([[expik]]),
                     expik*np.eye(2),
                     np.array([[expik, 0], [0, 1/expik]])]
            return chars

        def _C_2(k, m, p_u, p_h, p_v):
            expm_2 = np.exp(1j*np.pi*m)
            chars = [np.array([[1]]),
                     -np.eye(2),
                     expm_2*np.eye(2)]
            return chars

        def _U_d(k, m, p_u, p_h, p_v):
            anti_eye = np.matrix('0 1; 1 0')
            chars = [np.array([[p_u]]), anti_eye, anti_eye]
            return chars

        def _sigma_v(k, m, p_u, p_h, p_v):
            chars = [np.array([[p_v]]),
                     np.matrix('1 0; 0 -1'), 
                     p_v*np.array([[1., 0], [0, np.exp(1j*np.pi*m)]])]
            return chars

        def _get_irrep(irrep_gen, powers):
            """ return powers of charaters
            """
            from copy import deepcopy
            irrep_list = deepcopy(irrep_gen[0])
            for gen_i, irrep_gen in enumerate(irrep_gen[1:]):
                for irrep_i, irrep in enumerate(irrep_gen):
                    irrep_list[irrep_i] = np.matrix(irrep) ** powers[gen_i] *\
                                          np.matrix(irrep_list[irrep_i])
            return irrep_list

        irrep_gen = [_E, _C_2, _U_d, _sigma_v]
        irrep_gen = [func(k, m, p_u, p_h, p_v) for func in irrep_gen]

        irrep_list = [[None] * len(self.operators) for _ in range(3)] 

        for op_i, op in enumerate(self.operators):
            ind = self.indice[op_i]
            irreps = _get_irrep(irrep_gen, ind)

            for irrep_kind, irrep in enumerate(irreps):
                irrep_list[irrep_kind][op_i] = irrep
        return irrep_list

    def get_char_table(self, tran, k, irrep_kind, 
                       m_list, p_u_list, p_h_list, p_v_list):
        from itertools import product

        char_table = []
        for m, p_u, p_h, p_v in product(m_list, p_u_list, p_h_list, p_v_list):
            irreps = self._get_irrep_gens(k * tran, m, p_u, p_h, p_v)

            chars = [[np.trace(irrep) for irrep in irrep_row]
                     for irrep_row in irreps]
            chars = np.array(chars)
            char_table.append(chars[irrep_kind, :])

            self.dim_irrep += [len(irreps[irrep_kind][0])]
        return np.array(char_table)

    def get_proj_op(self, operators, irrep_i, char_table):
        """projection operator to irreducible representation
        """
        sym_ops = operators

        proj_mat = np.zeros(sym_ops[0].shape, dtype=np.cfloat)

        for i, op in enumerate(sym_ops):
            proj_mat += np.conjugate(char_table[irrep_i, i]) * op
        return proj_mat

    def find_irrep(self, kpt, wavefunc):
        """find irrep for wave frunctions
        """
        from fractions import Fraction
        if kpt > 0:
            tran_gen = Fraction(1 / kpt).limit_denominator(100).numerator
        else:
            tran_gen= 1

        wavefunc_extend = self.get_wavefunc_extend(wavefunc, kpt)
        w = np.zeros((10, len(wavefunc_extend)), dtype=np.cfloat)

        # for trans in [0]:
        self.dim_irrep = []
        for trans in range(tran_gen):
            if kpt in [0, 0.5, -0.5]:
                char_table = lg.get_char_table(trans, kpt,
                                               0, [0], [1, -1], [0], [1, -1])
                char_table = np.append(char_table,
                                lg.get_char_table(trans, kpt,
                                                  1, [1], [0], [0], [0]),
                                axis=0)
            else:
                char_table = lg.get_char_table(trans, kpt,
                                               2, [0, 1], [0], [0], [1, -1])
            # calc w
            # print np.rint(char_table[:, :] * 1E3) / 1E3
            operators = self._get_operators_kpt(trans, kpt)

            for irrep in range(len(char_table)):
                proj_op = lg.get_proj_op(operators, irrep, char_table)
                w[irrep, :] += np.dot(proj_op, wavefunc_extend)

        proj = np.zeros(len(char_table), dtype=np.cfloat)
        # wave_indice = np.argsort(np.absolute(wavefunc_extend))
        # wave_indice = np.argsort(wavefunc_extend)
        wf_len = np.linalg.norm(wavefunc_extend)
        for irrep in range(len(char_table)):
            proj[irrep] = np.linalg.norm(w[irrep]) /\
             (len(self.operators) * tran_gen / self.dim_irrep[irrep])
            # # for ind in range(len(wave_indice)/2):
            # for ind in range(2):
            # # for ind in range(len(wavefunc_extend)):
            #     ind = wave_indice[-ind-1]
            #     if np.abs(w[irrep, ind]) < 1E-3:
            #         ratio = 1E100
            #     else:
            #         ratio = (wavefunc_extend[ind] / w[irrep, ind] * \
            #                 len(self.operators)) * tran_gen / \
            #                 self.dim_irrep[irrep]
            #         # print self.dim_irrep[irrep]
            #     proj[irrep] += 1/ratio / 2

        return proj / wf_len#/ len(wavefunc) * 2


class L_3m(LineGroup):
    """Line group L-3m for n_fold=3
    Args:
        norm_vec: 3-length list indicating norm_vec of delta_v
    """
    def __init__(self, pos, lat_mat, n_orbit,
                 cv_norm_vec, s2n_norm_vec, center=None):
        """ tool tip missing
        """
        self.generators = []
        self.orders = []
        self.operators = []
        self.irreps = []
        self.indice = []
        self.char_table = None

        if center is None:
            center = [0, 0, 0]

        sym_ops = SymOper(pos, lat_mat, n_orbit=n_orbit, tol=5E-2)
        sym_ops.build_rot_operator(center, 2*np.pi/3)
        sym_ops.build_refl_operator(center, cv_norm_vec)
        
        self._set_generator(sym_ops.operators)
        self._build_operators()

    def _build_operators(self):
        """build whole operators using generator
        """
        def _power_oper(op, power):
            if power == 0:
                return np.eye(len(op))
            else:
                return np.dot(op, _power_oper(op, power - 1))

        generators = self.generators

        # identity
        self.operators.append(_power_oper(generators[0], 0))
        self.indice.append([0, 0])

        # C_3
        self.operators.append(_power_oper(generators[0], 1))
        self.indice.append([1, 0])
        # C_3^2
        self.operators.append(_power_oper(generators[0], 2))
        self.indice.append([2, 0])

        # sigma_v
        self.operators.append(_power_oper(generators[1], 1))
        self.indice.append([0, 1])
        # sigma_v'
        self.operators.append(np.dot(_power_oper(generators[1], 1), 
                                     _power_oper(generators[0], 1)))
        self.indice.append([1, 1])
        # sigma_v''
        self.operators.append(np.dot(_power_oper(generators[1], 1), 
                                     _power_oper(generators[0], 1)))
        self.indice.append([2, 1])

    def _get_char_gens(self, k, m, p_v):
        """return generators of charecters
        see 
            M. Damnjonovic and I. Milosevic, 
            "Line Groups in Physics: Theory and Applications to Nanotures and 
            Polymers", Lect. Notes Phys. 801 (Springer Berlin Heidelberg 2010)
        Args:
            # op_i: index for operators; 0 for E, 1 for C_3, 2 for C_v
            k: k vector [0 ~ 0.5] pi is ommited
            m: angular momentum, [0, 1]
            p_[u, h, v]: parities
        """
        def _E(k, m, p_v):
            expik = np.exp(2j*np.pi*k)
            # expik = 1.+0j
            chars = [np.array([[expik]]),
                     expik*np.eye(2)]
            return chars

        def _C_3(k, m, p_v):
            expm = np.exp(2j*np.pi*m/3)
            chars = [np.array([[expm]]),
                     np.array([[expm, 0],[0, 1/expm]])]
            return chars

        def _sigma_v(k, m, p_v):
            chars = [np.array([[p_v]]),
                     np.matrix('0 1; 1 0')]
            return chars

        def _get_irrep(irrep_gen, powers):
            """ return powers of charaters
            """
            from copy import deepcopy
            irrep_list = deepcopy(irrep_gen[0])
            for gen_i, irrep_gen in enumerate(irrep_gen[1:]):
                for irrep_i, irrep in enumerate(irrep_gen):
                    irrep_list[irrep_i] *= irrep ** powers[gen_i]
            return irrep_list

        irrep_gen = [_E, _C_3, _sigma_v]
        irrep_gen = [func(k, m, p_v) for func in irrep_gen]

        char_table = np.zeros((2, len(self.operators)), dtype=np.cfloat)

        for op_i, op in enumerate(self.operators):
            ind = self.indice[op_i]
            chars = _get_irrep(irrep_gen, ind)
            for irrep_kind, char in enumerate(chars):
                char_table[irrep_kind, op_i] = np.trace(char)

        return char_table

    def get_char_table(self, tran, k, irrep_kind, 
                       m_list, p_v_list):
        from itertools import product

        char_table = []
        for m, p_v in product(m_list, p_v_list):
            chars = self._get_char_gens(k * tran, m, p_v)
            char_table.append(chars[irrep_kind, :])

        return np.array(char_table)

    def find_irrep(self, kpt, wavefunc):
        """find irrep for wavefunc
                how to?!
                del_kpt = (kpt_prim - kpt_sc)
        1. build char_table w.r.t k
        2. build projection operator
        3. projection
        """

        from fractions import Fraction
        if kpt > 0:
            tran_gen = Fraction(1 / kpt).limit_denominator(100).numerator
        else:
            tran_gen = 1
        # tran_gen = 1
        print 'tran_gen', tran_gen
        w = np.zeros((10, len(wavefunc)), dtype=np.cfloat)

        for tran in range(tran_gen):
            char_table = lg.get_char_table(tran, kpt, 0, [0], [1, -1])

            char_table = np.append(char_table,
                                   lg.get_char_table(tran, kpt, 1, [1], [0]),
                                   axis=0)
            for irrep in range(len(char_table)):
                proj_op = lg.get_proj_op(irrep, char_table)
                w[irrep, :] += np.dot(proj_op,
                                      wavefunc * np.exp(2j*np.pi*kpt*tran))


        for irrep in range(len(char_table)):
            # proj_op = lg.get_proj_op(irrep, char_table)
            # w = np.dot(proj_op, wavefunc)
            ind = np.argmax(wavefunc)
            if np.abs(w[irrep, ind]) < 1E-5:
                print 0
            else:
                print (wavefunc[ind] / w[irrep, ind] * \
                       tran_gen * len(self.operators))


if __name__ == '__main__':
    # 001 R3
    path = '/home/users/nwan/02Project/14_DIELC_NW/00_NW/001/00_high'
    # path = '/home/users/nwan/02Project/14_DIELC_NW/00_NW/001/00'
    # CBM 60

    # 110 Si24
    # path = '~/cj01/00_NW/110/Si24H16/'

    # 111 R3
    path = '/home/users/nwan/cj01/00_NW/R3/00_pristine/unfold/1U_20'
    # CBM 91

    struct, Kpts, Eigs, Proj_cmplx = read_elec_structure(path)

    kpt_i = -1
    band_i = 0 # CBM
    n_orbit = 4

    kpt = Kpts[kpt_i, 2]
    # print 'kpt', kpt

    lat_mat = struct.lattice.get_matrix()
    n_Si = struct.get_n_elements()[0]
    pos = struct.get_pos()[:n_Si]

    
    # lg = L_4m2m(pos, lat_mat, n_orbit, [1, 1, 0], [0, 0, 1])
    lg = L_3m(pos, lat_mat, n_orbit, [1, 0, 0], [0, 0, 1])
    for band_i in [1]:
    # for band_i in range(10):
        # for kpt_i in [-1]:
        for kpt_i in range(len(Kpts)):
            kpt = Kpts[kpt_i, 2]
            # print 'kpt', kpt

            wavefunc = Proj_cmplx[kpt_i, band_i, :n_Si, :n_orbit]
            wavefunc = np.ravel(wavefunc[:, :n_orbit])

            proj = lg.find_irrep(kpt, wavefunc)
            proj = np.rint(proj * 1E5) / 1E5

            ind = np.argmax(np.absolute(proj))

            if not proj[ind].imag == 0:
                print '--', band_i, ind, proj
            else:
                # pass
                print band_i, ind, proj
