"""Structure, Lattice, Atom class"""
import numpy as np
from vasp_io import writePOSCAR, readCONTCAR, writeKPOINTS


class Structure(object):
    """ Defines atomic structure"""
    def __init__(self, lattice, atoms, dynamics=None):
        """
        Args:
            lattice:
                Lattice object
            atoms:
                list of Atom object
        """
        assert isinstance(lattice, Lattice), 'not Lattice object'
        assert isinstance(atoms, list), 'atoms is not list'
        assert isinstance(atoms[0], Atom), 'atom is not Atom object'
        self.lattice = lattice
        self.atoms = atoms
        self.finger_print = None
        self.dynamics = dynamics or [True, True, True]
        # self.finger_print = self.get_finger_print()
    
    def write_kpoints(self, target_dir, file_name='KPOINTS', \
                      kspacing=0.3, span=[True, True, True]):
        '''
        if span is false only 1 kpoint is used 
        for the corresponding direction.
        '''
        rec_lat_mat = self.lattice.get_rec_lattice()

        n_kpt = [max(1, 2 * np.pi * np.linalg.norm(rec_vec) / kspacing) \
                 for rec_vec in rec_lat_mat]

        n_kpt = [int(item) if span[index] else 1
                 for index, item in enumerate(n_kpt)]

        writeKPOINTS("{}/{}".format(target_dir, file_name), n_kpt)


    def write_poscar(self, target_dir, file_name='POSCAR', l_selective=False):
        """
        write structure with vasp poscar format

        Args:
            target_dir:
                directory path to write POSCAR
            file_name:
                you may want to write POSCAR_1, probably not
            l_selective:
                if selective dynamics or not
        """
        self.sort_atom()
        lat_vecs = self.lattice.get_matrix()
        dynamics = ['T' if l_move else 'F' for l_move in self.dynamics]
        if l_selective:
            atom_set = [atom.to_list() + [dynamics] for atom in self.atoms]
        else:
            atom_set = [atom.to_list() for atom in self.atoms]
        writePOSCAR("{}/{}".format(target_dir, file_name), 1, lat_vecs,
                    atom_set, lSelective=l_selective, lDirect=True)

    def get_elements(self):
        """return list of elements eg) ['Si', 'O']"""
        from collections import OrderedDict
        return list(OrderedDict.fromkeys([atom.element for atom in self.atoms]))

    def get_n_elements(self):
        """return list of number of atoms"""
        elements = self.get_elements()
        n_elements = []

        el_atoms = [atom.element for atom in self.atoms]

        for element in elements:
            n_elements.append(el_atoms.count(element))

        return n_elements

    def sort_atom(self, elements=None):
        """
        sort atoms in the order of elements

        Args:
            elements:
                list of elements; eg) ['Si', 'O']
                if not given it use self.get_elements()
                so usually don't give it
        """
        elements = elements or self.get_elements()
        atoms = self.atoms
        atoms = sorted(atoms, key=lambda atom: elements.index(atom.element))
        self.atoms = atoms

    def get_pair_corr(self, bin_spacing=0.1, r_max=5, multi=None):
        """
        return list of pair correlation function of structure
        [pc_11, pc_12, ....]
        Args:
            bin_spacing:
                bin_spacing
            r_max:
                calculate inter-atomic distance up to r_max
            multi:
                supercell multiplication (I hope you know my intention.)
                default (2, 2, 2)
        """
        multi = multi if multi is not None else np.array([2, 2, 2])
        elements = self.get_elements()
        pair_corr = []
        for i in np.arange(len(elements)):
            for j in np.arange(i, len(elements)):
                element_1 = elements[i]
                element_2 = elements[j]
                radius, rdf = self._get_pair_corr_ab(element_1, element_2,
                                                     r_max, bin_spacing, multi)
                pair_corr.append(rdf)

        return radius, pair_corr

    @staticmethod
    def _multi_atoms(atoms, multi):
        '''multiply atoms'''
        multi_atoms = []
        for n_x in range(multi[0]):
            for n_y in range(multi[1]):
                for n_z in range(multi[2]):
                    for atom in atoms:
                        multi_atoms.append(Atom(atom.element, atom.pos +
                                                np.array([n_x, n_y, n_z])))
        return multi_atoms

    def _get_pair_corr_ab(self, element_1, element_2,
                          r_max, bin_spacing, multi):
        """pair correlation between element_1 and element_2"""

        volume = self.lattice.get_volume()

        multi_atoms = []
        # multiply supercell
        multi_atoms = self._multi_atoms(self.atoms, multi)

        r = []
        for atom1 in self.atoms:
            for atom2 in multi_atoms:
                if atom1.element == element_1 and atom2.element == element_2:
                    diff = atom1.pos - atom2.pos
                    if np.linalg.norm(diff) == 0:
                        continue
                    diff = diff - np.round(diff / multi) * multi
                    diff = np.dot(self.lattice.get_matrix().T, diff)
                    diff = np.linalg.norm(diff)
                    r.append(diff)

        if (np.array(r) < r_max).any():
            hist, bin_edges = np.histogram(r, bins=int(r_max / bin_spacing),
                                           range=(0, r_max), density=True)
            rdf = hist / 4. / np.pi * volume / bin_spacing  / \
                  (bin_edges + bin_spacing / 2.)[:-1] ** 2
            return bin_edges[:-1], rdf
        else:
            bins = np.linspace(0, r_max, num=int(r_max / bin_spacing),
                               endpoint=False)
            rdf = np.zeros((len(bins)))
            return bins, rdf

    def _get_finger_print_atom_b(self, atom_1_index, element_2, bin_spacing=0.1,
                                 r_max=5, multi=None):
        """ see Organov's paper"""
        multi = multi or np.array([2, 2, 2])
        volume = self.lattice.get_volume()

        multi_atoms = self._multi_atoms(self.atoms, multi)

        r = []
        atom1 = self.atoms[atom_1_index]
        for atom2 in multi_atoms:
            if atom2.element == element_2:
                diff = atom1.pos - atom2.pos
                if np.linalg.norm(diff) == 0:
                    continue
                diff = diff - np.round(diff / multi) * multi
                diff = np.dot(self.lattice.get_matrix().T, diff)
                diff = np.linalg.norm(diff)
                r.append(diff)

        if (np.array(r) < r_max).any():
            hist, bin_edges = np.histogram(r, bins=int(r_max / bin_spacing),
                                           range=(0, r_max), density=True)
            rdf = hist / 4. / np.pi * volume / bin_spacing  / \
                  (bin_edges + bin_spacing / 2.)[:-1] ** 2
            return bin_edges[:-1], rdf - 1
        else:
            bins = np.linspace(0, r_max, num=int(r_max / bin_spacing),
                               endpoint=False)
            fp = np.zeros((len(bins)))
            fp.fill(-1.)
            return bins, fp

    def get_finger_print(self, bin_spacing=0.1, r_max=5, multi=None):
        """
        list of finger print
        see Organov's paper
        """
        # TODO(Sunghyun Kim): finger print should include r or independent on r
        if hasattr(self, 'finger_print') and self.finger_print is not None:
            return [], self.finger_print

        multi = multi or np.array([2,2,2])
        r, pair_corr = self.get_pair_corr(bin_spacing=bin_spacing, r_max=r_max,
                                          multi=multi)
        finger_print = []

        for pc in pair_corr:
            finger_print.append(pc - 1.)
        self.finger_print = finger_print
        return r, finger_print

    def get_pos(self):
        """return positions of atoms"""
        return [atom.pos for atom in self.atoms]

    def get_center_mass(self):
        "return center of mass"
        cm = np.zeros(3)
        pos = self.get_pos()
        for p in pos:
            cm += p
        return cm / len(pos)

    def set_center_mass(self, center=None):
        "set ceter of mass"
        center = center or np.array([0, 0, 0])
        cm = self.get_center_mass()
        for atom in self.atoms:
            atom.pos -= cm
            atom.pos += center

    def get_norm_axis(self):
        """
        return one of lattice vector index
        that is normal to atomic layers
        """
        lattice = self.lattice.get_matrix()
        lat_plane_norm = np.zeros(3)

        for atom in self.atoms:
            pos = atom.pos
            # print pos
            for j, p in enumerate(pos):
                if p > 0.5:
                    pos[j] -= 1.
            lat_plane_norm += np.abs(pos)
            print pos
        return np.argmin(lat_plane_norm)

    def get_degree_order(self, bin_spacing=0.1, r_max=5, multi=None):
        """see Organov's paper or ask the author"""
        multi = multi or np.array([2, 2, 2])
        n_elements = self.get_n_elements()
        volume = self.lattice.get_volume()
        r, fp = self.get_finger_print(bin_spacing=bin_spacing, r_max=r_max,
                                      multi=multi)
        d_order = bin_spacing / (volume / np.sum(n_elements)) ** (1 / 3.) * \
                  np.linalg.norm(fp) ** 2
        return d_order

    def get_local_order(self, bin_spacing=0.1, r_max=5.):
        """see Organov's paper or ask the author"""
        local_orders = []
        elements = self.get_elements()
        n_elements = self.get_n_elements()
        volume = self.lattice.get_volume()

        for atom_1_index, atom in enumerate(self.atoms):
            r, lo_element = self._get_finger_print_atom_b(
                              atom_1_index, elements[0], bin_spacing, r_max)
            lo_element = 1. * n_elements[0] / np.sum(n_elements) * bin_spacing \
                         / (volume / np.sum(n_elements)) ** (1 / 3.) \
                         * np.linalg.norm(lo_element) ** 2
            lo = lo_element

            for i, element in enumerate(elements[1:]):
                r, lo_element = self._get_finger_print_atom_b(atom_1_index, 
                                       element, bin_spacing, r_max)
                lo_element = 1. * n_elements[i] / np.sum(n_elements) * \
                             bin_spacing / (volume / np.sum(n_elements)) ** \
                            (1 / 3.) * np.linalg.norm(lo_element) ** 2
                
                lo += lo_element

            lo = np.sqrt(lo)

            local_orders.append(lo)
        return local_orders

    def distance_matrix(self):
        elements = self.get_elements()
        lat_vecs = self.lattice.get_matrix()
        volume = self.lattice.get_volume()
        atoms = self.atoms
        d_mat = np.zeros((len(atoms), len(atoms)))

        for i, atom1 in enumerate(atoms):
            for j, atom2 in enumerate(atoms):
                diff = get_dist(atom1.pos, atom2.pos, lat_vecs)
                d_mat[i, j] = diff

        return d_mat

    def distance_matrix_vec(self):
        elements = self.get_elements()
        lat_vecs = self.lattice.get_matrix()
        volume = self.lattice.get_volume()
        atoms = self.atoms
        d_mat = np.zeros((len(atoms), len(atoms), 3))

        for i, atom1 in enumerate(atoms):
            for j, atom2 in enumerate(atoms):
                diff = get_dist_vec(atom1.pos, atom2.pos, lat_vecs)
                d_mat[i, j, :] = diff

        return d_mat

    def frac_dist_mat(self, span=None, offset=None):
        """
            Args:
                span: list(3) of boolean if image cell is considered or not
        """
        if span is None:
            span = [1, 1, 1]
        if offset is None:
            offset = np.array([0, 0, 0])
        elements = self.get_elements()
        atoms = self.atoms
        pos = np.array([(p - offset) - np.dot(np.round(p - offset), np.diag(span))
                        for p in self.get_pos()])

        d_mat = np.zeros((len(atoms), len(atoms), 3))

        for i, atom1 in enumerate(atoms):
            for j, atom2 in enumerate(atoms):
                diff = atom1.pos - atom2.pos
                diff = pos[j] - pos[i]
                diff = diff - np.dot(np.round(diff), np.diag(span))
                d_mat[i, j, :] = diff
        return d_mat

    def get_bond_eig(self):
        struct = self.copy()
        atoms = struct.atoms
        struct.atoms = sorted(atoms, key =lambda atom: np.linalg.norm(atom.pos))

        import scipy.linalg
        BOND_LENGTH = 2.1
        d_mat = np.array(struct.distance_matrix())
        bond_mat = d_mat < BOND_LENGTH
        # print bond_mat[7]
        # d_mat += np.eye(len(d_mat))
        # inv_d_mat = 1. / d_mat
        # inv_d_mat -= np.eye(len(d_mat))
        return scipy.linalg.eig(bond_mat)
        # return inv_d_mat

    def get_inv_dist_eig(self):
        import scipy.linalg
        struct = self.copy()
        atoms = struct.atoms
        struct.atoms = sorted(atoms, key =lambda atom: np.linalg.norm(atom.pos))
        
        d_mat = np.array(struct.distance_matrix())
        d_mat += np.eye(len(d_mat))
        inv_d_mat = 1. / d_mat
        inv_d_mat -= np.eye(len(d_mat))
        return scipy.linalg.eig(inv_d_mat)
        # return inv_d_mat


    def get_min_dist(self):
        '''return minimum distance btw atoms'''
        dist_mat = self.distance_matrix()
        min_dist = np.min(dist_mat + 1E10 * np.identity(len(dist_mat)))
        return min_dist

    def copy(self):
        import copy
        return copy.deepcopy(self)

    def _add_vac(self, vac, center=[0.5, 0.5, 0.5]):
        """
        add vacuum to struct
        center
        """
        atoms = self.atoms
        bravias_lat = np.array(self.lattice.get_list())

        for atom in atoms:
            pos = np.array(atom.pos)
            offset = (np.array(center) + np.array(vac))/\
                     (bravias_lat[:3] + np.array(vac))
            atom.pos = pos * bravias_lat[:3] / (bravias_lat[:3] + np.array(vac))
            for i, coord in enumerate(pos):
                if coord > center[i]:
                    atom.pos[i] += offset[i]

        bravias_lat[:3] += np.array(vac)

        self.lattice = Lattice(*bravias_lat)
        self.atoms = atoms

    @staticmethod
    def read_contcar(dir_name, file_name = 'CONTCAR'):
        lat_const, lattice_vecs, atom_set_direct, dynamics = \
            readCONTCAR(fileName = dir_name + '/' + file_name, rtspecies = False)

        atoms = []
        for a in atom_set_direct:
            atoms.append(Atom(a[0], a[1]))


        bravais_lat = np.array(lattice_vecs)# lat_const
        lattice = Lattice(bravais_lat, lat_const)

        # TODO(Sunghyun Kim): atom specific dynamics
        structure = Structure(lattice, atoms, dynamics=dynamics[0])

        return structure


class Lattice:
    """ represent lattice of structure """
    def __init__(self, *args):
        """
        Args:
            a, b, c, alpha, beta, gamma
        """
        if len(args) == 1 and isinstance(args[0], int):
            self.rand_lattice(args)
        elif len(args) == 1 and (isinstance(args[0], list) or  
                               isinstance(args[0], np.ndarray)):
            a, b, c, alpha, beta, gamma = args[0]
            self.a = a
            self.b = b 
            self.c = c
            self.alpha = alpha
            self.beta  = beta
            self.gamma = gamma
        elif len(args) > 5:
            a, b, c, alpha, beta, gamma = args
            self.a = a
            self.b = b 
            self.c = c
            self.alpha = alpha
            self.beta  = beta
            self.gamma = gamma
        else:
            matrix, lat_const = args
            self.a, self.b, self.c, self.alpha, self.beta , self.gamma = \
            self._to_list(matrix, lat_const)
    
    def rand_lattice(self, spg_num=1):
        self.a, self.b, self.c, self.alpha, self.beta, self.gamma = self._rnd_bravais_lat(spg_num)

    def get_list(self):
        return self.a, self.b, self.c, self.alpha, self.beta, self.gamma

    def get_matrix(self):
        matrix = self._to_matrix()
        return matrix

    def get_volume(self):
        matrix = self.get_matrix()
        volume = np.dot(matrix[0], np.cross(matrix[1], matrix[2]))
        return volume

    def set_volume(self, vol, fix=None):
        vol_prev = self.get_volume()
        fix = fix or np.array([0, 0, 0])
        assert sum(fix) < 3,\
               'at least one axis should be released'
        power = 1. / (3 - sum(fix))
        if not fix[0]:
            self.a *= (vol / vol_prev) ** power
        if not fix[1]:
            self.b *= (vol / vol_prev) ** power
        if not fix[2]:
            self.c *= (vol / vol_prev) ** power    

    def _bravais_lat_dict(self, spg_num):
        def triclinic():    return 'a, b, c, alpha, beta, gamma'
        def monoclinic():    return 'a, b, c, 90, beta, 90'
        def orthorhobic():    return 'a, b, c, 90, 90, 90'
        def tetragonal():    return 'a, a, c, 90, 90, 90'
        def rhombohedral():    return 'a, a, a, alpha, alpha, alpha'
        def hexagonal():    return 'a, a, c, 90, 90, 120'
        def cubic():    return 'a, a, a, 90, 90, 90'
        if spg_num in range(1, 2 + 1):
            bravias_lat_formula = triclinic()
        elif spg_num in range(3, 15 + 1 ):
            bravias_lat_formula = monoclinic()
        elif spg_num in range(16, 74 + 1):
            bravias_lat_formula = orthorhobic()
        elif spg_num in range(75, 167 + 1):
            bravias_lat_formula = rhombohedral()
        elif spg_num in range(168, 195):
            bravias_lat_formula = hexagonal ()
        else:
            bravias_lat_formula = cubic()
        return bravias_lat_formula

    def _rnd_bravais_lat(self, spg_num = 1, dim = 3, H=20.):
        max_ratio = 5.
        bravias_lat_formula = self._bravais_lat_dict(spg_num)

        if dim == 2 :
            # fix b length 
            b = H
            # random length for a and c
            while True:
                a, c = np.random.rand(2)
                if  1. / max_ratio < a / c < max_ratio :
                    break
        else :
            # random length for all lattice vectors
            while True:
                a, b, c = np.random.rand(3)
                if  1. / max_ratio < a / b < max_ratio \
                and 1. / max_ratio < a / c < max_ratio \
                and 1. / max_ratio < b / c < max_ratio:
                    break
        
        while True:
            # random angle 
            alpha, beta, gamma = np.random.rand(3) * 30 + 60.

            bravias_lat = eval(bravias_lat_formula)
            bravias_lat = np.array([float(i) for i in bravias_lat])

            bravias_lat[3:6] *= np.pi / 180.

            if alpha + beta > gamma and \
               beta + gamma > alpha and \
               gamma + alpha > beta: 
                break 
        
        return bravias_lat

    def _to_matrix(self):
        # see http://en.wikipedia.org/wiki/Fractional_coordinates
        # For the special case of a monoclinic cell (a common case) where alpha = gamma = 90 degree and beta > 90 degree, this gives: <- special care needed
        # so far, alpha, beta, gamma < 90 degree
        a, b, c, alpha, beta, gamma = self.a, self.b, self.c, self.alpha, self.beta, self.gamma

        v = a * b * c * np.sqrt(1. - np.cos(alpha) ** 2 - np.cos(beta) ** 2 - np.cos(gamma) ** 2 + 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma) )

        T = np.zeros((3, 3))
        T = np.array([ \
                  [a, b * np.cos(gamma), c * np.cos(beta)                                                  ] ,\
                  [0, b * np.sin(gamma), c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)] ,\
                  [0, 0                , v / (a * b * np.sin(gamma))                                       ] 
                      ])
        matrix = np.zeros((3, 3))
        matrix[:,0] = np.dot(T, np.array((1, 0, 0)))
        matrix[:,1] = np.dot(T, np.array((0, 1, 0)))
        matrix[:,2] = np.dot(T, np.array((0, 0, 1)))
        return matrix.T

    def _to_list(self, matrix, lat_const):
        """
        see http://en.wikipedia.org/wiki/Fractional_coordinates
        """
        from numpy.linalg import norm

        a = matrix[0] * lat_const
        b = matrix[1] * lat_const
        c = matrix[2] * lat_const

        alpha = np.arctan2(norm(np.cross(b, c)), np.dot(b, c))
        beta  = np.arctan2(norm(np.cross(c, a)), np.dot(c, a))
        gamma = np.arctan2(norm(np.cross(a, b)), np.dot(a, b))

        return norm(a), norm(b), norm(c), alpha, beta, gamma

    def get_rec_lattice(self):
        """
        b_i = (a_j x a_k)/ a_i . (a_j x a_k)
        """
        lat_mat = self.get_matrix()
        rec_lat_mat = np.linalg.inv(lat_mat).T
        return rec_lat_mat

    def __repr__(self):
        _repr = [self.a, self.b, self.c, self.alpha, self.beta , self.gamma]
        _repr = [str(i) for i in _repr]
        return ' '.join(_repr)


class Atom:
    def __init__(self, element, pos, dyn = ('T', 'T', 'T')):
        """
        Object to represent atom
        Args:
            element:
                atomic symbol eg) 'Si'
            pos:
                atom position (fractional coordinate) eg) [0.5, 0.5, 0] 
            dyn:
                fixed position or movable position
        """
        self.element = element
        self.pos = np.array(pos)
        self.dyn = dyn

    def to_list(self):
        return [self.element, self.pos]

def cos_dist(structure1, structure2):
    r1, fp_1 = structure1.get_finger_print()
    r2, fp_2 = structure2.get_finger_print()

    F1_norm = np.linalg.norm(fp_1)
    F2_norm = np.linalg.norm(fp_2)

    dist  = 0.5 * (1. - (np.array(fp_1) * np.array(fp_2)).sum() / F1_norm / F2_norm)
    return dist

def get_dist(pos1, pos2, lat_vecs):
    # p1, p2 direct 
    # return angstrom
    # latConst is included in lat_vecs
    diff = np.array(pos2) - np.array(pos1)
    if np.linalg.norm(diff) ==  0:
        return 0
    diff = diff - np.round(diff)
    #check check check this out
    diff = np.dot(lat_vecs.T,diff)
    diff = np.linalg.norm(diff)
    return diff

def get_dist_vec(pos1, pos2, lat_vecs):
    # p1, p2 direct 
    # return angstrom
    # latConst is included in lat_vecs
    diff = np.array(pos2) - np.array(pos1)
    if np.linalg.norm(diff) ==  0:
        return 0
    diff = diff - np.round(diff)
    #check check check this out
    diff = np.dot(lat_vecs.T,diff)
    return diff

def conv_cart_to_frac(pos1, lat_mat):
    # latConst is included in lat_mat
    a = np.dot(pos1, lat_mat[0,:]) / np.linalg.norm((lat_mat[0,:])) ** 2 
    b = np.dot(pos1, lat_mat[1,:]) / np.linalg.norm((lat_mat[1,:])) ** 2
    c = np.dot(pos1, lat_mat[2,:]) / np.linalg.norm((lat_mat[2,:])) ** 2
    return np.array((a, b, c))


if __name__ == '__main__':
    import timeit
    # import matplotlib.pyplot as plt
    # import gen_pos
    # for i in range(3):
    #     # for j in range(5)[i + 1:]:
    #     st0 = gen_pos.read_contcar('./test_B_run/00005', 'CONTCAR')
    #     st1 = gen_pos.read_contcar('./test_B_run/results', 'CONTCAR_00'+str(i))
    #     print cos_dist(st1, st0)
    struct = Structure.read_contcar('./', 'CONTCAR')
    struct.write_kpoints('./', span=[True, 0, False])
    setup = '''
from structure import Structure, Lattice, Atom
import numpy as np
lattice = [5.0544, 5.6199, 6.9873, 90 * np.pi/180, 90 * np.pi/180, 90 * np.pi/180]
lattice = Lattice(lattice)
# print lattice.get_matrix()
# print lattice.get_list()
atoms = [Atom('B', np.random.rand(3)) for _ in range(28)]
    '''
    # print timeit.timeit('struct = Structure(lattice, atoms)', setup=setup, number=2)
    # d_mat = st.distance_matrix()
    # print d_mat
    #print st.get_elements()
    #print st.get_n_elements()
    #print st.get_degree_order()
    #print st._get_pair_corr_ab('Mo','S',bin_spacing = 0.1, r_max = 5, multi = np.array([2, 2, 2]))
    # print st.get_finger_print()
    #print st._get_finger_print_atom_b(2, 'Mo')
    #print st.get_local_order(r_max = 5.)
