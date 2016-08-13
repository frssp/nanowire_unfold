'''
draw Brillouin zone 
'''
# from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import matplotlib.pyplot as plt
import numpy as np
from polyhedron import Vrep, Hrep
import mayavi.mlab as mlab
mlab.options.backend = 'simple'


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

def get_mesh(rec_lat_mat):
    mesh = []
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            for k in [-1, 0, 1]:
                if i==j==k==0:
                    continue
                pos = np.dot(np.array([i, j, k], dtype=float), rec_lat_mat)
                mesh.append(pos)
    return mesh

def get_dd_matrix(rec_lat_mat):
    '''
    double description method
    see ftp://ftp.ifor.math.ethz.ch/pub/fukuda/cdd/cddlibman/node3.html
    '''
    mesh = get_mesh(rec_lat_mat)

    A = np.zeros((len(mesh), 3))
    b = np.zeros((len(mesh)))
    # print A.shape, b.shape
    for i, neigh in enumerate(mesh):
        # b[i] = np.dot(neigh, neigh) / 2.
        b[i] = np.linalg.norm(neigh) / 2.
        A[i, :] = np.array(neigh) / np.linalg.norm(neigh)

    return A, b

def get_edge(adj_list):
    '''
    return edge(a pair of indices of vertices)

    '''
    edges = set([])
    for i, adj in enumerate(adj_list):
        for pair in [(i, j) for j in adj]:
            if pair[0] > pair[1]:
                pair = (pair[1], pair[0])
            edges.add(pair)
    return edges


def draw_edges(ax, verts, edges, color=None, ls_list=None, diff=0):
    """draw edges
    """
    color = color or 'k'
    view_vec = np.array([1, -1, 0])
    print len(edges)
    for edge_i, edge in enumerate(edges):
        pos = [verts[i] for i in edge]
        x = [pos[0][0], pos[1][0]]
        y = [pos[0][1], pos[1][1]]
        z = [pos[0][2], pos[1][2]]
        pos = [(pos[0][0] + pos[1][0])/2., (pos[0][1] + pos[1][1])/2., (pos[0][2] + pos[1][2])/2.]

        depth = np.dot(pos, view_vec)
        zorder = depth
        if ls_list is not None:
            ls = ls_list[edge_i]
        else:
            ls = '-'

        if ls == '-':
            zorder = diff
            lw = 3
        else:
            zorder =  - diff
            lw = 2
        # ax.color_cycle(['k'])
        # temp = ax.plot(x, y, z, ls=ls, lw=lw, alpha=1, zdir='z', zorder=zorder)
        mlab.plot3d(x, y, z, color=black)
        # temp = ax.plot(x, y, z, ls=ls, lw=lw, alpha=1, zdir='z', zorder=zorder)
        # temp[0]._color=color


def draw_BZ_edge(ax, rec_lat_mat, color=None, ls_list=None, diff=0):
    """ tool_tip_missing
    """
    color = color or 'k'

    A, b = get_dd_matrix(rec_lat_mat)
    p = Hrep(A, b)
    verts = p.generators
    edges = get_edge(p.adj)
    # print 'number of vertices:', len(verts)
    draw_edges(ax, verts, edges, color, ls_list, diff=diff)
    # return ax


def draw_arrow(ax, st_point, end_point, color=None, label=None):
    # length = 1.
    color = color or 'k'

    x_list = [st_point[0], end_point[0]]
    y_list = [st_point[1], end_point[1]]
    z_list = [st_point[2], end_point[2]]
    arrow = Arrow3D(x_list, y_list, z_list,\
                    mutation_scale=20, lw=2, arrowstyle="-|>", color=color, zorder=0)
    ax.add_artist(arrow)
    if label is not None:
        zdir = np.array(end_point) - np.array(st_point)
        ax.text(end_point[0], end_point[1], end_point[2], label, zdir=zdir)


def draw_BZ_points(ax, points, c, color=None, norm_factor=None):
    norm_factor = norm_factor or 1.
    points  = np.array(points)
    # print points
    xs = points[:,0]
    ys = points[:,1]
    zs = points[:,2]

    if c is None or len(c) != len(xs):
        c = [1.] * len(xs)
    s = np.array(c) / float(sum(c)) * 500 * norm_factor
    print max(c), sum(c), max(c)/sum(c)
    alpha = s / s.max()
    # ax.scatter(xs, ys, zs, s=s, c=c, lw=1)
    # ax.scatter(xs, ys, zs, s=s, c=color, lw=1)
    mlab.points3d(xs, ys, zs, color=color)


def main(ax, rec_lat_mat):
    ls_list = [1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 
               0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1]
    ls_list = ['-' if item else '--' for item in ls_list]

    draw_BZ_edge(ax, np.array(rec_lat_mat*100, dtype=int)/100., ls_list=ls_list)
    # draw_arrow(ax, [0, 0, 0], rec_lat_mat[0], label='1')
    # draw_arrow(ax, [0, 0, 0], rec_lat_mat[1], label='2')
    # draw_arrow(ax, [0, 0, 0], rec_lat_mat[2], label='3')

if __name__ == '__main__':
    # fig = plt.figure()
    black = (0,0,0)
    white = (1,1,1)
    mlab.figure(bgcolor=white)
    mlab.clf()
    # mlab.points3d(0, 0, 0, color=black,)
    N = 100
    n_mer, n_long = 6, 11
    pi = np.pi
    dphi = pi/1000.0
    phi = np.arange(0.0, 2*pi + 0.5*dphi, dphi)
    mu = phi*n_mer
    x = np.cos(mu)*(1+np.cos(n_long*mu/n_mer)*0.5)
    y = np.sin(mu)*(1+np.cos(n_long*mu/n_mer)*0.5)
    z = np.sin(n_long*mu/n_mer)*0.5

    mlab.plot3d(x, y, z, color=black)

    # lat = np.array([[1,1,0], [1,0,1], [0,1,1]]) / 2.
    # rec_lat_mat = np.linalg.inv(lat).T

    # main(None, rec_lat_mat)
    # draw_BZ_points(None, [[0, 0, 0]], [1], color=black)
    # mlab.view(azimuth=0, elevation=45, distance=10.)
    # mlab.view()
    mlab.savefig('test.eps')

    mlab.show()



    # ax = fig.gca(projection='3d')
    # ax.grid(False)
    # ax.set_axis_off()
    # ax.set_aspect('equal')

    # lat = np.array([[1,1,0], [1,0,1], [0,1,1]]) / 2.
    # rec_lat = np.linalg.inv(lat).T
    # print 'rec_lat'
    # print rec_lat
    # main(ax, rec_lat)

    # lat = np.array([[1,0,-1], [1,-2,1], [1,1,1]])
    # # lat = np.array([[1,0,0], [0,1,0], [0,0,1]])
    # print 'rec_lat'
    # rec_lat = np.array([[ -0.26, 0.16, 0.001],
    #                     [  0.26, 0.16, 0.001],
    #                     [  0,    0,    0.156]])
    # rec_lat = np.linalg.inv(lat).T
    # print rec_lat
    # main(ax, rec_lat)
    # plt.show()
