import itertools
import mdp


# TODO: Could use or add to the graph package of MDP.

class GridProcessingNode(mdp.Node):
    """
    GridProcessingNode is a non-trainable node that
    translates an n-dimensional cartesian grid to a topological map
    of graph nodes labeled with n-dimensional coordinates and
    a 1D index. Therefore, the node provides conversions
    from grid coordinates to graph coordinates or graph indices.

    The type of conversion can be selected by setting the 'output_type'
    attribute. Accepted values:
    'gridx' - default (just returns the input)
    'graphx' - returns graph coordinates
    'graphindx' - returns graph indices of the input.

    The node is also invertible, enabling conversions
    back from 'graphx' and 'graphindx' to 'gridx'.

    Eg.

    3x3 cartesian grid points to  'graphx' output_type

    (0,0)   (2,0)   (4,0)       (0,0)   (1,0)   (2,0)

    (0,2)   (2,2)   (4,2)  <->  (0,1)   (1,1)   (2,1)

    (0,4)   (2,4)   (4,4)       (0,2)   (1,2)   (2,2)

    3x3 cartesian grid points to 'graphindx' output_type

    (0,0)   (2,0)   (4,0)       0   1   2

    (0,2)   (2,2)   (4,2)  <->  3   4   5

    (0,4)   (2,4)   (4,4)       6   7   8


    The node also provides a utility method:
     'get_neighbors' - Returns the neighboring
     grid points based on a distance function applied to
     the graph coordinates. The distance function
     can be specified using the 'nbr_dist_fn' argument
     and the threshold can be specified using the
     'nbr_radius' argument.

     'get_adjacency_matrix' - Returns an adjacency matrix
     of the graph.

    """

    def __init__(self, grid_lims, n_grid_pts=None, output_type=None, nbr_dist_fn=1, nbr_radius=1,
                 include_central_elem=True, input_dim=None, dtype=None):
        """
        grid_lims - a tuple of lower and upper bounds for each dimension of the input grid.
                Eg., [(lower1, lower2, ...,), (upper1, upper2, ...,)]

        n_grid_pts - number of grid points within each dimension. This provides the spacing between
                the grid points. Default is set to the number of integers between the bounds.

        output_type - the type of conversion. Accepted values:
                    'gridx' - default (just returns the input)
                    'graphx' - returns graph coordinates
                    'graphindx' - returns graph indices of the input.

        nbr_dist_fn - distance function to define the closest neighbors (applied on the graph
                    coordinates). This is the same as
                    "order" argument of the numx_linalg function. (Doc added here for quick reference)
                     Accepted values:
                    {non-zero int, inf, -inf, 'fro', 'nuc'}, inf means numpy's `inf` object.

                    For values of ``nbr_dist_fn <= 0``, the result is, strictly speaking, not a
                    mathematical 'norm', but it may still be useful for various numerical
                    purposes.

                    The following norms can be calculated:

                ==========  ============================  ==========================
                nbr_dist_fn    norm for matrices             norm for vectors
                ==========  ============================  ==========================
                    None   Frobenius norm                2-norm
                    'fro'  Frobenius norm                --
                    'nuc'  nuclear norm                  --
                    inf    max(sum(abs(x), axis=1))      max(abs(x))
                    -inf   min(sum(abs(x), axis=1))      min(abs(x))
                    0      --                            sum(x != 0)
                    1      max(sum(abs(x), axis=0))      as below
                    -1     min(sum(abs(x), axis=0))      as below
                    2      2-norm (largest sing. value)  as below
                    -2     smallest singular value       as below
                    other  --                            sum(abs(x)**ord)**(1./ord)
                    =====  ============================  ==========================

                For further info, refer to Numpy or Scipy linalg.norm documentation.

        nbr_radius - The radial distance to determine neighbors.

        include_central_elem - Includes distance 0 also as a neighour (that is, the grid point itself)

        """
        super(GridProcessingNode, self).__init__(input_dim, output_dim=None, dtype=dtype)
        if len(grid_lims) != 2:
            raise mdp.NodeException("'grid_lims' has %d elements given, required 2 "
                                    "[(lower1, lower2, ...,), (upper1, upper2, ...,)]" % (len(grid_lims)))
        if mdp.numx.isscalar(grid_lims[0]):
            grid_lims = [tuple((lim,)) for lim in grid_lims]
        if len(grid_lims[0]) != len(grid_lims[1]):
            raise mdp.NodeException("Length of lower_bounds ('grid_lims[0]=%d') does not match the length "
                                    "of the upper_bounds ('grid_lims[1]=%d)." % (len(grid_lims[0]), len(grid_lims[1])))
        self.grid_lims = mdp.numx.asarray(grid_lims)

        if n_grid_pts is None:
            n_grid_pts = [int(self.grid_lims[1][i] - self.grid_lims[0][i] + 1) for i in xrange(len(self.grid_lims[0]))]
        elif mdp.numx.isscalar(n_grid_pts):
            n_grid_pts = [n_grid_pts] * len(self.grid_lims[0])
        if len(n_grid_pts) != len(self.grid_lims[0]):
            raise mdp.NodeException("Length of 'n_grid_pts' (given = %d) does not match with the "
                                    "number of grid dimensions (%d)." % (len(n_grid_pts), len(self.grid_lims[0])))
        self.n_grid_pts = mdp.numx.asarray(n_grid_pts, dtype='int')

        self.nbr_dist_fn = nbr_dist_fn
        self.nbr_radius = nbr_radius
        self.include_central_elem = include_central_elem

        self._output_type = None
        self.output_type = output_type

        self._grid = [mdp.numx.linspace(self.grid_lims[0, i], self.grid_lims[1, i], self.n_grid_pts[i], endpoint=True)
                      for i in xrange(self.n_grid_pts.shape[0])]

        self._graph_dim = self.n_grid_pts.shape[0]
        self._tot_graph_nodes = mdp.numx.product(self.n_grid_pts)
        self._graph_lims = mdp.numx.array([mdp.numx.zeros(self._graph_dim), self.n_grid_pts - 1])
        self._graph_elem = mdp.numx.asarray(self._get_graph_elem(self._graph_dim,
                                                                 self.nbr_radius,
                                                                 self.nbr_dist_fn, self.include_central_elem))

    @staticmethod
    def _get_graph_elem(graph_dim, radius, nbr_dist_fn, inclued_central_elem):
        _a = list(itertools.product(range(-radius, radius + 1), repeat=graph_dim))
        if inclued_central_elem:
            a = filter(lambda x: (mdp.numx_linalg.norm(x, nbr_dist_fn) <= radius), _a)
        else:
            a = filter(lambda x: (mdp.numx_linalg.norm(x, nbr_dist_fn) <= radius)
                                 and not (mdp.numx.array(x) == 0).all(), _a)
        return a

    def _get_graph_neighbors(self, graphx):
        nbrs = filter(lambda x: (self._graph_lims[0, :] <= x).all() and (x <= self._graph_lims[1, :]).all(),
                      graphx + self._graph_elem)
        return mdp.numx.atleast_2d(nbrs)

    # properties

    @property
    def output_type(self):
        return self._output_type

    @output_type.setter
    def output_type(self, typ):
        if typ is None:
            typ = 'gridx'
        elif typ not in ['gridx', 'graphx', 'graphindx']:
            raise mdp.NodeException("'output_type' must be either of 'gridx', 'graphx' or "
                                    "'graphindx', given %s." % str(typ))
        self._output_type = typ
        if self._output_type == 'gridx':
            self._output_dim = self._input_dim
        elif self._output_type == 'graphx':
            self._output_dim = self._input_dim
        else:
            self._output_dim = 1

    # conversion methods

    def _gridx_to_graphx(self, gridx):
        graphx = mdp.numx.zeros(gridx.shape)
        for i in xrange(self._graph_dim):
            graphx[:, i] = mdp.numx.argmin(mdp.numx.absolute(self._grid[i] - gridx[:, i:i + 1]), axis=1)
        return graphx

    def _graphx_to_gridx(self, graphx):
        gridx = mdp.numx.zeros(graphx.shape)
        for i in xrange(self._graph_dim):
            gridx[:, i] = self._grid[i][graphx[:, i].astype('int')]
        return gridx

    def _graphx_to_graphindx(self, graphx):
        graphindx = mdp.numx.zeros([graphx.shape[0], 1])
        for i in xrange(self._graph_dim):
            graphindx = graphindx * self.n_grid_pts[i] + graphx[:, i:i + 1]
        return graphindx

    def _graphindx_to_graphx(self, graphindx):
        _graphindx = graphindx.astype('int')
        graphx = mdp.numx.zeros((graphindx.shape[0], self._graph_dim))
        for i in xrange(self._graph_dim):
            _d = int(mdp.numx.product(self.n_grid_pts[i + 1:]))
            graphx[:, i:i + 1] = _graphindx / _d
            _graphindx %= _d
        return graphx

    def _gridx_to_graphindx(self, gridx):
        graphindx = mdp.numx.zeros([gridx.shape[0], 1])
        for i in xrange(self._graph_dim):
            graphindx = graphindx * self.n_grid_pts[i] + \
                        mdp.numx.argmin(mdp.numx.absolute(self._grid[i] - gridx[:, i:i + 1]), axis=1)[:, None]
        return graphindx

    def _graphindx_to_gridx(self, graphindx):
        _graphindx = graphindx.astype('int')
        gridx = mdp.numx.zeros((graphindx.shape[0], self._graph_dim))
        for i in xrange(self._graph_dim):
            _d = int(mdp.numx.product(self.n_grid_pts[i + 1:]))
            graphx_i = _graphindx / _d
            _graphindx %= _d
            gridx[:, i:i + 1] = self._grid[i][graphx_i]
        return gridx

    @staticmethod
    def is_trainable():
        return False

    def _train(self, x):
        pass

    def _execute(self, x):
        y = x
        if self.output_type == 'graphx':
            y = self._gridx_to_graphx(x)
        elif self.output_type == 'graphindx':
            y = self._gridx_to_graphindx(x)
        return self._refcast(y)

    def _inverse(self, y):
        x = y
        if self.output_type == 'graphx':
            x = self._graphx_to_gridx(y)
        elif self.output_type == 'graphindx':
            x = self._graphindx_to_gridx(y)
        return self._refcast(x)

    # utility methods
    def get_neighbors(self, gridx):
        return self._graphx_to_gridx(self._get_graph_neighbors(self._gridx_to_graphx(gridx)))

    def get_adjacency_matrix(self):
        adj = mdp.numx.zeros([self._tot_graph_nodes, self._tot_graph_nodes])
        for i in xrange(adj.shape[0]):
            i_arr = mdp.numx.array([[i]]).astype('float')
            for j in self._graphx_to_graphindx(self._get_graph_neighbors(self._graphindx_to_graphx(i_arr))):
                adj[i, int(j)] = 1
        return self._refcast(adj)
