
import mdp
from mdp.online import INode
from mdp.utils import mult

class CCIPCANode(INode):
    """

    Candid-Covariance free Incremental Principal Component Analysis (CCIPCA)
    extracts the principal components from the input data incrementally.
    More information about Candid-Covariance free Incremental Principal
    Component Analysis can be found in Weng J., Zhang Y. and Hwang W.,
    Candid covariance-free incremental principal component analysis,
    IEEE Trans. Pattern Analysis and Machine Intelligence,
    vol. 25, 1034--1040, 2003.

    **Inputs**

      ``input_dim``
          Input Dimension

      ``output_dim``
          Output Dimension


    **kwargs**

      ``reduce`` (default: False)
          Automatically reduce dimensionality. 

      ``var_rel`` (default: 0.001)
          Relative variance threshold to reduce dimensionality

      ``beta`` (default: 1.1)
          Variance ratio threshold to reduce dimensionality

      ``amn_params`` (default: [20,200,2000,3])
          Amnesic Parameters

      ``init_eigen_vectors`` (default: randomly generated)
          initial eigen vectors

    **Instance variables of interest (stored in cache)**

      ``eigen_vectors``
         Normalized eigen vectors

      ``eigen_values``
         Corresponding eigen values

    """

    def __init__(self, input_dim=None, output_dim=None, dtype=None, numx_rng=None, **kwargs):
        super(CCIPCANode, self).__init__(input_dim, output_dim, dtype, numx_rng)
        self.kwargs = kwargs

        self.reduce = self.kwargs.get('reduce', False)
        self.var_rel = self.kwargs.get('var_rel', 0.001)
        self.beta = self.kwargs.get('beta', 1.1)
        self.amn_params = self.kwargs.get('amn_params', [20., 200., 2000., 3.])
        self._init_v = self.kwargs.get('init_eigen_vectors', None)

        self._v = None  # Internal Eigen Vector (unNormalized and transposed)
        self.explained_variance_tot = None # Total Explained Variance
        self.v = None  # Eigen Vector (Normalized) (reduced if reduce is True)
        self.d = None  # Eigen Value (reduced if reduce is True)
        self.reduced_dim = self.output_dim

        self._cache = {'eigen_vectors': None, 'eigen_values': None}

    @property
    def init_eigen_vectors(self):
        return self._init_v

    @init_eigen_vectors.setter
    def init_eigen_vectors(self, init_eigen_vectors=None):
        self._init_v = init_eigen_vectors

    def _check_input(self, x):
        super(CCIPCANode, self)._check_input(x)

        if self.output_dim is None:
            self.output_dim = self.input_dim
            self.reduced_dim = self.output_dim

    def _check_params(self, x):
        if self._init_v is None:
            self._init_v = 0.1 * self.numx_rng.randn(self.output_dim, self.input_dim)

        if self._v is None:
            self._v = self._init_v.copy()
            self._d = mdp.numx.sum(mdp.numx.absolute(self._v) ** 2, axis=-1) ** (1. / 2)
            self._vn = self._v / self._d.reshape(self._v.shape[0], 1)

    def _amnesic(self, n):
        _i = float(n + 1)
        n1, n2, m, C = self.amn_params
        if _i < n1:
            l = 0
        elif (_i >= n1) and (_i < n2):
            l = C * (_i - n1) / (n2 - n1)
        else:
            l = C + (_i - n2) / m
        _wold = float(_i - 1 - l) / _i
        _wnew = float(1 + l) / _i
        return [_wold, _wnew]

    def _train(self, x):
        [w1, w2] = self._amnesic(self.get_current_train_iteration()+1)
        red_j = self.output_dim
        red_j_Flag = False
        explained_var = 0.0

        r = x.copy()
        for j in xrange(self.output_dim):
            v = self._v[j:j + 1].copy()
            v = w1 * v + w2 * mult(r, v.T) / self._d[j] * r
            self._d[j] = mdp.numx.linalg.norm(v)
            vn = v / self._d[j]
            r = r - mult(r, vn.T) * vn
            explained_var += self._d[j]
            if (self.reduce is True) and (red_j_Flag is False):
                ratio1 = self._d[j] / self._d[0]
                ratio2 = explained_var / self.explained_variance_tot
                # print j, " :  ", ratio1, " :  ", ratio2, " :  ",self._d[j]
                if (ratio1 < self.var_rel or ratio2 > self.beta):
                    red_j = j
                    red_j_Flag = True
                    # print j,  " :  ", ratio1, " :  ", ratio2, " :  ", self._d[j]
            self._v[j] = v.copy()
            self._vn[j] = vn.copy()

        if explained_var > 0.0001:
            self.explained_variance_tot = explained_var
        self.v = self._vn[:red_j].T.copy()
        self.d = self._d[:red_j].copy()
        self.reduced_dim = red_j

        self.cache['eigen_vectors'] = self.v
        self.cache['eigen_values'] = self.d

    def get_explained_variance(self):
        """Return the fraction of the original variance that can be
        explained by self._output_dim PCA components.
        """
        return self.explained_variance_tot

    def get_projmatrix(self, transposed=1):
        """Return the projection matrix."""
        self._if_training_stop_training()
        if transposed:
            return self.v
        return self.v.T

    def get_recmatrix(self, transposed=1):
        """Return the back-projection matrix (i.e. the reconstruction matrix).
        """
        self._if_training_stop_training()
        if transposed:
            return self.v.T
        return self.v

    def _execute(self, x, n=None):
        """Project the input on the first 'n' principal components.
        If 'n' is not set, use all available components."""
        if n is not None:
            return mult(x, self.v[:, :n])
        return mult(x, self.v)

    def _inverse(self, y, n=None):
        """Project 'y' to the input space using the first 'n' components.
        If 'n' is not set, use all available components."""
        if n is None:
            n = y.shape[1]
        if n > self.output_dim:
            error_str = ("y has dimension %d,"
                         " should be at most %d" % (n, self.output_dim))
            raise mdp.NodeException(error_str)

        v = self.get_recmatrix()
        if n is not None:
            return mult(y, v[:n, :])
        return mult(y, v)


