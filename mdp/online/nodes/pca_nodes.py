
import mdp
from mdp.online import INode
from mdp.utils import mult
from past.utils import old_div

class CCIPCANode(INode):
    """
    Candid-Covariance free Incremental Principal Component Analysis (CCIPCA)
    extracts the principal components from the input data incrementally.
    More information about Candid-Covariance free Incremental Principal
    Component Analysis can be found in Weng J., Zhang Y. and Hwang W.,
    Candid covariance-free incremental principal component analysis,
    IEEE Trans. Pattern Analysis and Machine Intelligence,
    vol. 25, 1034--1040, 2003.

    **Instance variables of interest (stored in cache)**

      ``eigen_vectors`` (can also be accessed as self.v)
         Eigen vectors

      ``eigen_values`` (can also be accessed as self.d)
         Eigen values

    """

    def __init__(self, input_dim=None, output_dim=None, dtype=None, numx_rng=None, 
                 amn_params=(20,200,2000,3), init_eigen_vectors=None, var_rel=1):
        """
        amn_params: Amnesic parameters. Default set to (n1=20,n2=200,m=2000,c=3).
                            For n < n1, ~ moving average.
                            For n1 < n < n2 - Transitions from moving average to amnesia. m denotes the scaling param and
                            c typically should be between (2-4). Higher values will weigh recent data.

        init_eigen_vectors: initial eigen vectors. Default - randomly set

        var_rel: Ratio cutoff to get reduced dimensionality. (Explained variance of reduced dimensionality <= beta * Total variance). Default = 1
        """

        super(CCIPCANode, self).__init__(input_dim, output_dim, dtype, numx_rng)

        self.amn_params = amn_params
        self._init_v = init_eigen_vectors
        self.var_rel = var_rel

        self._v = None  #Internal eigenvectors (unnormalized and transposed)
        self.v = None  #Eigenvectors (Normalized)
        self.d = None  #Eigenvalues

        self._var_tot = 1.0
        self._reduced_dims = self.output_dim
        self._cache = {'eigen_vectors': None, 'eigen_values': None}

    @property
    def init_eigen_vectors(self):
        return self._init_v

    @init_eigen_vectors.setter
    def init_eigen_vectors(self, init_eigen_vectors=None):
        self._init_v = init_eigen_vectors
        if self._input_dim is None:
            self._input_dim = self._init_v.shape[0]
        else:
            assert(self.input_dim == self._init_v.shape[0]), mdp.NodeException('Dimension mismatch. init_eigen_vectors shape[0] must be'
                                                                               '%d, given %d'%(self.input_dim, self._init_v.shape[0]))
        if self._output_dim is None:
            self._output_dim = self._init_v.shape[1]
        else:
            assert(self.output_dim == self._init_v.shape[1]), mdp.NodeException('Dimension mismatch. init_eigen_vectors shape[1] must be'
                                                                               '%d, given %d'%(self.output_dim, self._init_v.shape[1]))
        if self.v is None:
            self._v = self._init_v.copy()
            self.d = mdp.numx_linalg.norm(self._v, axis=0)
            self.v = old_div(self._v,self.d)

    def _check_params(self, *args):
        if self._init_v is None:
            self.init_eigen_vectors = 0.1 * self.numx_rng.randn(self.input_dim, self.output_dim)

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

        r = x
        for j in xrange(self.output_dim):
            v = self._v[:,j:j + 1]
            d = self.d[j]

            v = w1 * v + w2 * mult(r, v) / d * r.T
            d = mdp.numx_linalg.norm(v)
            vn = old_div(v,d)
            r = r - mult(r, vn) * vn.T
            explained_var += d

            if not red_j_Flag:
                ratio = explained_var / self._var_tot
                if (ratio > self.var_rel):
                    red_j = j
                    red_j_Flag = True

            self._v[:,j:j+1] = v
            self.v[:,j:j+1] = vn
            self.d[j] = d

        self._var_tot = explained_var
        self._reduced_dims = red_j
        
        self.cache['eigen_vectors'] = self.v.copy()
        self.cache['eigen_values'] = self.d.copy()

    def get_var_tot(self):
        """Return the  variance that can be
        explained by self._output_dim PCA components.
        """
        return self._var_tot

    def get_reduced_dimsensionality(self):
        """Return reducible dimensionality based on the set thresholds"""
        return self._reduced_dims

    def get_projmatrix(self, transposed=1):
        """Return the projection matrix."""
        if transposed:
            return self.v
        return self.v.T

    def get_recmatrix(self, transposed=1):
        """Return the back-projection matrix (i.e. the reconstruction matrix).
        """
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



class WhiteningNode(CCIPCANode):
    """

    Incrementally updates whitening vectors for the input data using CCIPCA.

    """
    __doc__ = __doc__ + "\t" +  "#"*30 +  CCIPCANode.__doc__

    def __init__(self, input_dim=None, output_dim=None, dtype=None, numx_rng=None,
                 amn_params=(20,200,2000,3), init_eigen_vectors=None, var_rel=1):
        super(WhiteningNode, self).__init__(input_dim, output_dim, dtype, numx_rng,
                 amn_params, init_eigen_vectors, var_rel)

    def _train(self, x):
        super(WhiteningNode,self)._train(x)
        self.v = old_div(self.v, mdp.numx.sqrt(self.d))
        self.cache['eigen_vectors'] = self.v
