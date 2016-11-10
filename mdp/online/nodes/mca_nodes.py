
import mdp
from mdp.online import INode
from mdp.utils import mult
from past.utils import old_div

class MCANode(INode):
    """
    Minor Component Analysis (MCA) extracts minor components from the
    input data incrementally. More information about MCA can be found in
    Peng, D. and Yi, Z, A new algorithm for sequential minor component analysis,
    International Journal of Computational Intelligence Research,
    2(2):207--215, 2006.


    **Instance variables of interest (stored in cache)**

      ``eigen_vectors`` (can also be accessed as self.v)
         Eigen vectors

      ``eigen_values`` (can also be accessed as self.d)
         Eigen values


    """

    def __init__(self, input_dim=None, output_dim=None, dtype=None, numx_rng=None, eps=0.1, gamma=1.0,
                 normalize=True, init_eigen_vectors=None):
        """
        eps: Learning rate (default: 0.1)

        gamma: Sequential addition coefficient (default: 1.0)

        normalize: If True, eigenvectors are normalized after every update.
                      Useful for non-stationary input data.  (default: True)

        init_eigen_vectors: initial eigen vectors. Default - randomly set
        """
        super(MCANode, self).__init__(input_dim, output_dim, dtype, numx_rng)
        self.eps = eps
        self.gamma = gamma
        self.normalize = normalize

        self._init_v = init_eigen_vectors

        self.v = None  # Eigenvectors
        self.d = None  # Eigenvalues

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
            self.v = self._init_v.copy()
            self.d = mdp.numx_linalg.norm(self.v, axis=0)

    def _check_params(self, *args):
        if self._init_v is None:
            self.init_eigen_vectors = 0.1 * self.numx_rng.randn(self.input_dim, self.output_dim)

    def _train(self, x):
        C = mult(x.T, x)
        for j in xrange(self.output_dim):
            v = self.v[:,j:j + 1]
            d = self.d[j]

            n = self.eps / (1 + j * 1.2)
            a = mult(C, v)
            if self.normalize:
                v = (1.5 - n) * v - n * a
            else:
                v = (1.5 - n * (d ** 2)) * v - n * a
            l = mult(v.T, v)
            C = C + self.gamma * mult(v, v.T) / l

            self.v[:,j:j+1] = v
            self.d[j] = mdp.numx.sqrt(l)
            if self.normalize:
                self.v[:,j:j+1] = old_div(v,self.d[j])

        self.cache['eigen_vectors'] = self.v.copy()
        self.cache['eigen_values'] = self.d.copy()

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

