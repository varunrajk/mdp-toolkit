
import mdp
from mdp.online import INode, IFlow
from .pca_nodes import MCANode, WhiteningNode
from mdp.utils import mult


class TimeDiffNode(mdp.PreserveDimNode):
    def __init__(self, input_dim=None, output_dim=None, dtype=None):
        super(TimeDiffNode, self).__init__(input_dim, output_dim, dtype)
        self.x_prev = None

    def is_trainable(self):
        return False

    def _pre_execution_checks(self, x):
        if self.x_prev is None:
            self.x_prev = mdp.numx.zeros([1,x.shape[1]])

    def _execute(self, x, include_last_sample=True):
        if include_last_sample:
            x = mdp.numx.vstack((self.x_prev,x))
        self.x_prev = x[-1]
        return x[1:] - x[:-1]


class IncSFANode(INode):
    """

    Incremental Slow Feature Analysis (IncSFA) extracts the slowly varying
    components from the input data incrementally. More information about IncSFA
    can be found in Kompella V.R, Luciw M. and Schmidhuber J., Incremental Slow
    Feature Analysis: Adaptive Low-Complexity Slow Feature Updating from
    High-Dimensional Input Streams, Neural Computation, 2012.

    **kwargs**

      ``eps``
          Learning rate

      ``whitening_output_dim`` (default: input_dim)
          dimensionality reduction for the ccipca step 

      ``amn_params`` (default: [20,200,2000,3])
          Amnesic Parameters (for ccipca)

      ``avg_n`` (default: None)
          Exponentially weighted moving average coefficient

      ``remove_mean`` (default: True)
          Subtract signal average.

      ``init_pca_vectors`` (default: randomly generated)
          initial eigen vectors

      ``init_mca_vectors`` (default: randomly generated)
          initial eigen vectors


    **Instance variables of interest (stored in cache)**

      ``slow_features`` (can also be accessed as self.v)
         Slow feature vectors

      ``whitening_vectors`` (can also be accessed as self.wv)
         Whitening vectors

      ``weight_change``
         Difference in slow features after update

    """
    def __init__(self, input_dim=None, output_dim=None, dtype=None, numx_rng=None, **kwargs):
        super(IncSFANode, self).__init__(input_dim, output_dim, dtype, numx_rng)
        self.kwargs = kwargs

        self.eps = self.kwargs.get('eps', 0.01)
        self.whitening_output_dim = self.kwargs.get('whitening_output_dim', None)

        self._init_wv = self.kwargs.get('init_pca_vectors', None)
        self.whiteningnode = WhiteningNode(input_dim=input_dim, output_dim=self.whitening_output_dim,
                                           dtype=dtype, numx_rng=numx_rng, init_eigen_vectors=self._init_wv, **kwargs)

        self.tdiffnode = TimeDiffNode()

        self._init_mv = self.kwargs.get('init_mca_vectors', None)
        self.mcanode = MCANode(input_dim=self.whitening_output_dim, output_dim=output_dim,
                               dtype=dtype, numx_rng=numx_rng, init_eigen_vectors=self._init_mv, **kwargs)
        self.mcanode.gamme = 1.2*(0.2/self.eps)

        self._new_episode = True
        # this flow is only trained when the new_episode flag is True.
        # once trained the flag is set to false.
        self._first_sample_train_flow = IFlow([self.whiteningnode, self.tdiffnode])
        self._train_flow = IFlow([self.whiteningnode, self.tdiffnode, self.mcanode])
        self._execute_flow = IFlow([self.whiteningnode, self.mcanode])

        if self.kwargs.get('remove_mean', True):
            self.avgnode = mdp.online.nodes.SignalAvgNode(avg_n=self.kwargs.get('avg_n', None))
            self._train_flow.insert(0,self.avgnode)
            self._execute_flow.insert(0,self.avgnode)

        self._init_v = None
        self.wv = None
        self.v = None

        # cache to store variables
        self._cache = {'slow_features': None, 'whitening_vectors': None, 'weight_change': None}

    @property
    def new_episode(self):
        return self._new_episode

    @new_episode.setter
    def new_episode(self, flag):
        self._new_episode = flag

    def _train(self, x):
        if self.new_episode:
            self._first_sample_train_flow.train(x)
            self.new_episode = False
        else:
            self._train_flow.train(x)

        self.wv = self.whiteningnode.v

        if self.v is None:
            v_old = mult(self.mcanode.init_eigen_vectors, self.whiteningnode.init_eigen_vectors)
        else:
            v_old = self.v.copy()
        self.v = mult(self.mcanode.v, self.wv)

        self.cache['slow_features'] = self.v
        self.cache['whitening_vectors'] = self.wv
        self.cache['weight_change'] = [mdp.numx_linalg.norm(self.v - v_old)]

    def _execute(self, x):
        return self._execute_flow(x)


