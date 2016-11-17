import warnings as _warn

import mdp
from .mca_nodes_online import MCANode
from .pca_nodes_online import CCIPCAWhiteningNode as WhiteningNode
from .stats_nodes_online import MovingAvgNode, MovingTimeDiffNode
from mdp.utils import mult


class IncSFANode(mdp.OnlineNode):
    """

    Incremental Slow Feature Analysis (IncSFA) extracts the slowly varying
    components from the input data incrementally. More information about IncSFA
    can be found in Kompella V.R, Luciw M. and Schmidhuber J., Incremental Slow
    Feature Analysis: Adaptive Low-Complexity Slow Feature Updating from
    High-Dimensional Input Streams, Neural Computation, 2012.


    **Instance variables of interest (stored in cache)**

      ``slow_features`` (can also be accessed as self.sf)
         Slow feature vectors

      ``whitening_vectors``
         Whitening vectors

      ``weight_change``
         Difference in slow features after update

    """
    def __init__(self, input_dim=None, output_dim=None, dtype=None, numx_rng=None, eps=0.05,
                 whitening_output_dim=None, remove_mean=True, avg_n=None, amn_params=(20,200,2000,3),
                 init_pca_vectors=None, init_mca_vectors=None):
        """
        eps: Learning rate (default: 0.1)

        whitening_output_dim: Whitening output dimension. (default: input_dim)

        remove_mean: Remove input mean incrementally (default: True)

        avg_n - When set, the node updates an exponential weighted moving average.
                avg_n intuitively denotes a window size. For a large avg_n, avg_n samples
                represents about 86% of the total weight. (Default:None)

        amn_params: pca amnesic parameters. Default set to (n1=20,n2=200,m=2000,c=3).
                            For n < n1, ~ moving average.
                            For n1 < n < n2 - Transitions from moving average to amnesia. m denotes the scaling param and
                            c typically should be between (2-4). Higher values will weigh recent data.

        init_pca_vectors: initial whitening vectors. Default - randomly set

        init_mca_vectors: initial mca vectors. Default - randomly set

        """

        super(IncSFANode, self).__init__(input_dim, output_dim, dtype, numx_rng)
        self.eps = eps
        self.whitening_output_dim = whitening_output_dim

        self.whiteningnode = WhiteningNode(input_dim=input_dim, output_dim=self.whitening_output_dim,
                                           dtype=dtype, numx_rng=numx_rng, init_eigen_vectors=init_pca_vectors,
                                           amn_params=amn_params)
        self.tdiffnode = MovingTimeDiffNode(numx_rng=numx_rng)
        self.mcanode = MCANode(input_dim=self.whitening_output_dim, output_dim=output_dim,
                               dtype=dtype, numx_rng=numx_rng, init_eigen_vectors=init_mca_vectors, eps=eps)

        self.remove_mean = remove_mean
        self.avg_n = avg_n
        if remove_mean:
            self.avgnode = MovingAvgNode(numx_rng=numx_rng, avg_n=avg_n)

        self._new_episode = True

        self._init_sf = None
        self.wv = None
        self.sf = None

        # cache to store variables
        self._cache = {'slow_features': None, 'whitening_vectors': None, 'weight_change': None}

    @property
    def new_episode(self):
        return self._new_episode

    @new_episode.setter
    def new_episode(self, flag):
        self._new_episode = flag

    @property
    def init_slow_features(self):
        return self._init_sf

    @property
    def init_pca_vectors(self):
        return self.whiteningnode.init_eigen_vectors

    @property
    def init_mca_vectors(self):
        return self.mcanode.init_eigen_vectors

    def set_training_type(self, training_type):
        if training_type != 'incremental':
            _warn.warn("Cannot set training type to %s. Only 'incremental' is supported"%(training_type))

    def _check_params(self, x):
        if self._init_sf is None:
            if self.remove_mean:
                self._pseudo_check_fn(self.avgnode, x)
            x = self.avgnode.execute(x)
            self._pseudo_check_fn(self.whiteningnode, x)
            x = self.whiteningnode.execute(x)
            self._pseudo_check_fn(self.tdiffnode, x)
            self._pseudo_check_fn(self.mcanode, x)
            self._init_sf = mult(self.whiteningnode.init_eigen_vectors,self.mcanode.init_eigen_vectors)
            self.sf = self._init_sf

    def _pseudo_check_fn(self, node, x):
        node._check_input(x)
        node._check_params(x)

    def _pseudo_train_fn(self, node, x):
        node._train(x)
        node._train_iteration+=x.shape[0]

    def _train(self, x):
        if self.remove_mean:
            self._pseudo_train_fn(self.avgnode, x)
            x = self.avgnode._execute(x)

        self._pseudo_train_fn(self.whiteningnode, x)
        x = self.whiteningnode._execute(x)

        self._pseudo_train_fn(self.tdiffnode, x)

        if self.new_episode:
            self.new_episode = False
            return

        x = self.tdiffnode._execute(x)

        self._pseudo_train_fn(self.mcanode, x)

        sf = mult(self.whiteningnode.v,self.mcanode.v)
        sf_change = mdp.numx_linalg.norm(sf - self.sf)
        self.sf = sf

        self.cache['slow_features'] = self.sf.copy()
        self.cache['whitening_vectors'] = self.whiteningnode.cache['eigen_vectors']
        self.cache['weight_change'] = sf_change

    def _execute(self, x):
        if self.remove_mean:
            x = self.avgnode._execute(x)
        return mult(x, self.sf)


    def __repr__(self):
        # print all args
        name = type(self).__name__
        inp = "input_dim=%s" % str(self.input_dim)
        out = "output_dim=%s" % str(self.output_dim)
        if self.dtype is None:
            typ = 'dtype=None'
        else:
            typ = "dtype='%s'" % self.dtype.name
        numx_rng = "numx_rng=%s" % str(self.numx_rng)
        eps = "\neps=%s"% str(self.eps)
        whit_dim = "whitening_output_dim=%s"%str(self.whitening_output_dim)
        remove_mean = "remove_mean=%s"%str(self.remove_mean)
        avg_n = "avg_n=%s"%(self.avg_n)
        amn = "\namn_params=%s" % str(self.whiteningnode.amn_params)
        init_pca_vecs = "init_pca_vectors=%s" % str(self.init_pca_vectors)
        init_mca_vecs = "init_pca_vectors=%s" % str(self.init_mca_vectors)
        args = ', '.join((inp, out, typ, numx_rng, eps, whit_dim, remove_mean, avg_n, amn, init_pca_vecs, init_mca_vecs))
        return name + '(' + args + ')'

