

import mdp
from mdp.online import PreserveDimINode

class SignalAvgNode(PreserveDimINode):
    """Compute moving average on the input data.
     Also supports exponentially weighted moving average when
     the parameter avg_n is set.

    This is an online learnable node (INode)

    **Internal variables of interest (stored in cache)**

      ``self.avg``
          The current average of the input data
    """
    def __init__(self, input_dim=None, dtype=None, numx_rng=None, avg_n=None):
        """
        avg_n - (Default:None).
                When set, the node updates an exponential weighted moving average.
                avg_n intuitively denotes a window size. For a large avg_n, avg_n samples
                represents about 86% of the total weight.
        """

        super(SignalAvgNode, self).__init__(input_dim=input_dim, output_dim=None, dtype=dtype, numx_rng=numx_rng)
        self.avg_n = avg_n
        self.avg = None
        self._cache = {'avg': None}

    def _check_params(self, x):
        if self.avg is None:
            self.avg = mdp.numx.zeros(x.shape[1])

    def _train(self, x):
        if (self.avg_n is None):
            alpha = 1.0/(self.get_current_train_iteration()+1)
        else:
            alpha = 2.0/(self.avg_n+1)
        self.avg = (1-alpha) * self.avg + alpha*x
        self._cache['avg']=self.avg

    def _execute(self, x):
        if self.get_current_train_iteration() <= 1:
            return x
        else:
            return (x - self.avg)

    def _inverse(self, x):
        if self.get_current_train_iteration() <= 1:
            return x
        else:
            return (x + self.avg)

    def get_average(self):
        return self.avg


class MovingDiffNode(PreserveDimINode):
    def __init__(self, input_dim=None, output_dim=None, dtype=None, numx_rng=None):
        super(MovingDiffNode, self).__init__(input_dim, output_dim, dtype, numx_rng)
        self.x_prev = None
        self.x_cur = None

    def _check_params(self, x):
        if self.x_prev is None:
            self.x_prev = mdp.numx.zeros(x.shape)
            self.x_cur = mdp.numx.zeros(x.shape)

    def _train(self, x):
        self.x_prev = self.x_cur
        self.x_cur = x[-1:]

    def _execute(self, x):
        x = mdp.numx.vstack((self.x_prev,x))
        return x[1:] - x[:-1]

