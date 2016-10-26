


from mdp.online import INode

class SignalAvgNode(INode):
    """Compute moving average on the input data.
     Also supports exponentially weighted moving average when
     the parameter avg_n is set.

    This is an online learnable node (INode)

    **Internal variables of interest**

      ``self.avg``
          The current average of the input data
    """
    def __init__(self, input_dim=None, output_dim=None, dtype=None, **kwargs):
        """
        :Additional Arguments:

          avg_n - (Default:None).
                When set, the node updates an exponential weighted moving average.
                avg_n intuitively denotes a window size. For a large avg_n, avg_n samples
                represents about 86% of the total weight.
        """

        super(SignalAvgNode, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype)
        self.kwargs = kwargs
        self.avg_n = kwargs.get('avg_n')
        self.avg = 0

    def _train(self, x):
        if (self.avg_n is None):
            alpha = 1.0/(self.get_current_train_iteration()+1)
        else:
            alpha = 2.0/(self.avg_n+1)
        self.avg = (1-alpha) * self.avg + alpha*x

    def _execute(self, x):
        if self.get_current_train_iteration() == 1:
            return x
        else:
            return (x - self.avg)

    def _inverse(self, x):
        if self.get_current_train_iteration() == 1:
            return x
        else:
            return (x + self.avg)

    def get_current_average(self):
        return self.avg

