__docformat__ = "restructuredtext en"

import mdp
from mdp import NodeException, IsNotTrainableException
from mdp import TrainingException, TrainingFinishedException,IsNotInvertibleException
from mdp import Node

class INode(Node):

    """An incremental Node (INode) is the basic building block of
        an online (incremental) MDP application.

        It represents a data processing element, like for example a learning
        algorithm, a data filter, or a visualization step.
        Each node has a continuous training phase, during which the
        internal structures are learned from training data (e.g. the weights
        of a neural network are adapted) and an execution phase, where new
        data can be processed forwards (by processing the data through the node)
        or backwards (by applying the inverse of the transformation computed by
        the node if defined).

        INodes have been designed to be updated incrementally or block-incrementally
        by a continuous stream of input data. It is thus possible to perform
        computations on amounts of data that would not fit into memory or
        to generate data on-the-fly.

        An `INode` also defines some utility methods, like for example
        `copy` and `save`, that return an exact copy of a node and save it
        in a file, respectively. Additional methods may be present, depending
        on the algorithm.

        INodes also supports using a pre-seeded random number generator (through
        'numx_rng' parameter. This can be used to replicate results.

        `INode` subclasses should take care of overwriting (if necessary)
        the functions `_train`, `_stop_training`, `_execute`,
        `is_invertible`, `_inverse`, '_init_params', and `_get_supported_dtypes`.
        If you need to overwrite the getters and setters of the
        node's properties refer to the docstring of `get_input_dim`/`set_input_dim`,
        `get_output_dim`/`set_output_dim`, `get_dtype`/`set_dtype`, 'get_numx_rng'/'set_numx_rng'.
    """

    def __init__(self, input_dim=None, output_dim=None, dtype=None, numx_rng=None):
        """If the input dimension and the output dimension are
        unspecified, they will be set when the `train` or `execute`
        method is called for the first time.
        If dtype is unspecified, it will be inherited from the data
        it receives at the first call of `train` or `execute`.
        If numx_rng is unspecified, it will be set a random number generator
        with a random seed.

        Every subclass must take care of up- or down-casting the internal
        structures to match this argument (use `_refcast` private
        method when possible).
        """
        super(INode, self).__init__(input_dim,output_dim,dtype)
        # this var stores the index of the current training iteration
        self._train_iteration = 0
        # this cache var dict stores an interemediate result or paramenter values
        # at the end of each train iteration. INode subclasses should
        # initialize the required keys
        self._cache = dict()
        # this var stores random number generator
        self._numx_rng = None
        self.set_numx_rng(numx_rng)

    def get_numx_rng(self):
        """Return input dimensions."""
        return self._numx_rng

    def set_numx_rng(self, rng):
        """Set numx random number generator.
            Perform type checks
        """
        if rng is None:
            pass
        elif not isinstance(rng, mdp.numx_rand.mtrand.RandomState):
                raise NodeException('numx_rng should be of type %s but given %s'
                                    %(str(mdp.numx_rand.mtrand.RandomState), str(type(rng))))
        else:
            self._set_numx_rng(rng)

    def _set_numx_rng(self, rng):
        self._numx_rng = rng

    numx_rng = property(get_numx_rng,
                         set_numx_rng,
                         doc="Numpy seeded random number generator")

    def get_cache(self):
        return self._cache

    def set_cache(self, c):
        self._cache = c

    cache = property(get_cache, set_cache, doc="Internal cache dict")

    def get_current_train_iteration(self):
        """Return the index of the current training iteration."""
        return self._train_iteration

    def _pre_execution_checks(self, x):
        """This method contains all pre-execution checks.
        It can be used when a subclass defines multiple execution methods.
        """
        # control the dimension x
        self._check_input(x)

        # set the output dimension if necessary
        if self.output_dim is None:
            self.output_dim = self.input_dim

    def _pre_inversion_checks(self, y):
        """This method contains all pre-inversion checks.

        It can be used when a subclass defines multiple inversion methods.
        """
        if not self.is_invertible():
            raise IsNotInvertibleException("This node is not invertible.")

        # set the output dimension if necessary
        if self.output_dim is None:
            # if the input_dim is not defined, raise an exception
            if self.input_dim is None:
                errstr = ("Number of input dimensions undefined. Inversion"
                          "not possible.")
                raise NodeException(errstr)
            self.output_dim = self.input_dim

        # control the dimension of y
        self._check_output(y)

    def _check_params(self, x):
        # set in the subclass
        pass

    def _check_input(self, x):
        super(INode, self)._check_input(x)

        # set numx_rng if necessary
        if self.numx_rng is None:
            self.numx_rng = mdp.numx_rand.RandomState()



    def train(self, x, *args, **kwargs):
        """Update the internal structures according to the input data `x`.

        `x` is a matrix having different variables on different columns
        and observations on the rows.

        By default, subclasses should overwrite `_train` to implement their
        training phase. The docstring of the `_train` method overwrites this
        docstring.

        Note: a subclass supporting multiple training phases should implement
        the *same* signature for all the training phases and document the
        meaning of the arguments in the `_train` method doc-string. Having
        consistent signatures is a requirement to use the node in a flow.
        """

        if not self.is_trainable():
            raise IsNotTrainableException("This node is not trainable.")

        if not self.is_training():
            err_str = "The training phase has already finished."
            raise TrainingFinishedException(err_str)

        self._check_input(x)
        self._check_params(x)
        self._check_train_args(x, *args, **kwargs)

        self._train_phase_started = True

        self._train_phase_started = True
        for i in xrange(x.shape[0]):
            for _phase in xrange(len(self._train_seq)):
                self._train_seq[_phase][0](self._refcast(x[i:i+1]), *args, **kwargs)
            self._train_iteration += 1

    def stop_training(self, *args, **kwargs):
        """Stop the training phase.

        By default, subclasses should overwrite `_stop_training` to implement
        this functionality. The docstring of the `_stop_training` method
        overwrites this docstring.
        """
        if self.is_training() and self._train_phase_started == False:
            raise TrainingException("The node has not been trained.")

        if not self.is_training():
            err_str = "The training phase has already finished."
            raise TrainingFinishedException(err_str)

        # close the current phase.
        for _phase in xrange(len(self._train_seq)):
            self._train_seq[_phase][1](*args, **kwargs)
        self._train_phase = len(self._train_seq)
        self._train_phase_started = False
        # check if we have some training phase left
        if self.get_remaining_train_phase() == 0:
            self._training = False

    def __add__(self, other):
        # check other is a node
        if isinstance(other, INode):
            return mdp.Flow([self, other])
        elif isinstance(other, mdp.Flow):
            flow_copy = other.copy()
            flow_copy.insert(0, self)
            return flow_copy.copy()
        else:
            err_str = ('can only concatenate node'
                       ' (not \'%s\') to node' % (type(other).__name__))
            raise TypeError(err_str)


