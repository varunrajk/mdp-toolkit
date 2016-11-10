
from builtins import str
import mdp
from mdp import numx
from mdp.linear_flows import  FlowException, FlowExceptionCR, _sys, _traceback
from mdp.online import INode

class IFlow(mdp.Flow):
    """A 'Flow' is a sequence of nodes that are trained and executed
    together to form a more complex algorithm.  Input data is sent to the
    first node and is successively processed by the subsequent nodes along
    the sequence.

    Using a flow as opposed to handling manually a set of nodes has a
    clear advantage: The general flow implementation automatizes the
    training (including supervised training and multiple training phases),
    execution, and inverse execution (if defined) of the whole sequence.

    Crash recovery is optionally available: in case of failure the current
    state of the flow is saved for later inspection. A subclass of the
    basic flow class ('CheckpointFlow') allows user-supplied checkpoint
    functions to be executed at the end of each phase, for example to save
    the internal structures of a node for later analysis.
    Flow objects are Python containers. Most of the builtin 'list'
    methods are available. A 'Flow' can be saved or copied using the
    corresponding 'save' and 'copy' methods.
    """

    def __init__(self, flow, crash_recovery=False, verbose=False):
        super(IFlow, self).__init__(flow, crash_recovery, verbose)
        self._check_iflow_compatibilitiy(flow)
        self._cache = self._get_cache_from_flow(flow)

    @property
    def cache(self):
        return self._cache

    def _train_node(self, data_iterable, nodenr):
        err_str = ('Not used in IFlow')
        FlowException(err_str)

    def _train_nodes(self, data_iterable):
        train_arg_keys_list = [self._get_required_train_args(node) for node in self.flow]
        train_args_needed_list = [bool(len(train_arg_keys)) for train_arg_keys in train_arg_keys_list]

        empty_iterator = True
        for x in data_iterable:
            empty_iterator = False
            # the arguments following the first are passed only to the
            # currently trained node, allowing the implementation of
            # supervised nodes
            if (type(x) is tuple) or (type(x) is list):
                arg = x[1:]
                x = x[0]
            else:
                arg = ()
            for nodenr in xrange(len(self.flow)):
                try:
                    node = self.flow[nodenr]
                    # check if the required number of arguments was given
                    if train_args_needed_list[nodenr]:
                        if len(train_arg_keys_list[nodenr]) != len(arg):
                            err = ("Wrong number of arguments provided by " +
                                   "the iterable for node #%d " % nodenr +
                                   "(%d needed, %d given).\n" %
                                   (len(train_arg_keys_list[nodenr]), len(arg)) +
                                   "List of required argument keys: " +
                                   str(train_arg_keys_list[nodenr]))
                            raise FlowException(err)
                    # filter x through the previous nodes
                    if node.is_training():
                        # if trainable, train current node else skip
                        if nodenr > 0:
                            x = self.flow[nodenr-1].execute(x)
                        node.train(x, *arg)
                except FlowExceptionCR as e:
                    # this exception was already propagated,
                    # probably during the execution  of a node upstream in the flow
                    (exc_type, val) = _sys.exc_info()[:2]
                    prev = ''.join(_traceback.format_exception_only(e.__class__, e))
                    prev = prev[prev.find('\n') + 1:]
                    act = "\nWhile training node #%d (%s):\n" % (nodenr,
                                                                 str(self.flow[nodenr]))
                    err_str = ''.join(('\n', 40 * '=', act, prev, 40 * '='))
                    raise FlowException(err_str)
                except Exception as e:
                    # capture any other exception occured during training.
                    self._propagate_exception(e, nodenr)
        if empty_iterator:
            if self.flow[-1].get_current_train_phase() == 1:
                err_str = ("The training data iteration "
                           "could not be repeated for the "
                           "second training phase, you probably "
                           "provided an iterator instead of an "
                           "iterable." )
                raise FlowException(err_str)
            else:
                err_str = ("The training data iterator "
                           "is empty." )
                raise FlowException(err_str)
        self._stop_training_hook()
        if self.flow[-1].get_remaining_train_phase() > 1:
            # close the previous training phase
            self.flow[-1].stop_training()

    def _train_check_iterables(self, data_iterables):
        """Return the data iterables after some checks and sanitizing.

        Note that this method does not distinguish between iterables and
        iterators, so this must be taken care of later.
        """
        # verifies that the number of iterables matches that of
        # the signal nodes and multiplies them if needed.
        flow = self.flow

        # if a single array is given IFlow trains the nodes
        # incrementally if it is a 2D array or block incrementally if the
        # array has 3d shape.
        # note that a list of 2d arrays is not valid
        if isinstance(data_iterables, numx.ndarray):
            if data_iterables.ndim == 2:
                data_iterables = [data_iterables[:,mdp.numx.newaxis,:]] * len(flow)
            else:
                data_iterables = [data_iterables] * len(flow)

        if not isinstance(data_iterables, list):
            err_str = ("'data_iterables' must be either a list of "
                       "iterables or an array, and not %s" %
                       type(data_iterables))
            raise FlowException(err_str)

        # check that all elements are iterable
        for i, iterable in enumerate(data_iterables):
            if (iterable is not None) and (not hasattr(iterable, '__iter__')):
                err = ("Element number %d in the data_iterables"
                       " list is not an iterable." % i)
                raise FlowException(err)

        # check that the number of data_iterables is correct
        if (len(data_iterables) != 1) and (len(data_iterables) != len(flow)):
            err_str = ("%d data iterables specified,"
                       " %d needed" % (len(data_iterables), len(flow)))
            raise FlowException(err_str)

        return data_iterables


    def train(self, data_iterables):
        """Train all trainable nodes in the flow.

        'data_iterables' is a list of iterables, one for each node in the flow.
        The iterators returned by the iterables must return data arrays that
        are then used for the node training (so the data arrays are the 'x' for
        the nodes). Note that the data arrays are processed by the nodes
        which are in front of the node that gets trained, so the data dimension
        must match the input dimension of the first node.

        If a node has only a single training phase then instead of an iterable
        you can alternatively provide an iterator (including generator-type
        iterators). For nodes with multiple training phases this is not
        possible, since the iterator cannot be restarted after the first
        iteration. For more information on iterators and iterables see
        http://docs.python.org/library/stdtypes.html#iterator-types .

        In the special case that 'data_iterables' is one single array,
        it is used as the data array 'x' for all nodes and training phases.

        Instead of a data array 'x' the iterators can also return a list or
        tuple, where the first entry is 'x' and the following are args for the
        training of the node (e.g. for supervised training).
        """

        #Only the first iterable is used to train all the nodes.

        data_iterable = self._train_check_iterables(data_iterables)[0]

        if self.verbose:
            strn = [str(self.flow[i]) for i in xrange(len(self.flow))]
            print("Training nodes %s simultaneously" % (strn))
        self._train_nodes(data_iterable)

        if not isinstance(self.flow[-1], INode):
            self._close_last_node()

    ###### private container methods

    def _check_value_type_is_iflow_compat(self, value):
        # inodes and non-trainable nodes are iflow compatible
        if not isinstance(value, mdp.Node):
            raise TypeError("flow item must be a Node instance")
        elif not isinstance(value, INode):
            # classic mdp Node
            if value.is_trainable():
                raise TypeError("flow item must be either an INode instance or a non-trainable Node")
        else:
            # INode
            pass

    def _check_iflow_compatibilitiy(self, flow):
        [self._check_value_type_is_iflow_compat(item) for item in flow[:-1]]
        self._check_value_type_isnode(flow[-1])

    def _get_cache_from_flow(self, flow):
        _cache = {}
        for i, node in enumerate(flow):
            if not hasattr(node, 'cache'):
                _cache['%s-%d' % (str(node), i)] = {}
            else:
                _cache['%s-%d' % (str(node), i)] = node.cache
        return _cache

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            [self._check_value_type_is_iflow_compat(item) for item in value]
        else:
            self._check_value_type_is_iflow_compat(value)

        # make a copy of list
        flow_copy = list(self.flow)
        flow_copy[key] = value
        # check dimension consistency
        self._check_nodes_consistency(flow_copy)
        [self._check_value_type_is_iflow_compat(item) for item in flow_copy[:-1]]
        # if no exception was raised, accept the new sequence
        self.flow = flow_copy
        self._cache = self._get_cache_from_flow(flow_copy)

    def __delitem__(self, key):
        # make a copy of list
        flow_copy = list(self.flow)
        del flow_copy[key]
        # check dimension consistency
        self._check_nodes_consistency(flow_copy)
        [self._check_value_type_is_iflow_compat(item) for item in flow_copy[:-1]]
        # if no exception was raised, accept the new sequence
        self.flow = flow_copy
        self._cache = self._get_cache_from_flow(flow_copy)

    def __add__(self, other):
        # append other to self
        if isinstance(other, mdp.Flow):
            flow_copy = list(self.flow).__add__(other.flow)
            # check iflow compatibility
            self._check_iflow_compatibilitiy(flow_copy)
            # check dimension consistency
            self._check_nodes_consistency(flow_copy)
            # if no exception was raised, accept the new sequence
            return self.__class__(flow_copy)
        elif isinstance(other, mdp.Node):
            flow_copy = list(self.flow)
            flow_copy.append(other)
            # check iflow compatibility
            self._check_iflow_compatibilitiy(flow_copy)
            # check dimension consistency
            self._check_nodes_consistency(flow_copy)
            # if no exception was raised, accept the new sequence
            return self.__class__(flow_copy)
        else:
            err_str = ('can only concatenate iflow or flow with trained (or non-trainable) nodes'
                       ' (not \'%s\') to iflow' % (type(other).__name__))
            raise TypeError(err_str)

    def __iadd__(self, other):
        # append other to self
        if isinstance(other, mdp.Flow):
            self.flow += other.flow
        elif isinstance(other, mdp.Node):
            self.flow.append(other)
        else:
            err_str = ('can only concatenate flow or node'
                       ' (not \'%s\') to flow' % (type(other).__name__))
            raise TypeError(err_str)
        self._check_iflow_compatibilitiy(self.flow)
        self._check_nodes_consistency(self.flow)
        self._cache = self._get_cache_from_flow(self.flow)
        return self

    ###### public container methods

    def append(self, x):
        """flow.append(node) -- append node to flow end"""
        self[len(self):len(self)] = [x]
        self._check_nodes_consistency(self.flow)
        self._cache = self._get_cache_from_flow(self.flow)

    def extend(self, x):
        """flow.extend(iterable) -- extend flow by appending
        elements from the iterable"""
        if not isinstance(x, mdp.Flow):
            err_str = ('can only concatenate flow'
                       ' (not \'%s\') to flow' % (type(x).__name__))
            raise TypeError(err_str)
        self[len(self):len(self)] = x
        self._check_nodes_consistency(self.flow)
        self._cache = self._get_cache_from_flow(self.flow)

    def insert(self, i, x):
        """flow.insert(index, node) -- insert node before index"""
        self[i:i] = [x]
        self._check_nodes_consistency(self.flow)
        self._cache = self._get_cache_from_flow(self.flow)
