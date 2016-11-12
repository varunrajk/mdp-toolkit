
from builtins import str
import mdp
from mdp import numx
from mdp.linear_flows import  FlowException, FlowExceptionCR, _sys, _traceback
from mdp.online import OnlineNode, ExecutableNode
from collections import deque as _deque

class OnlineFlow(mdp.Flow):
    """An 'OnlineFlow' is a sequence of online/executable nodes that are trained and executed
    together to form a more complex algorithm.  Input data is sent to the
    first node and is successively processed by the subsequent nodes along
    the sequence.

    Using an online flow as opposed to handling manually a set of nodes has a
    clear advantage: The general online flow implementation automatizes the
    training (including supervised training and multiple training phases),
    execution, and inverse execution (if defined) of the whole sequence.

    OnlineFlow sequence can contain trained or non-trainable Nodes and optionally
    a terminal trainable Node.

    Crash recovery is optionally available: in case of failure the current
    state of the flow is saved for later inspection.

    OnlineFlow objects are Python containers. Most of the builtin 'list'
    methods are available. An 'OnlineFlow' can be saved or copied using the
    corresponding 'save' and 'copy' methods.

    OnlineFlow like an OnlineNode has 'cache' that is a dict of stored cache variables
    from individual nodes.
    """

    def __init__(self, flow, crash_recovery=False, verbose=False):
        super(OnlineFlow, self).__init__(flow, crash_recovery, verbose)
        # check if the list of nodes is compatible.
        self._check_compatibilitiy(flow)
        # collect cache from individual nodes.
        self._cache = self._get_cache_from_flow(flow)

    @property
    def cache(self):
        return self._cache

    def _train_node(self, data_iterable, nodenr):
        err_str = ('Not used in OnlineFlow')
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
                    # filter x through the previous node
                    if nodenr > 0:
                        x = self.flow[nodenr-1].execute(x)
                    if node.is_training():
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

    def _train_check_iterables(self, data_iterable):
        """Return the data iterable after some checks and sanitizing.

        Note that this method does not distinguish between iterables and
        iterators, so this must be taken care of later.
        """

        # if a single array is given OnlineFlow trains the nodes
        # incrementally if it is a 2D array or block incrementally if the
        # array has 3d shape (num_blocks, block_size, dim).
        if isinstance(data_iterable, numx.ndarray):
            if data_iterable.ndim == 2:
                data_iterable = data_iterable[:,mdp.numx.newaxis,:]
            return data_iterable

        # check it it is an iterable
        if (data_iterable is not None) and (not hasattr(data_iterable, '__iter__')):
            err = ("data_iterable is not an iterable.")
            raise FlowException(err)

        return data_iterable


    def train(self, data_iterable):
        """Train all trainable nodes in the flow.

        'data_iterable' is an iterable that must return data arrays to train nodes
         (so the data arrays are the 'x' for the nodes). Note that the data arrays
         are processed by the nodes which are in front of the node that gets trained,
         so the data dimension must match the input dimension of the first node.

        If a node has only a single training phase then instead of an iterable
        you can alternatively provide an iterator (including generator-type
        iterators). For nodes with multiple training phases this is not
        possible, since the iterator cannot be restarted after the first
        iteration. For more information on iterators and iterables see
        http://docs.python.org/library/stdtypes.html#iterator-types .

        In the special case that 'data_iterable' is one single array,
        it is used as the data array 'x' for all nodes and training phases.

        Instead of a data array 'x' the iterator can also return a list or
        tuple, where the first entry is 'x' and the following are args for the
        training of the node (e.g. for supervised training).
        """

        data_iterable = self._train_check_iterables(data_iterable)

        if self.verbose:
            strn = [str(self.flow[i]) for i in xrange(len(self.flow))]
            print("Training nodes %s simultaneously" % (strn))
        self._train_nodes(data_iterable)

        if not isinstance(self.flow[-1], OnlineNode):
            self._close_last_node()

    ###### private container methods

    def _check_value_type_is_compatible(self, value):
        # onlinenodes, trained and non-trainable nodes are compatible
        if not isinstance(value, mdp.Node):
            raise TypeError("flow item must be a Node instance and not %s"%(type(value)))
        elif isinstance(value, ExecutableNode) or isinstance(value, OnlineNode):
            pass
        else:
            # classic mdp Node
            if value.is_training():
                raise TypeError("flow item must either be an OnlineNode instance, a trained or a non-trainable Node.")

    def _check_compatibilitiy(self, flow):
        [self._check_value_type_is_compatible(item) for item in flow[:-1]]
        # terminal node can be a trainable Node
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
            [self._check_value_type_is_compatible(item) for item in value]
        else:
            self._check_value_type_is_compatible(value)

        # make a copy of list
        flow_copy = list(self.flow)
        flow_copy[key] = value
        # check dimension consistency
        self._check_nodes_consistency(flow_copy)
        self._check_compatibilitiy(flow_copy)
        # if no exception was raised, accept the new sequence
        self.flow = flow_copy
        self._cache = self._get_cache_from_flow(flow_copy)

    def __delitem__(self, key):
        # make a copy of list
        flow_copy = list(self.flow)
        del flow_copy[key]
        # check dimension consistency
        self._check_nodes_consistency(flow_copy)
        self._check_compatibilitiy(flow_copy)
        # if no exception was raised, accept the new sequence
        self.flow = flow_copy
        self._cache = self._get_cache_from_flow(flow_copy)

    def __add__(self, other):
        # append other to self
        if isinstance(other, mdp.Flow):
            flow_copy = list(self.flow).__add__(other.flow)
            # check iflow compatibility
            self._check_compatibilitiy(flow_copy)
            # check dimension consistency
            self._check_nodes_consistency(flow_copy)
            # if no exception was raised, accept the new sequence
            return self.__class__(flow_copy)
        elif isinstance(other, mdp.Node):
            flow_copy = list(self.flow)
            flow_copy.append(other)
            # check iflow compatibility
            self._check_compatibilitiy(flow_copy)
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
        self._check_compatibilitiy(self.flow)
        self._check_nodes_consistency(self.flow)
        self._cache = self._get_cache_from_flow(self.flow)
        return self

    ###### public container methods

    def append(self, x):
        """flow.append(node) -- append node to flow end"""
        self[len(self):len(self)] = [x]
        self._check_nodes_consistency(self.flow)
        self._check_compatibilitiy(self.flow)
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
        self._check_compatibilitiy(self.flow)
        self._cache = self._get_cache_from_flow(self.flow)

    def insert(self, i, x):
        """flow.insert(index, node) -- insert node before index"""
        self[i:i] = [x]
        self._check_nodes_consistency(self.flow)
        self._check_compatibilitiy(self.flow)
        self._cache = self._get_cache_from_flow(self.flow)




class CircularFlow(OnlineFlow):
    """A 'CircularFlow' is a cyclic sequence of online/executable nodes that are trained and executed
    together to form a more complex algorithm.  Input data can be optionally sent to any node
    in the sequence and is successively processed by the subsequent nodes along the loop.
    In the absence of external Input data, each node receives input from the previous node in the sequence.

    CircularFlow sequence can contain trained or non-trainable Nodes.

    Crash recovery is optionally available: in case of failure the current
    state of the flow is saved for later inspection.

    CircularFlow objects are Python containers. Most of the builtin 'list'
    methods are available. CircularFlow can be saved or copied using the
    corresponding 'save' and 'copy' methods.

    CircularFlow like an OnlineNode has 'cache' that is a dict of stored cache variables
    from individual nodes.

    - set an input node ptr and output node ptr
    - input and output dims need to be reset based on these
    - train can take data_iterables or scalar cnt to run loop

    """

    def __init__(self, flow, crash_recovery=False, verbose=False):
        super(CircularFlow, self).__init__(flow, crash_recovery, verbose)
        self.flow = _deque(flow)
        self.output_node_idx = len(self.flow)-1
        self._stored_input = None

    def set_stored_input(self, x):
        if self.flow[0].input_dim is not None:
            if x.shape[-1] != self.flow[0].input_dim:
                raise FlowException("Dimension mismatch! should be %d, given %d"%(self.flow[0].input_dim, x.shape[-1]))
            self._stored_input = x

    def get_stored_input(self):
        return self._stored_input

    def _train_nodes(self, data_iterable):
        train_arg_keys_list = [self._get_required_train_args(node) for node in self.flow]
        train_args_needed_list = [bool(len(train_arg_keys)) for train_arg_keys in train_arg_keys_list]

        for x in data_iterable:
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
                    # filter x through the previous node
                    if nodenr > 0:
                        x = self.flow[nodenr-1].execute(x)
                    if node.is_training():
                        node.train(x, *arg)
                    if nodenr == (len(self.flow)-1):
                        x = node.execute(x)
                        if len(arg) > 1:
                            self._stored_input = (x, arg)
                        else:
                            self._stored_input = x
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

    def _train_check_iterables(self, data_iterable):
        """Return the data iterable after some checks and sanitizing.

        Note that this method does not distinguish between iterables and
        iterators, so this must be taken care of later.
        """
        if mdp.numx.isscalar(data_iterable):
            cnt = data_iterable
            def iterfn():
                for i in xrange(cnt):
                    d = self.get_stored_input()
                    yield d
            data_iterable = iterfn()

        # if a single array is given OnlineFlow trains the nodes
        # incrementally if it is a 2D array or block incrementally if the
        # array has 3d shape (num_blocks, block_size, dim).
        if isinstance(data_iterable, numx.ndarray):
            if data_iterable.ndim == 2:
                data_iterable = data_iterable[:,mdp.numx.newaxis,:]
            return data_iterable

        # check it it is an iterable
        if (data_iterable is not None) and (not hasattr(data_iterable, '__iter__')):
            err = ("data_iterable is not an iterable.")
            raise FlowException(err)

        return data_iterable

    def train(self, data_iterable=1):
        """Train all trainable nodes in the flow.

        'data_iterable' is either an iterator that returns data arrays to train the
        nodes (so the data arrays are the 'x' for
        the nodes) or it can be a scalar value that trains the loop with the stored inputs.

        Note that the data arrays are processed by the nodes
        which are in front of the node that gets trained, so the data dimension
        must match the input dimension of the current input node.

        Instead of a data array 'x' the iterators can also return a list or
        tuple, where the first entry is 'x' and the following are args for the
        training of the node (e.g. for supervised training).
        """

        if self.verbose:
            strn = [str(self.flow[i]) for i in xrange(len(self.flow))]
            if mdp.numx.isscalar(data_iterable):
                print ("Training nodes %s simultaneously using the stored inputs for %d loops"%(strn, data_iterable))
            else:
                print("Training nodes %s simultaneously using  the given inputs" % (strn))

        data_iterable = self._train_check_iterables(data_iterable)
        self._train_nodes(data_iterable)


    def execute(self, iterable):
        """Process the data through all nodes between input and the output node.

        'iterable' is an iterable or iterator (note that a list is also an
        iterable), which returns data arrays that are used as input to the flow.
        Alternatively, one can specify one data array as input.

        """
        if isinstance(iterable, numx.ndarray):
            return self._execute_seq(iterable, self.output_node_idx)
        res = []
        empty_iterator = True
        for x in iterable:
            empty_iterator = False
            res.append(self._execute_seq(x, self.output_node_idx))
        if empty_iterator:
            errstr = ("The execute data iterator is empty.")
            raise FlowException(errstr)
        return numx.concatenate(res)

    def _inverse_seq(self, x):
        #Successively invert input data 'x' through all nodes backwards from the output node to the input node.
        flow = self.flow[:self.output_node_idx]
        for i in range(len(flow)-1, -1, -1):
            try:
                x = flow[i].inverse(x)
            except Exception as e:
                self._propagate_exception(e, i)
        return x

    def __call__(self, iterable):
        """Calling an instance is equivalent to call its 'execute' method."""
        return self.execute(iterable)


    def set_input_node(self, node_idx):
        if (node_idx > len(self.flow)) or (node_idx < 0):
            raise FlowException("Accepted 'node_idx' values: 0 <= node_idx < %d, given %d"%(len(self.flow), node_idx))
        self.flow.rotate(-node_idx)
        self.output_node_idx = (self.output_node_idx-node_idx)%len(self.flow)
        self._input_dim = self.flow[0].input_dim


    def set_output_node(self, node_idx):
        if (node_idx > len(self.flow)) or (node_idx < 0):
            raise FlowException("Accepted 'node_idx' values: 0 <= node_idx < %d, given %d"%(len(self.flow), node_idx))
        self.output_node_idx = node_idx
        self._output_dim = self.flow[self.output_node_idx].output_dim


    def _check_compatibilitiy(self, flow):
        [self._check_value_type_is_compatible(item) for item in flow]

    def reset_output_node(self):
        self.output_node_idx = len(self.flow)-1


    def __setitem__(self, key, value):
        super(CircularFlow, self).__setitem__(key, value)
        self.flow = _deque(self.flow)
        if (key.start < self.output_node_idx) and (self.output_node_idx < key.stop()):
            print 'Output node is replaced! Resetting the output node.'
            self.reset_output_node()

    def __delitem__(self, key):
        super(CircularFlow, self).__delitem__(key)
        self.flow = _deque(self.flow)
        if (key.start < self.output_node_idx) and (self.output_node_idx < key.stop()):
            print 'Output node deleted! Resetting the output node to the default last node.'
            self.reset_output_node()
        elif self.output_node_idx > key.stop():
            self.set_output_node(self.output_node_idx-key.stop+key.start)

    def __add__(self, other):
        # append other to self
        if isinstance(other, OnlineFlow):
            flow_copy = list(self.flow)
            flow_copy.append(other)
            self._check_compatibilitiy(flow_copy)
            self._check_nodes_consistency(flow_copy)
            # if no exception was raised, accept the new sequence
            return self.__class__(flow_copy)
        elif isinstance(other, OnlineNode):
            flow_copy = list(self.flow)
            flow_copy.append(other)
            # check onlineflow compatibility
            self._check_compatibilitiy(flow_copy)
            # check dimension consistency
            self._check_nodes_consistency(flow_copy)
            # if no exception was raised, accept the new sequence
            return self.__class__(flow_copy)
        else:
            err_str = ('can only concatenate onlineflow or onlinenode'
                       ' (not \'%s\') to circularflow' % (type(other).__name__))
            raise TypeError(err_str)

    def __iadd__(self, other):
        # append other to self
        if isinstance(other, OnlineFlow):
            self.flow += other.flow
        elif isinstance(other, OnlineNode):
            self.flow.append(other)
        else:
            err_str = ('can only concatenate onlineflow or onlinenode'
                       ' (not \'%s\') to flow' % (type(other).__name__))
            raise TypeError(err_str)
        self._check_compatibilitiy(self.flow)
        self._check_nodes_consistency(self.flow)
        self._cache = self._get_cache_from_flow(self.flow)
        return self


    ###### public container methods

    def append(self, x):
        """flow.append(node) -- append node to flow end"""
        self[len(self):len(self)] = [x]
        self._check_nodes_consistency(self.flow)
        self._check_compatibilitiy(self.flow)
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
        self._check_compatibilitiy(self.flow)
        self._cache = self._get_cache_from_flow(self.flow)


    def insert(self, i, x):
        """flow.insert(index, node) -- insert node before index"""
        self[i:i] = [x]
        self._check_nodes_consistency(self.flow)
        self._check_compatibilitiy(self.flow)
        self._cache = self._get_cache_from_flow(self.flow)

        if self.output_node_idx >= i:
            self.set_output_node(self.output_node_idx+1)





