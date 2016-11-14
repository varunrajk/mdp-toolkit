
from builtins import str
import mdp
from mdp import numx
from .linear_flows import  FlowException, FlowExceptionCR, _sys, _traceback

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

    Differences between Flow and an OnlineFlow:
    a) In Flow, data is processed sequentially training one node at a time. Whereas, in an
       OnlineFlow data is processed simultaneously training all the nodes at the same time.

    b) Flow requires a list of dataiterables with a length equal to the
       number of nodes or a single numpy array. OnlineFlow requires only one
       input dataiterable as each node is trained simultaneously.

    c) Additional train args (supervised labels etc) are passed to each node through the
       node specific dataiterable. OnlineFlow requires the dataiterable to return a list
       that contains tuples of args for each node: [x, (node0 args), (node1 args), ...]. See
       train docstring.

    d) OnlineFlow also has the cache attribute that retrieves any caches variables
       stored in the OnlineNodes.

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
        err_str = ('Not used in %s'%str(type(self).__name__))
        FlowException(err_str)

    def _train_nodes(self, data_iterables):
        train_arg_keys_list = [self._get_required_train_args(node) for node in self.flow]
        train_args_needed_list = [bool(len(train_arg_keys)) for train_arg_keys in train_arg_keys_list]

        empty_iterator = True
        for x in data_iterables:
            empty_iterator = False
            for nodenr in xrange(len(self.flow)):
                # the nodenr'th arguments tuple following the first are passed only to the
                # currently trained node, allowing the implementation of
                # supervised nodes
                if (type(x) is tuple) or (type(x) is list):
                    x = x[0]
                    if nodenr >= len(x[1:]):
                        arg = ()
                    else:
                        arg = x[1:][nodenr]
                else:
                    arg = ()
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
                    if node.is_training():
                        node.train(x, *arg)
                    # input for the next node
                    x = node.execute(x)
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
        """Return the data iterable after some checks and sanitizing.

        Note that this method does not distinguish between iterables and
        iterators, so this must be taken care of later.
        """

        # if a single array is given, nodes are trained
        # incrementally if it is a 2D array or block incrementally if it
        # is a 3d array (num_blocks, block_size, dim).
        if isinstance(data_iterables, numx.ndarray):
            if data_iterables.ndim == 2:
                data_iterables = data_iterables[:,mdp.numx.newaxis,:]
            return data_iterables

        # check it it is an iterable
        if (data_iterables is not None) and (not hasattr(data_iterables, '__iter__')):
            err = ("data_iterable is not an iterable.")
            raise FlowException(err)

        return data_iterables


    def train(self, data_iterables):
        """Train all trainable nodes in the flow.

        'data_iterable' is an iterable (including generator-type  iterators) that
         must return data arrays to train nodes (so the data arrays are the 'x'
         for the nodes). Note that the data arrays are processed by the nodes
         which are in front of the node that gets trained,
         so the data dimension must match the input dimension of the first node.

        Instead of a data array 'x' the iterator can also return a list or
        tuple, where the first entry is 'x' and the following are args for
        training all the node (e.g. for supervised training). eg.:

        (x,(node0 args), (node1 args))  - only args for the first two nodes are given

        (x, (node0 args), (None), (node2 args))  - args for the first and third nodes are given

        """

        data_iterables = self._train_check_iterables(data_iterables)

        if self.verbose:
            strn = [str(self.flow[i]) for i in xrange(len(self.flow))]
            print("Training nodes %s simultaneously" % (strn))
        self._train_nodes(data_iterables)

        if not isinstance(self.flow[-1], mdp.OnlineNode):
            self._close_last_node()

    ###### private container methods

    def _check_value_type_is_compatible(self, value):
        # onlinenodes, (special case) Executable FlowNode, trained and non-trainable nodes are compatible
        if not isinstance(value, mdp.Node):
            raise TypeError("flow item must be a Node instance and not %s"%(type(value)))
        elif isinstance(value, mdp.hinet.ExecutableFlowNode) or isinstance(value, mdp.OnlineNode):
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
                _cache['node#%d' % (i)] = {}
            else:
                _cache['node#%d' % (i)] = node.cache
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



