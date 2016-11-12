"""
Module for OnlineLayers.

"""
import mdp
from mdp.online import OnlineNode


class OnlineLayer(mdp.hinet.Layer, OnlineNode):
    """OnlineLayers are nodes which consist of multiple horizontally parallel OnlineNodes.
    OnlineLayer also supports trained or non-trainable Nodes.

    The incoming data is split up according to the dimensions of the internal
    nodes. For example if the first node has an input_dim of 50 and the second
    node 100 then the layer will have an input_dim of 150. The first node gets
    x[:,:50], the second one x[:,50:].

    Any additional arguments are forwarded unaltered to each node.
    Warning: This might change in the next release (2.5).

    Since they are nodes themselves layers can be stacked in a flow (e.g. to
    build a layered network). If one would like to use flows instead of nodes
    inside of a layer one can use an OnlineFlowNode.
    """
    def __init__(self, nodes, dtype=None, numx_rng=None):
        """Setup the layer with the given list of nodes.

        The input and output dimensions for the nodes must be already set
        (the output dimensions for simplicity reasons).

        Keyword arguments:
        nodes -- List of the nodes to be used.
        """
        super(OnlineLayer, self).__init__(nodes, dtype=dtype)
        self._check_compatibilitiy(nodes)
        self._cache = self._get_cache_from_nodes(nodes)
        # numx_rng will not be set through the super call.
        # Have to set it explicitly here:
        self._numx_rng = None
        self.set_numx_rng(numx_rng)
        # set training type
        self._set_training_type_from_nodes(nodes)

    def _check_compatibilitiy(self, nodes):
        [self._check_value_type_is_compatible(item) for item in nodes]

    def _check_value_type_is_compatible(self, value):
        # onlinenodes, trained and non-trainable nodes are compatible
        if not isinstance(value, mdp.Node):
            raise TypeError("'nodes' item must be a Node instance and not %s"%(type(value)))
        elif isinstance(value, ExecutableNode) or isinstance(value, OnlineNode):
            pass
        else:
            # classic mdp Node
            if value.is_training():
                raise TypeError("'nodes' item must either be an OnlineNode, a trained or a non-trainable Node.")

    def set_cache(self, c):
        raise mdp.NodeException("Can't set the read only cache attribute. ")

    def _get_cache_from_nodes(self, nodes):
        _cache = {}
        for i, node in enumerate(nodes):
            if not hasattr(node, 'cache'):
                _cache['%s-%d' % (str(node), i)] = {}
            else:
                _cache['%s-%d' % (str(node), i)] = node.cache
        return _cache

    def _set_training_type_from_nodes(self, nodes):
        for node in nodes:
            if hasattr(node, 'training_type') and (node.training_type == 'incremental'):
                self._training_type = 'incremental'
                return
        self._training_type = 'batch'

    def set_numx_rng(self, rng):
        super(OnlineLayer, self).set_numx_rng(rng)
        # set the numx_rng for all the nodes to be the same.
        for node in self.nodes:
            if hasattr(node, 'set_numx_rng'):
                node.set_numx_rng(rng)

    def train(self, x, *args, **kwargs):
        super(OnlineLayer, self).train(x, *args, **kwargs)

    def _get_train_seq(self):
        """Return the train sequence.
           Use OnlineNode train_seq not Layer's
        """
        return OnlineNode._get_train_seq(self)



class CloneOnlineLayer(mdp.hinet.CloneLayer, OnlineLayer):
    """OnlineLayer with a single node instance that is used multiple times.

    The same single node instance is used to build the layer, so
    CloneIlayer(node, 3) executes in the same way as OnlineLayer([node]*3).
    But OnlineLayer([node]*3) would have a problem when closing a training phase,
    so one has to use CloneOnlineLayer.

    An CloneOnlineLayer can be used for weight sharing in the training phase. It might
    be also useful for reducing the memory footprint use during the execution
    phase (since only a single node instance is needed).
    """

    def __init__(self, node, n_nodes=1, dtype=None, numx_rng=None):
        """Setup the layer with the given list of nodes.

        Keyword arguments:
        node -- Node to be cloned.
        n_nodes -- Number of repetitions/clones of the given node.
        """
        super(CloneOnlineLayer, self).__init__(node=node, n_nodes=n_nodes, dtype=dtype)
        self._check_compatibilitiy([node])
        self._cache = node.cache
        # numx_rng will not be set through the super call.
        # Have to set it explicitly here:
        self._numx_rng = None
        self.set_numx_rng(numx_rng)
        # set training type
        self._set_training_type_from_nodes([node])


class SameInputOnlineLayer(mdp.hinet.SameInputLayer, OnlineLayer):
    """SameInputOnlineLayer is an OnlineLayer were all nodes receive the full input.

    So instead of splitting the input according to node dimensions, all nodes
    receive the complete input data.
    """

    def __init__(self, nodes, dtype=None, numx_rng=None):
        """Setup the layer with the given list of nodes.

        The input dimensions for the nodes must all be equal, the output
        dimensions can differ (but must be set as well for simplicity reasons).

        Keyword arguments:
        nodes -- List of the nodes to be used.
        """
        super(SameInputOnlineLayer, self).__init__(nodes=nodes, dtype=dtype)
        self._check_compatibilitiy(nodes)
        self._cache = self._get_cache_from_nodes(nodes)
        # numx_rng will not be set through the super call.
        # Have to set it explicitly here:
        self._numx_rng = None
        self.set_numx_rng(numx_rng)
        # set training type
        self._set_training_type_from_nodes(nodes)


class OnlineFlowNode(mdp.hinet.FlowNode, OnlineNode):
    """OnlineFlowNode wraps an OnlineFlow of OnlineNodes into a single OnlineNode.

    This is handy if you want to use a OnlineFlow where a OnlineNode is required.
    Additional args and kwargs for train and execute are supported.

    Unlike an OnlineFlow, OnlineFlowNode only supports either a
    terminal OnlineNode, trained or non-trainable Node.

    All the read-only container slots are supported and are forwarded to the
    internal flow.
    """
    def __init__(self, flow, input_dim=None, output_dim=None, dtype=None, numx_rng=None):
        super(OnlineFlowNode, self).__init__(flow=flow, input_dim=input_dim, output_dim=output_dim, dtype=dtype)
        self._check_compatibilitiy(flow)
        self._cache = flow.cache
        # numx_rng will not be set through the super call.
        # Have to set it explicitly here:
        self._numx_rng = None
        self.set_numx_rng(numx_rng)
        # set training type
        self._set_training_type_from_flow(flow)

    def _check_compatibilitiy(self, flow):
        if not isinstance(flow, mdp.online.OnlineFlow):
            raise TypeError("Flow must be an OnlineFlow type and not %s"%(type(flow)))
        if not isinstance(flow[-1], mdp.Node):
            raise TypeError("Flow item must be a Node instance and not %s"%(type(flow[-1])))
        elif isinstance(flow[-1], OnlineNode) or isinstance(flow[-1], ExecutableNode):
            pass
        else:
            # classic mdp Node
            if flow[-1].is_training():
                raise TypeError("OnlineFlowNode supports either only a terminal OnlineNode, a trained or a non-trainable Node.")


    def _set_training_type_from_flow(self, flow):
        for node in flow:
            if hasattr(node, 'training_type') and (node.training_type == 'incremental'):
                self._training_type = 'incremental'
                return
        self._training_type = 'batch'

    def set_training_type(self, training_type):
        if self.training_type != training_type:
            raise mdp.NodeException("Cannot change the training type to %s. It is inferred from "
                                    "the flow and is set to '%s'. "%(training_type, self.training_type))

    def set_cache(self, c):
        raise mdp.NodeException("Cannot set the read only cache attribute. ")

    def set_numx_rng(self, rng):
        super(OnlineFlowNode, self).set_numx_rng(rng)
        # set the numx_rng for all the nodes to be the same.
        for node in self._flow:
            if hasattr(node, 'set_numx_rng'):
                node.set_numx_rng(rng)


class ExecutableNode(mdp.hinet.FlowNode):
    """ExecutableNode wraps a FlowNode over a Node, or a list of Nodes or a Flow, to make it executable without
    calling the stop_training function implicitly.

    An _interim_execute function is called until the training is complete.
    Once the training is complete (done by calling the required number of stop_training
     calls explicitly), the standard _execute function is called from then on.

    This node could be useful to combine Nodes with OnlineNodes in an OnlineFlow.

    """
    def __init__(self, flow, input_dim=None, output_dim=None, dtype=None):
        if isinstance(flow, mdp.Node):
            flow = mdp.Flow([flow])
        elif isinstance(flow, list):
            flow = mdp.Flow(flow)
        super(ExecutableNode, self).__init__(flow=flow, input_dim=input_dim, output_dim=output_dim, dtype=dtype)

    def _interim_execute(self, x, *args, **kwargs):
        # Can be replaced in a subclass
        outx = mdp.numx.zeros((x.shape[0], self.output_dim))
        outx[:] = None
        return outx

    def execute(self, x, *args, **kwargs):
        """Process the data contained in `x`.

        If the object is still in the training phase, the function
        `stop_training` will be called.
        `x` is a matrix having different variables on different columns
        and observations on the rows.

        By default, subclasses should overwrite `_execute` to implement
        their execution phase. The docstring of the `_execute` method
        overwrites this docstring.
        """
        if self.is_training():
            # control the dimension x
            self._check_input(x)
            # set the output dimension if necessary
            if self.output_dim is None:
                self.output_dim = self.input_dim
            return self._interim_execute(self._refcast(x), *args, **kwargs)
        else:
            self._pre_execution_checks(x)
            return self._execute(self._refcast(x), *args, **kwargs)
