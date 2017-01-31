"""
Module for OnlineLayers.

"""
import mdp
from .layer import Layer, CloneLayer, SameInputLayer

class OnlineLayer(Layer, mdp.OnlineNode):
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
        self._check_compatibility(nodes)
        self._cache = self._get_cache_from_nodes(nodes)
        # numx_rng will not be set through the super call.
        # Have to set it explicitly here:
        self.numx_rng = numx_rng
        # set training type
        self._set_training_type_from_nodes(nodes)

    def _check_compatibility(self, nodes):
        [self._check_value_type_is_compatible(item) for item in nodes]

    def _check_value_type_is_compatible(self, value):
        # onlinenodes, trained and non-trainable nodes are compatible
        if not isinstance(value, mdp.Node):
            raise TypeError("'nodes' item must be a Node instance and not %s"%(type(value)))
        elif isinstance(value, mdp.OnlineNode):
            pass
        else:
            # classic mdp Node
            if value.is_training():
                raise TypeError("'nodes' item must either be an OnlineNode, a trained or a non-trainable Node.")

    def _get_cache_from_nodes(self, nodes):
        _cache = {}
        for i, node in enumerate(nodes):
            if not hasattr(node, 'cache'):
                _cache['node#%d' % (i)] = {}
            else:
                _cache['node#%d' % (i)] = node.cache
        return _cache

    def _set_training_type_from_nodes(self, nodes):
        for node in nodes:
            if hasattr(node, 'training_type') and (node.training_type == 'incremental'):
                self._training_type = 'incremental'
                return
        self._training_type = 'batch'

    def set_training_type(self, training_type):
        if self.training_type != training_type:
            raise mdp.NodeException("Cannot change the training type to %s. It is inferred from "
                                    "the nodes and is set to '%s'. "%(training_type, self.training_type))

    def _set_numx_rng(self, rng):
        # set the numx_rng for all the nodes to be the same.
        for node in self.nodes:
            if hasattr(node, 'set_numx_rng'):
                node.numx_rng = rng
        self._numx_rng = rng

    def _get_train_seq(self):
        """Return the train sequence.
           Use OnlineNode train_seq not Layer's
        """
        return mdp.OnlineNode._get_train_seq(self)



class CloneOnlineLayer(CloneLayer, OnlineLayer):
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
        self._check_compatibility([node])
        self._cache = node.cache
        # numx_rng will not be set through the super call.
        # Have to set it explicitly here:
        self.numx_rng = numx_rng
        # set training type
        self._set_training_type_from_nodes([node])


class SameInputOnlineLayer(SameInputLayer, OnlineLayer):
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
        self._check_compatibility(nodes)
        self._cache = self._get_cache_from_nodes(nodes)
        # numx_rng will not be set through the super call.
        # Have to set it explicitly here:
        self.numx_rng = numx_rng
        # set training type
        self._set_training_type_from_nodes(nodes)

