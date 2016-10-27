"""
Module for ILayers.

"""
import mdp
from mdp.online import INode


class ILayer(mdp.hinet.Layer, INode):
    """ILayers are nodes which consist of multiple horizontally parallel INodes.

    The incoming data is split up according to the dimensions of the internal
    nodes. For example if the first node has an input_dim of 50 and the second
    node 100 then the layer will have an input_dim of 150. The first node gets
    x[:,:50], the second one x[:,50:].

    Any additional arguments are forwarded unaltered to each node.
    Warning: This might change in the next release (2.5).

    Since they are nodes themselves layers can be stacked in a flow (e.g. to
    build a layered network). If one would like to use flows instead of nodes
    inside of a layer one can use a FlowNode.
    """
    def __init__(self, nodes, dtype=None):
        """Setup the layer with the given list of nodes.

        The input and output dimensions for the nodes must be already set
        (the output dimensions for simplicity reasons).

        Keyword arguments:
        nodes -- List of the nodes to be used.
        """
        super(ILayer, self).__init__(nodes, dtype=dtype)
        self._cache = {'%s-%d'%(str(node),i): node._cache for i,node in enumerate(nodes)}

    def train(self, x, *args, **kwargs):
        super(ILayer, self).train(x, *args, **kwargs)

    def _get_train_seq(self):
        """Return the train sequence.
           Use INode train_seq not layer's
        """
        return INode._get_train_seq(self)



class CloneILayer(mdp.hinet.CloneLayer, ILayer):
    """ILayer with a single node instance that is used multiple times.

    The same single node instance is used to build the layer, so
    CloneIlayer(node, 3) executes in the same way as ILayer([node]*3).
    But ILayer([node]*3) would have a problem when closing a training phase,
    so one has to use CloneILayer.

    An CloneILayer can be used for weight sharing in the training phase. It might
    be also useful for reducing the memory footprint use during the execution
    phase (since only a single node instance is needed).
    """

    def __init__(self, node, n_nodes=1, dtype=None):
        """Setup the layer with the given list of nodes.

        Keyword arguments:
        node -- Node to be cloned.
        n_nodes -- Number of repetitions/clones of the given node.
        """
        super(CloneILayer, self).__init__(node=node, n_nodes=n_nodes, dtype=dtype)


class SameInputILayer(mdp.hinet.SameInputLayer, ILayer):
    """SameInputILayer is an ILayer were all nodes receive the full input.

    So instead of splitting the input according to node dimensions, all nodes
    receive the complete input data.
    """

    def __init__(self, nodes, dtype=None):
        """Setup the layer with the given list of nodes.

        The input dimensions for the nodes must all be equal, the output
        dimensions can differ (but must be set as well for simplicity reasons).

        Keyword arguments:
        nodes -- List of the nodes to be used.
        """
        super(SameInputILayer, self).__init__(nodes=nodes, dtype=dtype)


class IFlowNode(mdp.hinet.FlowNode, INode):
    """IFlowNode wraps an IFlow of Nodes into a single INode.

    This is handy if you want to use a iflow where a INode is required.
    Additional args and kwargs for train and execute are supported.

    All the read-only container slots are supported and are forwarded to the
    internal flow.
    """
    def __init__(self, flow, input_dim=None, output_dim=None, dtype=None):
        super(IFlowNode, self).__init__(flow=flow, input_dim=input_dim, output_dim=output_dim, dtype=dtype)
        self._cache = self.flow._cache


