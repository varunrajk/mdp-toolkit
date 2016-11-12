"""
This is the MDP package for online (incremental - sample by sample) processing.

It is designed to work with nodes that can be trained indefinitely.
Nodes can be executed at any point in time while training.

Similar to the Node, an OnlineNode is the basic building block of online processing.
Being the child class, OnlineNode inherits most of Node's functionalities.

OnlineNodes are fully compatible with other Nodes and can be used in conjunction.

For a fully online hierarchical processing, the package also provides
OnlineFlow, OnlineFlowNode and OnlineLayer.

"""

from .signal_inode import OnlineNode, PreserveDimOnlineNode

from .online_hinet import (
    OnlineFlowNode, CloneOnlineLayer, SameInputOnlineLayer, ExecutableNode
)

from .online_flows import OnlineFlow, CircularFlow

import nodes
import utils
from .test import test

__all__ = [
        'OnlineNode', 'PreserveDimOnlineNode','OnlineFlowNode',
        'OnlineFlow', 'CircularFlow','CloneOnlineLayer','SameInputOnlineLayer',
        'ExecutableNode',
        'nodes', 'utils'
        ]

from mdp.utils import fixup_namespace

fixup_namespace(__name__, __all__,
                ('onlineflows',
                 'onlinehinet',
                 'signal_inode',
                 'fixup_namespace'
                 ))
