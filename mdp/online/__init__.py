"""
This is the MDP package for online (incremental - sample by sample) processing.

It is designed to work with nodes that can be trained indefinitely.
Nodes can be executed at any point in time while training.

Similar to the Node, an INode is the basic building block of online processing.
Being the child class, INode inherits most of Node's functionalities.

INodes are fully compatible with other Nodes and can be used in conjunction.

For a fully online hierarchical processing, the package also provides
IFlow, IFlowNode and ILayer.

"""

from .signal_inode import INode

from .onlinehinet import (
    IFlowNode, CloneILayer, SameInputILayer
)

from .onlineflows import IFlow

import nodes

__all__ = [
        'INode',
        'IFlowNode',
        'IFlow',
        'CloneILayer',
        'SameInputILayer',
        'nodes'
        ]

from mdp.utils import fixup_namespace

fixup_namespace(__name__, __all__,
                ('onlineflows',
                 'onlinehinet',
                 'signal_inode',
                 'fixup_namespace'
                 ))
