

__docformat__ = "restructuredtext en"

from .stats_nodes_online import SignalAvgNode, MovingTimeDiffNode

from .pca_nodes_online import CCIPCANode, CCIPCAWhiteningNode

from .mca_nodes_online import MCANode

from .sfa_nodes_online import IncSFANode

__all__ = ['SignalAvgNode', 'MovingTimeDiffNode', 'CCIPCANode', 'CCIPCAWhiteningNode', 'MCANode',
           'IncSFANode']

from mdp.utils import fixup_namespace
fixup_namespace(__name__, __all__,
                ('standard_stats_nodes',
                 'pca_nodes',
                 'mca_nodes',
                 'sfa_nodes',
                 'fixup_namespace'
                 ))
