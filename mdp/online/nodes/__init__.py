

__docformat__ = "restructuredtext en"

from .standard_stats_nodes import SignalAvgNode, MovingDiffNode

from .pca_nodes import CCIPCANode, WhiteningNode

from .mca_nodes import MCANode

from .sfa_nodes import IncSFANode

__all__ = ['SignalAvgNode', 'MovingDiffNode', 'CCIPCANode', 'WhiteningNode', 'MCANode',
           'IncSFANode']

from mdp.utils import fixup_namespace
fixup_namespace(__name__, __all__,
                ('standard_stats_nodes',
                 'pca_nodes',
                 'mca_nodes',
                 'sfa_nodes',
                 'fixup_namespace'
                 ))
