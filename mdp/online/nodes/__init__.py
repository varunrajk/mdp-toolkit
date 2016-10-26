

__docformat__ = "restructuredtext en"

from .standard_stats_nodes import SignalAvgNode

__all__ = ['SignalAvgNode']

from mdp.utils import fixup_namespace
fixup_namespace(__name__, __all__,
                ('standard_stats_nodes',
                 'fixup_namespace'
                 ))
