"""
This is the MDP package for parallel processing.

It is designed to work with nodes for which a large part of the computations
are embaressingly parallel (like in PCANode). The hinet package is also fully
supported, i.e., there are parallel versions of all hinet nodes.

This package consists of two decoupled parts. The first part consists of
parallel versions of the familiar MDP structures (nodes and flows). At the top
there is the ParallelFlow, which generates tasks that can be processed in 
parallel. 
The second part consists of the schedulers. They take tasks and process them in 
a more or less parallel way (e.g. in multiple processes). So they are designed
to deal with the more technical aspects of the parallelization, but do not
have to know anything about flows or nodes.
"""


from scheduling import (ResultContainer, ListResultContainer,
                        OrderedResultContainer, TaskCallable, SqrTestCallable, 
                        Scheduler)
from process_schedule import ProcessScheduler
from parallelnodes import (ParallelNode, TrainingPhaseNotParallelException,
                           ParallelPCANode, ParallelWhiteningNode,
                           ParallelSFANode, ParallelSFA2Node)
from parallelflows import (FlowTrainCallable, FlowExecuteCallable,
                           NodeResultContainer,
                           ParallelFlowException, NoTaskException,
                           ParallelFlow, ParallelCheckpointFlow)
from parallelhinet import (ParallelFlowNode, ParallelLayer, ParallelCloneLayer)
from makeparallel import make_flow_parallel, unmake_flow_parallel

del scheduling
del process_schedule
del parallelnodes
del parallelflows
del parallelhinet
del makeparallel
