"""
Module for the OnlineFlowNode class.
"""

import mdp
from .flownode import FlowNode

class OnlineFlowNode(FlowNode, mdp.OnlineNode):
    """OnlineFlowNode wraps an OnlineFlow of OnlineNodes into a single OnlineNode.

    This is handy if you want to use a OnlineFlow where a OnlineNode is required.
    Additional args and kwargs for train and execute are supported.

    Unlike an OnlineFlow, OnlineFlowNode requires all the nodes to be either
     an OnlineNode, trained or non-trainable Node.

    All the read-only container slots are supported and are forwarded to the
    internal flow.
    """
    def __init__(self, flow, input_dim=None, output_dim=None, dtype=None, numx_rng=None):
        super(OnlineFlowNode, self).__init__(flow=flow, input_dim=input_dim, output_dim=output_dim, dtype=dtype)
        self._check_compatibilitiy(flow)
        self._cache = flow.cache
        # numx_rng will not be set through the super call.
        # Have to set it explicitly here:
        self.numx_rng = numx_rng
        # set training type
        self._set_training_type_from_flow(flow)

    def _check_compatibilitiy(self, flow):
        if not isinstance(flow, mdp.OnlineFlow):
            raise TypeError("Flow must be an OnlineFlow type and not %s"%(type(flow)))
        if not isinstance(flow[-1], mdp.Node):
            raise TypeError("Flow item must be a Node instance and not %s"%(type(flow[-1])))
        elif isinstance(flow[-1], mdp.OnlineNode):
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

    def _set_numx_rng(self, rng):
        # set the numx_rng for all the nodes to be the same.
        for node in self._flow:
            if hasattr(node, 'set_numx_rng'):
                node.numx_rng = rng
        self._numx_rng = rng

    def _get_train_seq(self):
        """Return a training sequence containing all training phases."""
        # Unlike thw FlowNode, the OnlineFlowNode requires only
        # one train_seq item for each node. Each node's train function
        # takes care of its multiple train phases (if any).
        train_seq = []
        last_tr_node_ptr = 0
        for i, node in enumerate(self._flow):
            if node.is_trainable():
                last_tr_node_ptr = i
                train_seq +=[(node.train, node.stop_training, node.execute)]
            else:
                train_seq +=[(lambda x, *args, **kwargs: None, lambda x, *args, **kwargs: None, node.execute)]

        # skip the non-trainable terminal phases (if any)
        train_seq = train_seq[:(last_tr_node_ptr+1)]

        # And it is unnecessary to fix the dimensions like in the FlowNode, as the terminal
        # execute phases takes care of it.
        return train_seq


class CircularOnlineFlowNode(FlowNode, mdp.OnlineNode):
    """CircularOnlineFlowNode wraps a CircularOnlineFlow of OnlineNodes into a single OnlineNode.

    This is handy if you want to use a CircularOnlineFlow where a OnlineNode is required.

    Once the node is initialized, the _flow_iterations and _ignore_input values of a CircularOnlineFlow cannot
    be changed.

    However, the stored_input can be changed (or set) using 'set_stored_input' method.

    All the read-only container slots are supported and are forwarded to the
    internal flow.
    """
    def __init__(self, flow, input_dim=None, output_dim=None, dtype=None, numx_rng=None):
        super(CircularOnlineFlowNode, self).__init__(flow=flow, input_dim=input_dim, output_dim=output_dim, dtype=dtype)
        self._check_compatibilitiy(flow)
        self._cache = flow.cache
        # numx_rng will not be set through the super call.
        # Have to set it explicitly here:
        self.numx_rng = numx_rng
        # set training type
        self._set_training_type_from_flow(flow)
        # get stored_input, flow_iterations and ignore_input flags from flow
        self._flow_iterations = flow._flow_iterations
        self._ignore_input = flow._ignore_input
        self._stored_input = flow._stored_input


    def set_stored_input(self, x):
        self._stored_input = x

    def _check_compatibilitiy(self, flow):
        if not isinstance(flow, mdp.CircularOnlineFlow):
            raise TypeError("Flow must be a CircularOnlineFlow type and not %s" % (type(flow)))

    def _set_training_type_from_flow(self, flow):
        for node in flow:
            if hasattr(node, 'training_type') and (node.training_type == 'incremental'):
                self._training_type = 'incremental'
                return
        self._training_type = 'batch'

    def set_training_type(self, training_type):
        if self.training_type != training_type:
            raise mdp.NodeException("Cannot change the training type to %s. It is inferred from "
                                    "the flow and is set to '%s'. " % (training_type, self.training_type))

    def _set_numx_rng(self, rng):
        # set the numx_rng for all the nodes to be the same.
        for node in self._flow:
            if hasattr(node, 'set_numx_rng'):
                node.numx_rng = rng
        self._numx_rng = rng

    def _get_train_seq(self):
        """Return a training sequence containing all training phases.

        Three possible train_seqs depending on the values of self._ignore_input
        and self._flow_iterations.

        1) self._ignore_input = False, self._flow_iterations = 1
            This is functionally similar to the standard OnlineFlowNode.

        2) self._ignore_input = False, self._flow_iterations > 1
            For each data point, the OnlineFlowNode trains 1 loop with the
            data point and 'self._flow_iterations-1' loops with the updating stored input.

        3) self._ignore_input = True, self._flow_iterations > 1
            Input data is ignored, however, for each data point, the flow trains
            'self._flow_iterations' loops with the updating stored input.
        """

        def get_train_function(node, ignore_input):
            def _train(x, *args, **kwargs):
                if ignore_input:
                    if self._stored_input is None:
                        raise mdp.TrainingException("There are no stored inputs to train on. "
                                                    "Set them using 'set_stored_input' method.")
                    x = self._stored_input
                node.train(x, *args, **kwargs)
            return _train

        def get_execute_function(node, ignore_input):
            def _execute(x, *args, **kwargs):
                if ignore_input:
                    if self._stored_input is None:
                        raise mdp.TrainingException("There are no stored inputs to execute. "
                                                    "Set them using 'set_stored_input' method.")
                    x = self._stored_input
                return node.execute(x, *args, **kwargs)
            return _execute

        def get_execute_wrapper(self, fun):
            def _execute_wrapper(x, *args, **kwargs):
                x = fun(x, *args, **kwargs)
                self._stored_input = x.copy()
                return x
            return _execute_wrapper

        train_seq = []
        for flow_iter in xrange(self._flow_iterations):
            for i, node in enumerate(self._flow):
                if (i==0):
                    if node.is_trainable():
                        if (flow_iter == 0):
                            # The first node (trainable) for the first iteration trains on the stored input
                            # if self._ignore_input is True, otherwise the given input.
                            train_seq += [(get_train_function(node, self._ignore_input), node.stop_training,
                                           get_execute_function(node, self._ignore_input))]
                        else:
                            # For the remaining iterations, the first node executes only the stored input.
                            train_seq += [(get_train_function(node, True), node.stop_training,
                                           get_execute_function(node, True))]
                    else:
                        if (flow_iter == 0):
                            # The first node (non-trainable) for the first iteration executes on the stored input
                            # if self._ignore_input is True, otherwise the given input.
                            train_seq += [(lambda x, *args, **kwargs: None, lambda x, *args, **kwargs: None,
                                           get_execute_function(node, self._ignore_input))]
                        else:
                            # For the remaining iterations, the first node executes only the stored input.
                            train_seq += [(lambda x, *args, **kwargs: None, lambda x, *args, **kwargs: None,
                                           get_execute_function(node, True))]
                else:
                    if node.is_trainable():
                        # Rest of the trainable nodes train on the input from the execution phase of their previous nodes.
                        train_seq += [(get_train_function(node, False), node.stop_training,
                                       get_execute_function(node, False))]
                    else:
                        # Rest of the non-trainable nodes execute the input from the execution phase of their previous nodes.
                        train_seq += [(lambda x, *args, **kwargs: None, lambda x, *args, **kwargs: None,
                                       get_execute_function(node, False))]

            # update the stored input after the execute method of the last node is called, at the end of each iteration.
            train_seq[-1] = (train_seq[-1][0], train_seq[-1][1], get_execute_wrapper(self, train_seq[-1][2]))

        return train_seq

