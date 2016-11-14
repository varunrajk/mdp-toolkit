
import mdp
import warnings as _warnings
import webbrowser, os, tempfile

class HSFANode(mdp.Node):
    def __init__(self, in_channels_xy, field_channels_xy, field_spacing_xy, n_features, in_channel_dim=1, n_training_fields=None, field_dstr='uniform', output_2d=True):
        super(HSFANode, self).__init__(input_dim=None, output_dim=None, dtype=None)

        self.field_channels_xy = field_channels_xy
        self.in_channels_xy = in_channels_xy
        self.in_channel_dim = in_channel_dim
        self.field_dstr = field_dstr
        self.output_2d = output_2d

        self.n_layers = len(field_channels_xy)
        self.field_spacing_xy = self._xy_to_layersxy(field_spacing_xy, 'field_spacing_xy')
        self.n_features = self._x_to_layersx(n_features, 'n_features')

        self._default_net, self._rec_n_training_fields, self._input_shape, self._output_shape = self._init_default_net()

        if n_training_fields is None:
            # Random sampling disabled
            self._training_flow = self._execution_flow = mdp.Flow(self._default_net)
            self.n_training_fields = n_training_fields
        else:
            # Random sampling enabled
            if n_training_fields == -1:
                # Automatically set n_training_fields to double the minimum recommended.
                # n_training_fields = [2*i for i in self._rec_n_training_fields]
                n_training_fields = self._rec_n_training_fields
            self.n_training_fields = self._x_to_layersx(n_training_fields, 'n_training_fields')
            self._check_n_training_fields()
            self._training_flow = self._init_random_sampling_net()  # not an mdp.Flow
            self._execution_flow = mdp.Flow(self._default_net)

        self._input_dim = mdp.numx.prod(self.input_shape)
        if self.output_2d:
            self._output_dim = mdp.numx.prod(self.output_shape)
        else:
            self._output_dim = self.output_shape

    def _init_default_net(self):
        if self.field_channels_xy[0] == (-1,-1):
            sb = mdp.hinet.Rectangular2dSwitchboard(self.in_channels_xy, self.in_channels_xy, (1,1), self.in_channel_dim, True)
        else:
            sb = mdp.hinet.Rectangular2dSwitchboard(self.in_channels_xy, self.field_channels_xy[0], self.field_spacing_xy[0], self.in_channel_dim,True)
        ln = self._hinet_node(sb.out_channel_dim, self.n_features[0])
        cl = mdp.hinet.CloneLayer(ln, n_nodes=sb.output_channels)
        n_training_fields = [sb.output_channels]
        node = mdp.hinet.FlowNode(mdp.Flow([sb, cl]))
        layers = [node]
        for i in xrange(1, self.n_layers):
            if self.field_channels_xy[i] == (-1, -1):
                sb = mdp.hinet.Rectangular2dSwitchboard(sb.out_channels_xy, sb.out_channels_xy, (1,1), ln.output_dim, True)
            else:
                sb = mdp.hinet.Rectangular2dSwitchboard(sb.out_channels_xy, self.field_channels_xy[i], self.field_spacing_xy[i], ln.output_dim, True)
            ln = self._hinet_node(sb.out_channel_dim, self.n_features[i])
            cl = mdp.hinet.CloneLayer(ln, n_nodes=sb.output_channels)
            n_training_fields.append(sb.output_channels)
            node = mdp.hinet.FlowNode(mdp.Flow([sb, cl]))
            layers.append(node)
        input_shape = (self.in_channels_xy[0], self.in_channels_xy[1], self.in_channel_dim)
        output_shape = (sb.out_channels_xy[0], sb.out_channels_xy[1], ln.output_dim)
        return layers, n_training_fields, input_shape, output_shape,

    def _init_random_sampling_net(self):
        if not hasattr(self, '_default_net'):
            raise mdp.NodeException("'_init_random_sampling_net' must be called after '_init_default_net'")
        if self.field_channels_xy[0] == (-1,-1):
            sb = mdp.hinet.RandomChannelSwitchboard(self.in_channels_xy, self.in_channels_xy, in_channel_dim=self.in_channel_dim, out_channels=1, field_dstr=self.field_dstr)
        else:
            sb = mdp.hinet.RandomChannelSwitchboard(self.in_channels_xy, self.field_channels_xy[0], in_channel_dim=self.in_channel_dim, out_channels=self.n_training_fields[0], field_dstr=self.field_dstr)
        cl = mdp.hinet.CloneLayer(self._default_net[0].flow[1].node, n_nodes=sb.output_channels)
        node = mdp.hinet.FlowNode(mdp.Flow([sb, cl]))
        layers = [node]
        for i in xrange(1, self.n_layers):
            if self.field_channels_xy[i] == (-1, -1):
                sb = mdp.hinet.RandomChannelSwitchboard(self._default_net[i-1].flow[0].out_channels_xy, self._default_net[i-1].flow[0].out_channels_xy,
                                                in_channel_dim=self._default_net[i-1].flow[1].node.output_dim, out_channels=1, field_dstr=self.field_dstr)
            else:
                sb = mdp.hinet.RandomChannelSwitchboard(self._default_net[i - 1].flow[0].out_channels_xy,
                                                    self.field_channels_xy[i],
                                                    in_channel_dim=self._default_net[i - 1].flow[1].node.output_dim,
                                                    out_channels=self.n_training_fields[i], field_dstr=self.field_dstr)
            cl = mdp.hinet.CloneLayer(self._default_net[i].flow[1].node, n_nodes=sb.output_channels)
            node = mdp.hinet.FlowNode(mdp.Flow([sb, cl]))
            layers.append(node)
        return layers

    def _set_args_from_net(self):
        def get_output_dims_from_flownode(flownode):
            outdims = []
            for node in flownode.flow:
                if node.input_dim == node.output_dim:
                    outdims.append(-1)
                else:
                    outdims.append(node.output_dim)
            return tuple(outdims)
        sb, cln = self._execution_flow[0].flow[:2]
        self.in_channel_xy = sb.in_channels_xy
        self.in_channel_dim = sb.in_channel_dim
        self.field_channels_xy = [sb.field_channels_xy]
        self.field_spacing_xy = [sb.field_spacing_xy]
        self.n_features = [get_output_dims_from_flownode(cln.node)]
        self.n_layers = len(self._execution_flow)
        n_training_fields = [self._training_flow[0].flow[0].output_channels]
        for i in xrange(1,self.n_layers):
            sb,cln = self._execution_flow[i].flow[:2]
            self.field_channels_xy.append(sb.field_channels_xy)
            self.field_spacing_xy.append(sb.field_spacing_xy)
            self.n_features.append(get_output_dims_from_flownode(cln.node))
            n_training_fields.append(self._training_flow[i].flow[0].output_channels)
        if self.n_training_fields is not None:
            self.n_training_fields = n_training_fields
            self.field_dstr = self._training_flow[0].flow[0].field_dstr

    def _check_n_training_fields(self):
        for i in xrange(len(self._rec_n_training_fields)):
            if self.n_training_fields[i] < self._rec_n_training_fields[i]:
                _warnings.warn("\nRecommended n_training_field for layer %d is %d, %d given."%(i, self._rec_n_training_fields[i], self.n_training_fields[i]))

    def _x_to_layersx(self, x, xname):
        """
        if scalar: convert scalar to a list of scalars for all layers
        if tuple: convert tuple to a list of tuples for all layers
        if list: check if the list has n_layers elements.
        """
        if mdp.numx.isscalar(x) or isinstance(x, tuple) or (x is None):
            x = [x] * self.n_layers
        if not isinstance(x, list):
            err_str = ("'%s' must be a list"
                       ", and not %s" % (xname, type(x)))
            raise mdp.NodeException(err_str)
        if not (len(x) == self.n_layers):
            raise mdp.NodeException("%d '%s' needed, "
                                    "%d given." % (self.n_layers, xname, len(x)))
        return x

    def _xy_to_layersxy(self, xy, xyname):
        """
        if scalar: convert scalar to a list of 2tuple for all layers
        if tuple: check if 2tuple and convert to a list of 2tuples for all layers
        if list: check if the list has n_layers elements and each element is a 2tuple
        """
        if mdp.numx.isscalar(xy) or (xy is None):
            xy = [(xy,xy)]* self.n_layers
        elif isinstance(xy, tuple):
            xy = [xy]* self.n_layers
        if not isinstance(xy, list):
            err_str = ("'%s' must be a list of 2-tuples"
                       ", and not %s" %(xyname, type(xy)))
            raise mdp.NodeException(err_str)
        if not (len(xy) == self.n_layers):
            raise mdp.NodeException("%d '%s' elements needed, "
                                    "%d given."%(self.n_layers, xyname, len(xy)))
        # check that all elements are 2-tuples
        for i, tup in enumerate(xy):
            if not (len(tup) == 2):
                err = ("Element number %d in the '%s'"
                       " list is not a 2tuple." %(i, xyname))
                raise mdp.NodeException(err)

        return xy

    def _hinet_node(self, input_dim, n_features):
        if mdp.numx.isscalar(n_features): n_features = [n_features]
        n_features = list(n_features)
        flow = []
        if (n_features[0] == -1): n_features[0] = input_dim
        if (n_features[0] > input_dim):
            _warnings.warn("\nNumber of output features of SFA1 node (%d) is greater than its input_dim (%d). Setting them equal."%(n_features[0], input_dim))
            sfa1_node = mdp.nodes.SFANode(input_dim=input_dim,output_dim=input_dim)
        else:
            sfa1_node = mdp.nodes.SFANode(input_dim=input_dim,output_dim=n_features[0])
        flow.append(sfa1_node)

        if len(n_features) > 1:
            exp_node = mdp.nodes.QuadraticExpansionNode(input_dim=sfa1_node.output_dim)
            if (n_features[1] == -1): n_features[1] = exp_node.output_dim
            if (n_features[1] > exp_node.output_dim):
                _warnings.warn("\nNumber of output features of SFA2 node (%d) is greater than its input_dim (%d). Setting them equal."%(n_features[1], exp_node.output_dim))
                sfa2_node = mdp.nodes.SFANode(input_dim=exp_node.output_dim,output_dim=exp_node.output_dim)
            else:
                sfa2_node = mdp.nodes.SFANode(input_dim=exp_node.output_dim,output_dim=n_features[1])
            flow.extend([exp_node, sfa2_node])
        node = mdp.hinet.FlowNode(mdp.Flow(flow))
        return node

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def output_shape(self):
        return self._output_shape


    def is_trainable(self):
        return True


    def is_invertible(self):
        return False


    def _get_train_seq(self):
        """Return a training sequence containing all training phases."""

        def get_train_function(_i_node, _node):
            # This internal function is needed to channel the data through
            # the nodes in front of the current nodes.
            # using nested scopes here instead of default args, see pep-0227
            def _train(x, *args, **kwargs):
                if _i_node > 0:
                    x = self._execution_flow.execute(x, nodenr=_i_node - 1)
                _node.train(x + 0.0001 * mdp.numx.randn(*x.shape), *args, **kwargs)
            return _train

        train_seq = []
        for i_node, node in enumerate(self._training_flow):
            if node.is_trainable():
                remaining_len = len(node._get_train_seq())
                train_seq += ([(get_train_function(i_node, node),
                                node.stop_training)] * remaining_len)
        return train_seq

    def _execute(self, x, n=None):
        return self._execution_flow.execute(x, nodenr=n)

    ###### string representation

    def __str__(self):
        netstr = ""
        netstr += type(self).__name__
        if self.n_training_fields is not None:
            netstr += " with random sampling"
        netstr += "\ninput_dim=%s, " % str(self.input_dim)
        netstr += "output_dim=%s, " % str(self.output_dim)
        if self.dtype is None:
            netstr += 'dtype=None, '
        else:
            netstr += "dtype='%s', " % self.dtype.name

        netstr += "input_shape=%s, " % str(self.input_shape)
        netstr += "output_shape=%s" % str(self.output_shape)

        hinet_dims = "\nhinet_dims %s = [ " % str(self._execution_flow[0]._flow[1].node._flow)
        layer_input_shapes = "\nlayer_input_shapes=[ "
        for i in xrange(self.n_layers):
            sb, cln = self._execution_flow[i]
            layer_input_shapes += "%s, " % str((sb.in_channels_xy[0], sb.in_channels_xy[1], sb.in_channel_dim))
            hinet_dims += "%s, " % str([(node.input_dim, node.output_dim) for node in cln.node])
        layer_input_shapes += "]"
        hinet_dims += "]"
        netstr += hinet_dims + layer_input_shapes
        netstr += "\nnum_training_fields=%s, recommended_min=%s"%(str(self.n_training_fields), str(self._rec_n_training_fields))
        return netstr

    def __repr__(self):
        name = type(self).__name__
        in_channels_xy = "in_channels_xy=%s"%str(self.in_channels_xy)
        field_channels_xy = "field_channels_xy=%s"%str(self.field_channels_xy)
        field_spacing_xy = "field_spacing_xy=%s"%str(self.field_spacing_xy)
        n_features = "\nn_features=%s"%str(self.n_features)
        in_channel_dim = "in_channel_dim=%s"%str(self.in_channel_dim)
        n_training_fields= "n_training_fields=%s"%str(self.n_training_fields)
        field_dstr = "field_dstr='%s'"%str(self.field_dstr)
        output_2d = "output_2d=%s"%str(self.output_2d)
        args = ', '.join((in_channels_xy, field_channels_xy, field_spacing_xy, n_features, in_channel_dim, n_training_fields, field_dstr, output_2d))
        return name + '(' + args + ')'

    ###### html representation

    def _css(self):
        """Return the class CSS."""
        css_filename = os.path.join(os.path.split(__file__)[0],"basic.css")
        with open(css_filename, 'r') as css_file:
            css = css_file.read()
        return css

    def show_flow(self, filename=None, show_size=False, browser_open=True):
        """Write hsfa flow into a HTML file, open it in the browser and
        return the file name.

        filename -- Filename for the HTML file to be created. If None
                    a temporary file is created.
        title -- Title for the HTML file.
        show_size -- Show the approximate memory footprint of all nodes.
        """
        if filename is None:
            (fd, filename) = tempfile.mkstemp(suffix=".html", prefix="MDP_")
            html_file = os.fdopen(fd, 'w')
        else:
            filename = os.path.abspath(filename)
            html_file = open(filename, 'w')
        title = type(self).__name__
        html_file.write('<html>\n<head>\n<title>%s</title>\n' % title)
        html_file.write('<style type="text/css" media="screen">')
        html_file.write(mdp.utils.basic_css() + mdp.hinet.HiNetHTMLVisitor.hinet_css())
        html_file.write('</style>\n</head>\n<body>\n')
        html_file.write('<h3>%s</h3>\n' % title)
        explanation = '(data flows from top to bottom)'
        html_file.write('<par class="explanation">%s</par>\n' % explanation)
        html_file.write('<br><br><br>\n')
        if self.n_training_fields is not None:
            html_file.write('<table class=\"flow1\"><tr><td>\n')
            html_file.write('<h4>Training Flow</h4>')
            for i in xrange(len(self._training_flow)):
                converter = mdp.hinet.HiNetHTMLVisitor(html_file, show_size=show_size)
                if i > 0:
                    html_file.write('<h5>Input from execution layer-%d</h5>'%(i-1))
                converter.convert_flow(flow=self._training_flow[i])
            html_file.write('</tr></td></table>\n')
            html_file.write('<table class=\"flow1\"><tr><td>\n')
            html_file.write('<h4>Execution Flow</h4>')
            converter = mdp.hinet.HiNetHTMLVisitor(html_file, show_size=show_size)
            converter.convert_flow(flow=self._execution_flow)
        else:
            converter = mdp.hinet.HiNetHTMLVisitor(html_file, show_size=show_size)
            converter.convert_flow(flow=self._execution_flow)
        html_file.write('</tr></td></table>\n')
        html_file.write('</body>\n</html>')
        html_file.close()
        if browser_open:
            webbrowser.open(filename)
        return filename

    ###### private container methods

    def __len__(self):
        return self.n_layers

    def __getitem__(self, key):
        return self._execution_flow[key]

    def __setitem__(self, key, value):
        execution_flow_copy = list(self._execution_flow)
        execution_flow_copy[key] = value
        # check dimension consistency
        self._execution_flow._check_nodes_consistency(execution_flow_copy)
        # if no exception was raised, accept the new sequence
        self._execution_flow = execution_flow_copy
        if self.n_training_fields is not None:
            self._training_flow = self._init_random_sampling_net()
        self._set_args_from_net()

    def __iter__(self):
        return self._execution_flow.__iter__()


    def __contains__(self, item):
        return self._execution_flow.__contains__(item)


    ###### public container methods

    def append(self, x):
        """apend layer to the node end"""
        self[len(self):len(self)] = [x]

    def extend(self, x):
        """extend layers by appending
        elements from the iterable"""
        if not isinstance(x, mdp.Flow):
            err_str = ('can only concatenate flow'
                       ' (not \'%s\') to flow' % (type(x).__name__))
            raise TypeError(err_str)
        self[len(self):len(self)] = x

    def insert(self, i, x):
        """insert layer before index"""
        self[i:i] = [x]


class HSFAPoolNode(HSFANode):
    def __init__(self, in_channels_xy, field_channels_xy, field_spacing_xy, n_features, pool_channels_xy, 
                 in_channel_dim=1, pool_spacing_xy=None, pool_mode='max', n_training_fields=None, field_dstr='uniform', output_2d=True):
        self.n_layers = len(field_channels_xy)
        self.pool_mode = pool_mode
        self.pool_channels_xy = self._xy_to_layersxy(pool_channels_xy, 'pool_channels_xy')
        self.pool_spacing_xy = self._xy_to_layersxy(pool_spacing_xy, 'pool_spacing_xy')
        super(HSFAPoolNode, self).__init__(in_channels_xy, field_channels_xy, field_spacing_xy, n_features, in_channel_dim=in_channel_dim,
                     n_training_fields=n_training_fields, field_dstr=field_dstr, output_2d=output_2d)

    def _init_default_net(self):
        if self.field_channels_xy[0] == (-1,-1):
            sb = mdp.hinet.Rectangular2dSwitchboard(self.in_channels_xy, self.in_channels_xy, (1,1), self.in_channel_dim, True)
        else:
            sb = mdp.hinet.Rectangular2dSwitchboard(self.in_channels_xy, self.field_channels_xy[0], self.field_spacing_xy[0], self.in_channel_dim,True)
        ln = self._hinet_node(sb.out_channel_dim, self.n_features[0])
        cl = mdp.hinet.CloneLayer(ln, n_nodes=sb.output_channels)
        rec_n_training_fields = [sb.output_channels]
        pool = mdp.hinet.Pool2D(in_channels_xy=sb.out_channels_xy, field_channels_xy=self.pool_channels_xy[0], field_spacing_xy=self.pool_spacing_xy[0], in_channel_dim=ln.output_dim, mode=self.pool_mode)
        node = mdp.hinet.FlowNode(mdp.Flow([sb, cl, pool]))
        layers = [node]
        for i in xrange(1, self.n_layers):
            if self.field_channels_xy[i] == (-1, -1):
                sb = mdp.hinet.Rectangular2dSwitchboard(pool.out_channels_xy, pool.out_channels_xy, (1,1), ln.output_dim, True)
            else:
                sb = mdp.hinet.Rectangular2dSwitchboard(pool.out_channels_xy, self.field_channels_xy[i], self.field_spacing_xy[i], ln.output_dim, True)
            ln = self._hinet_node(sb.out_channel_dim, self.n_features[i])
            cl = mdp.hinet.CloneLayer(ln, n_nodes=sb.output_channels)
            rec_n_training_fields.append(sb.output_channels)
            pool = mdp.hinet.Pool2D(in_channels_xy=sb.out_channels_xy, field_channels_xy=self.pool_channels_xy[i], field_spacing_xy=self.pool_spacing_xy[i],
                              in_channel_dim=ln.output_dim, mode=self.pool_mode)
            node = mdp.hinet.FlowNode(mdp.Flow([sb, cl, pool]))
            layers.append(node)
        input_shape = (self.in_channels_xy[0], self.in_channels_xy[1], self.in_channel_dim)
        output_shape = (pool.out_channels_xy[0], pool.out_channels_xy[1], ln.output_dim)
        return layers, rec_n_training_fields, input_shape, output_shape

    def _init_random_sampling_net(self):
        if self.field_channels_xy[0] == (-1,-1):
            sb = mdp.hinet.RandomChannelSwitchboard(self.in_channels_xy, self.in_channels_xy, in_channel_dim=self.in_channel_dim, out_channels=1, field_dstr=self.field_dstr)
        else:
            sb = mdp.hinet.RandomChannelSwitchboard(self.in_channels_xy, self.field_channels_xy[0], in_channel_dim=self.in_channel_dim, out_channels=self.n_training_fields[0], field_dstr=self.field_dstr)
        cl = mdp.hinet.CloneLayer(self._default_net[0].flow[1].node, n_nodes=sb.output_channels)
        node = mdp.hinet.FlowNode(mdp.Flow([sb, cl]))
        layers = [node]
        for i in xrange(1, self.n_layers):
            if self.field_channels_xy[i] == (-1, -1):
                sb = mdp.hinet.RandomChannelSwitchboard(self._default_net[i-1].flow[2].out_channels_xy, self._default_net[i-1].flow[2].out_channels_xy,
                                                in_channel_dim=self._default_net[i-1].flow[1].node.output_dim, out_channels=1, field_dstr=self.field_dstr)
            else:
                sb = mdp.hinet.RandomChannelSwitchboard(self._default_net[i-1].flow[2].out_channels_xy, self.field_channels_xy[i],
                                                    in_channel_dim=self._default_net[i-1].flow[1].node.output_dim, out_channels=self.n_training_fields[i], field_dstr=self.field_dstr)
            cl = mdp.hinet.CloneLayer(self._default_net[i].flow[1].node, n_nodes=sb.output_channels)
            node = mdp.hinet.FlowNode(mdp.Flow([sb, cl]))
            layers.append(node)
        return layers

    def _set_args_from_net(self):
        super(HSFAPoolNode, self)._set_args_from_net()
        self.pool_channels_xy = []
        self.pool_spacing_xy = []
        for i in xrange(1,self.n_layers):
            pool = self._execution_flow[i].flow[2]
            self.pool_channels_xy.append(pool.field_channels_xy)
            self.pool_spacing_xy.append(pool.field_spacing_xy)
            self.pool_mode = pool.mode

    def __str__(self):
        netstr = ""
        netstr += type(self).__name__
        if self.n_training_fields is not None:
            netstr += " with random sampling"
        netstr += "\ninput_dim=%s, " % str(self.input_dim)
        netstr += "output_dim=%s, " % str(self.output_dim)
        if self.dtype is None:
            netstr += 'dtype=None, '
        else:
            netstr += "dtype='%s', " % self.dtype.name

        netstr += "input_shape=%s, " % str(self.input_shape)
        netstr += "output_shape=%s" % str(self.output_shape)

        hinet_dims = "\nhinet_dims %s = [ " % str(self._execution_flow[0]._flow[1].node._flow)
        layer_input_shapes = "\nlayer_input_shapes=[ "
        pool_input_shapes = "\npool_input_shapes=[ "
        for i in xrange(self.n_layers):
            sb, cln, pool = self._execution_flow[i]
            layer_input_shapes += "%s, " % str((sb.in_channels_xy[0], sb.in_channels_xy[1], sb.in_channel_dim))
            hinet_dims += "%s, " % str([(node.input_dim, node.output_dim) for node in cln.node])
            pool_input_shapes += "%s, " % str((pool.in_channels_xy[0], pool.in_channels_xy[1], pool.in_channel_dim))
        layer_input_shapes += "]"
        hinet_dims += "]"
        pool_input_shapes += "]"
        netstr += hinet_dims + layer_input_shapes + pool_input_shapes
        netstr += "\nnum_training_fields=%s, recommended_min=%s"%(str(self.n_training_fields), str(self._rec_n_training_fields))
        return netstr

    def __repr__(self):
        name = type(self).__name__
        in_channels_xy = "in_channels_xy=%s"%str(self.in_channels_xy)
        field_channels_xy = "field_channels_xy=%s"%str(self.field_channels_xy)
        field_spacing_xy = "field_spacing_xy=%s"%str(self.field_spacing_xy)
        n_features = "\nn_features=%s"%str(self.n_features)
        pool_channels_xy = "pool_channels_xy=%s"%str(self.pool_channels_xy)
        in_channel_dim = "in_channel_dim=%s"%str(self.in_channel_dim)
        pool_spacing_xy = "\npool_spacing_xy=%s"%str(self.pool_spacing_xy)
        pool_mode = "pool_mode='%s'"%str(self.pool_mode)
        n_training_fields= "n_training_fields=%s"%str(self.n_training_fields)
        field_dstr = "field_dstr='%s'"%str(self.field_dstr)
        output_2d = "output_2d=%s"%str(self.output_2d)
        args = ', '.join((in_channels_xy, field_channels_xy, field_spacing_xy, n_features, pool_channels_xy,
                          in_channel_dim, pool_spacing_xy, pool_mode, n_training_fields, field_dstr, output_2d))
        return name + '(' + args + ')'

