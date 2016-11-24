

import mdp
import time
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
from multiprocessing import Process, Queue

class PG2DNode(mdp.PreserveDimNode):
    """ PG2DNode is a non-blocking fast online 2D plotting node. It uses
    the fast plotting python library called the PyQtGraph (http://www.pyqtgraph.org/).

    PG2DNode is a non-trainable node. It works similar to an oscilloscope;
    place it between two OnlineNodes in an OnlineFlow to visualize the data being
    passed between the nodes. The node can also be used standalone, that is it plots
    as soon as new data is passed through the execute call.

    PG2DNodes 'subclasses' should take care of overwriting these functions
    '_setup_plots', '_update_plots'. Check PGCurveNode and PGImageNode as examples.
    Care must be taken to not overwrite methods '_check_input', '__pg_process', '__pg_data' and
    '__plot' in a subclass unless really required.

    PG2DNodes also work like an identity node returning the input as the output.
    When the plotting windows are manually closed, the node continues to transmit input
    as the output without interfering the flow.

    """

    def __init__(self, use_buffer=False, x_range=None, y_range=None, interval=1, input_dim=None, output_dim=None, dtype=None):
        """
        user_buffer: If the data arrives sample by sample (like in an OnlineFlow), use_buffer can be set to store
        samples in a circular buffer. At each time-step the buffer contents are displayed.

        x_range: Denotes the range of x-axis values to be shown. When the use_buffer is set, this also denotes the size of
        the buffer.

        y_range: y-axis range

        interval: Time steps after which the plots are updated. Here, time step refers to the execute call count.
                 1 - Plots are updated after each execute call
                 10 - Plots are updated after every 10th execute call
                 -1 - Automatically optimize the interval such that the plot updates do not slow the flow's execution time.

         """
        super(PG2DNode, self).__init__(input_dim, output_dim, dtype)
        self.use_buffer = use_buffer
        self._x_range = x_range
        self._y_range = y_range
        self.interval = interval

        self._interval = 1 if self.interval == -1 else self.interval
        self._flow_time = 0
        self._tlen = 0
        self._viewer = None
        if use_buffer:
            if  (x_range is None):
                raise mdp.NodeException("Provide x_range to init buffer size.")
            self._buffer = mdp.nodes.NumxBufferNode(buffer_size=x_range[1])

        self.new_data = Queue(1)

    ### properties

    def get_x_range(self):
        return self._x_range

    def set_x_range(self, x_range):
        if x_range is None:
            return
        if (not isinstance(x_range, tuple)) and ((not isinstance(x_range, list))):
            raise mdp.NodeException("x_range must be a tuple or a list and not %s."%str(type(x_range)))
        if len(x_range) != 2:
            raise mdp.NodeException("x_range must contain 2 elements, given %s."%len(x_range))
        self._x_range = x_range

    x_range = property(get_x_range, set_x_range, doc="x-axis range")

    def get_y_range(self):
        return self._y_range

    def set_y_range(self, y_range):
        if y_range is None:
            return
        if (not isinstance(y_range, tuple)) and ((not isinstance(y_range, list))):
            raise mdp.NodeException("x_range must be a tuple or a list and not %s."%str(type(y_range)))
        if len(y_range) != 2:
            raise mdp.NodeException("x_range must contain 2 elements, given %s."%len(y_range))
        self._y_range = y_range

    y_range = property(get_y_range, set_y_range, doc="y-axis range")

    @staticmethod
    def is_trainable():
        return False

    def _get_supported_dtypes(self):
        return mdp.utils.get_dtypes('AllInteger') + mdp.utils.get_dtypes('Float')

    # -------------------------------------------
    # super private methods.
    # Do not overwrite unless you know what you are doing.

    def _check_input(self, x):
        super(PG2DNode, self)._check_input(x)
        if self._viewer is None:
            self._viewer = Process(target=self.__pg_process)
            self._viewer.start()

    def __pg_process(self):
        # spawned process
        self.app = QtGui.QApplication([])
        self._setup_plots()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.__pg_data)
        self.timer.start()
        self.app.exec_()

    def __pg_data(self):
        # communication function
        if not self.new_data.empty():
            data = self.new_data.get()
            if data is None:
                QtGui.QApplication.closeAllWindows()
                return
            self._update_plots(data)

    def __plot(self, x):
        # plot given data
        if not self._viewer.is_alive(): return
        while(self.new_data.full()):
            if not self._viewer.is_alive():
                self.new_data.get() # empty Queue
                return
            time.sleep(0.0001)
        self.new_data.put(x)
    # -------------------------------------------

    def _setup_plots(self):
        # Setup your plots, layout, etc. Overwrite in the subclass
        # Check PyQtGraph for examples
        # run in shell prompt: python -c "import pyqtgraph.examples; pyqtgraph.examples.run()"
        self._win = pg.GraphicsWindow()
        self._win.show()
        self._win.setWindowTitle("Blank Plot")

    def _update_plots(self, x):
        # Update your individual plotitems. Overwrite in the subclass
        pass

    def _execute(self, x):
        self._tlen+=1
        _flow_dur= time.time()-self._flow_time
        y = x
        if self.use_buffer:
            y = self._buffer(x)
        if self._tlen % int(self._interval) == 0:
            t = time.time()
            self.__plot(y)
            _plot_dur = time.time()-t
            if self.interval == -1:
                self._interval = self._interval*(100*_plot_dur/_flow_dur + (self._tlen/self._interval-1)*self._interval)/float(self._tlen)
                self._interval = mdp.numx.clip(self._interval, 1, 50)
        self._flow_time = time.time()
        return x

    def close(self):
        # Force close all the plots.
        # This is usually not required as the process terminates if the
        # windows are manually closed.
        if not self.new_data.empty():
            self.new_data.get()
        self.new_data.put(None)
        self._viewer.join()
        return


class PGCurveNode(PG2DNode):
    """ PGCurveNode is a PG2DNode that displays the input data as multiple curves.
        Use_buffer needs to be set if the data arrives sample by sample.
    """
    def __init__(self, title=None, plot_size_xy=(640,480), split_figs=False, display_dims=None, use_buffer=False, x_range=None, y_range=None, interval=1,
                 input_dim=None, output_dim=None, dtype=None):
        """
        title: Window title

        plot_size_xy: Plot size (x,y) tuple

        split_figs: When set, each data dimension is plotted in a separate figure, otherwise they are vertically stacked
        in a single plot.

        display_dims: Dimensions that are displayed in the plots. By default all dimensions are displayed. Accepted values:
                      scalar/list/array - displays the provided dimensions

         """
        super(PGCurveNode, self).__init__(use_buffer=use_buffer, x_range=x_range, y_range=y_range, interval=interval,
                                          input_dim=input_dim, output_dim=output_dim, dtype=dtype)
        self._title = title
        self._plot_size_xy= plot_size_xy
        self._split_figs = split_figs

        if display_dims is not None:
            if mdp.numx.isscalar(display_dims):
                display_dims = [display_dims]
            display_dims = mdp.numx.asarray(display_dims)
        self._display_dims = display_dims

    def _setup_plots(self):
        self._win = pg.GraphicsWindow()
        self._win.resize(*self._plot_size_xy)
        self._win.show()
        if self._title is not None:
            self._win.setWindowTitle(self._title)

        pg.setConfigOptions(antialias=True)

        # Main layout
        self._layout = pg.GraphicsLayout()

        # Set the layout as a central item
        self._win.setCentralItem(self._layout)

        if self._display_dims is None:
            self._display_dims = range(0,self.input_dim)

        n_disp_dims = len(self._display_dims)

        self._curves = [pg.PlotCurveItem(pen=(i,n_disp_dims*1.3)) for i in xrange(n_disp_dims)]
        self._plotitems = [pg.PlotItem() for _ in xrange(n_disp_dims)]
        if self._split_figs:
            num_rows = mdp.numx.ceil(mdp.numx.sqrt(n_disp_dims))
            for i in xrange(n_disp_dims):
                self._plotitems[i].addItem(self._curves[i])
                self._layout.addItem(self._plotitems[i], row=i / num_rows, col=i % num_rows)
                if self.y_range is not None:
                    self._plotitems[i].setYRange(*self.y_range)
                if self.x_range is not None:
                    self._plotitems[i].setXRange(*self.x_range)
        else:
            self._plotitems = self._plotitems[0]
            for i in xrange(n_disp_dims):
                self._plotitems.addItem(self._curves[i])
                if self.y_range is None:
                    self._curves[i].setPos(0, (i + 1) * 6)
                    self._plotitems.setYRange(0, (n_disp_dims + 1) * 6)
                else:
                    self._curves[i].setPos(0, (i + 1) * (self.y_range[1]-self.y_range[0]))
                    self._plotitems.setYRange(0, (n_disp_dims + 1) * (self.y_range[1]-self.y_range[0]))
            if self.x_range is not None:
                self._plotitems.setXRange(*self.x_range)

            self._layout.addItem(self._plotitems)

    def _update_plots(self, x):
        x = x[:,self._display_dims]
        for i in xrange(x.shape[1]):
            self._curves[i].setData(x[:, i])


class PGImageNode(PG2DNode):
    """ PGImageNode is a PG2DNode that displays the input data as an Image.
        use_buffer is forcefully unset as it is not required.
    """
    def __init__(self, img_shape, title=None, plot_size_xy=None, display_dims=None, cmap=None, origin='upper', axis_order='row-major', interval=1,
                 input_dim=None, output_dim=None, dtype=None):
        """
        img_shape: 2D or 3D shape of the image. Used to reshape the 2D data.
        
        title: Window title

        plot_size_xy: Plot size (x,y) tuple

        display_dims: Dimensions that are displayed in the plots. By default all dimensions are displayed. Accepted values:
                      scalar/list/array - displays the provided dimensions

        cmap: Color map to use. Supported: Matplotlib color maps - 'jet', 'gray', etc.

        origin: The origin is set at the upper left hand corner and rows (first dimension of the array)
                are displayed horizontally. It can also be set to 'lower' if you want the first
                row in the array to be at the bottom instead of the top.

        axis_order: Axis order can either be 'row-major' or 'col-major'. For 'row-major', image data is expected
                    in the standard (row, col) order. For 'col-major', image data is expected in reversed (col, row) order.

         """
        super(PGImageNode, self).__init__(use_buffer=False, x_range=None, y_range=None, interval=interval,
                                          input_dim=input_dim, output_dim=output_dim, dtype=dtype)
        self.img_shape = img_shape
        self._title = title
        self._plot_size_xy= self.img_shape[:2] if plot_size_xy is None else plot_size_xy

        if display_dims is not None:
            display_dims = mdp.numx.asarray(display_dims)
            if len(display_dims) != mdp.numx.product(img_shape):
                raise mdp.NodeException("Length of 'display_dims' (%d) do not match with the 'img_shape' dims (%d)"
                                        %(len(display_dims),mdp.numx.product(img_shape)))
        else:
            display_dims = mdp.numx.product(img_shape)

        self._display_dims = display_dims

        self.cmap = cmap
        if origin not in ['upper', 'lower']:
            raise mdp.NodeException("'origin' must either be 'upper' or 'lower' and not %s"%str(origin))
        self.origin = origin

        if axis_order not in ['row-major', 'col-major']:
            raise mdp.NodeException("'axis_order' must either be 'row-major' or 'col-major' and not %s"%str(axis_order))
        self.axis_order = axis_order

        # Force unset use_buffer
        self.use_buffer = False

    @staticmethod
    def _get_pglut(lutname=None):
        pg_lut = None
        if lutname is not None:
            from matplotlib.cm import get_cmap
            from matplotlib.colors import ColorConverter
            lut = []
            cmap = get_cmap(lutname, 1000)
            for i in range(1000):
                r, g, b = ColorConverter().to_rgb(cmap(i))
                lut.append([r * 255, g * 255, b * 255])
            pg_lut = mdp.numx.array(lut, dtype=mdp.numx.uint8)
            pg_lut[0, :] = [0, 0, 0]
        return pg_lut

    def _setup_plots(self):
        self._win = pg.GraphicsWindow()
        self._win.resize(*self._plot_size_xy)
        self._win.show()
        if self._title is not None:
            self._win.setWindowTitle(self._title)

        pg.setConfigOptions(antialias=True)
        pg.setConfigOptions(imageAxisOrder=self.axis_order)

        # Main layout
        self._plotitem = pg.PlotItem()

        # Set the layout as a central item
        self._win.setCentralItem(self._plotitem)

        self._img = pg.ImageItem(border='w', lut=self._get_pglut(self.cmap))

        self._plotitem.addItem(self._img)

        # hide axis and set title
        self._plotitem.hideAxis('left')
        self._plotitem.hideAxis('bottom')
        self._plotitem.hideAxis('top')
        self._plotitem.hideAxis('right')

    def _update_plots(self, x):
        x = x[:,self._display_dims].reshape(*self.img_shape)
        if self.origin == "upper":
            if (self.axis_order == 'row-major'):
                x = x[::-1]
            elif (self.axis_order == 'col-major'):
                x = x[:,::-1]
        self._img.setImage(x)

    def _execute(self, x):
        for i in xrange(x.shape[0]):
            super(PGImageNode, self)._execute(x[i:i+1])
        return x