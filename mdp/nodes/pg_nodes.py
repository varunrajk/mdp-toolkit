

import mdp
import time
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
from multiprocessing import Process, Queue

class PG2DNode(mdp.Node):
    def __init__(self, use_buffer=False, x_range=None, y_range=None, interval=1, input_dim=None, output_dim=None, dtype=None):
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

    def _check_input(self, x):
        super(PG2DNode, self)._check_input(x)
        if self._viewer == None:
            self._viewer = Process(target=self._pg_process)
            self._viewer.start()

    def _pg_process(self):
        self.app = QtGui.QApplication([])
        self._setup_pg_plots()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._pg_data)
        self.timer.start()
        self.app.exec_()

    def _pg_data(self):
        if not self.new_data.empty():
            data = self.new_data.get()
            if data == None:
                QtGui.QApplication.closeAllWindows()
                return
            self._update_pg_plots(data)

    def _setup_pg_plots(self):
        raise mdp.NodeException('Not implemented in the base class!')

    def _update_pg_plots(self, x):
        raise mdp.NodeException('Not implemented in the base class!')

    def close(self):
        self.new_data.put(None)
        self._viewer.join()
        return

    def _plot(self, x):
        if not self._viewer.is_alive(): return
        while(self.new_data.full()):
            if not self._viewer.is_alive():
                self.new_data.get() # empty Queue
                return
            time.sleep(0.0001)
        self.new_data.put(x)

    def _execute(self, x):
        self._tlen+=1
        _flow_dur= time.time()-self._flow_time
        y = x
        if self.use_buffer:
            y = self._buffer(x)
        if self._tlen % int(self._interval) == 0:
            t = time.time()
            self._plot(y)
            _plot_dur = time.time()-t
            if self.interval == -1:
                self._interval = self._interval*(100*_plot_dur/_flow_dur + (self._tlen/self._interval-1)*self._interval)/float(self._tlen)
                self._interval = mdp.numx.clip(self._interval, 1, 50)
        self._flow_time = time.time()
        return x

class PGCurveNode(PG2DNode):
    def __init__(self, title=None, plot_size=(640,480), split_figs=False, use_buffer=False, x_range=None, y_range=None, interval=1):
        super(PGCurveNode, self).__init__(use_buffer, x_range, y_range, interval)
        self.title = title
        self.plot_size = plot_size
        self.split_figs = split_figs

    def _setup_pg_plots(self):
        self._win = pg.GraphicsWindow()
        self._win.resize(*self.plot_size)
        self._win.show()
        if self.title is not None:
            self._win.setWindowTitle(self.title)

        pg.setConfigOptions(antialias=True)

        # Main layout
        self._layout = pg.GraphicsLayout()

        # Set the layout as a central item
        self._win.setCentralItem(self._layout)

        self._curves = [pg.PlotCurveItem(pen=(i,self.input_dim*1.3)) for i in xrange(self.input_dim)]
        self._plotitems = [pg.PlotItem() for _ in xrange(self.input_dim)]
        if self.split_figs:
            num_rows = int(mdp.numx.sqrt(self.input_dim))+1
            for i in xrange(self.input_dim):
                self._plotitems[i].addItem(self._curves[i])
                self._layout.addItem(self._plotitems[i], row=i / num_rows, col=i % num_rows)
                if self.y_range is not None:
                    self._plotitems[i].setYRange(*self.y_range)
                if self.x_range is not None:
                    self._plotitems[i].setXRange(*self.x_range)
        else:
            self._plotitems = self._plotitems[0]
            for i in xrange(self.input_dim):
                self._plotitems.addItem(self._curves[i])
                if self.y_range is None:
                    self._curves[i].setPos(0, (i + 1) * 6)
                    self._plotitems.setYRange(0, (self.input_dim + 1) * 6)
                else:
                    self._curves[i].setPos(0, (i + 1) * (self.y_range[1]-self.y_range[0]))
                    self._plotitems.setYRange(0, (self.input_dim + 1) * (self.y_range[1]-self.y_range[0]))
            if self.x_range is not None:
                self._plotitems.setXRange(*self.x_range)

            self._layout.addItem(self._plotitems)

    def _update_pg_plots(self, x):
        for i in xrange(self.input_dim):
            self._curves[i].setData(x[:, i])


class PGImageNode(PG2DNode):
    def __init__(self, img_xy, title=None, plot_size=None, lut=None, interval=1):
        super(PGImageNode, self).__init__(use_buffer=False, x_range=None, y_range=None, interval=interval)
        self.img_xy = img_xy
        self.title = title
        self.plot_size = plot_size
        self.lut = lut

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

    def _setup_pg_plots(self):
        self._win = pg.GraphicsWindow()
        if self.plot_size is not None:
            self._win.resize(*self.plot_size)
        self._win.show()
        if self.title is not None:
            self._win.setWindowTitle(self.title)

        pg.setConfigOptions(antialias=True)

        # Main layout
        self._plotitem = pg.PlotItem()

        # Set the layout as a central item
        self._win.setCentralItem(self._plotitem)

        self._img = pg.ImageItem(border='w', lut=self._get_pglut(self.lut))

        self._plotitem.addItem(self._img)

        # hide axis and set title
        self._plotitem.hideAxis('left')
        self._plotitem.hideAxis('bottom')
        self._plotitem.hideAxis('top')
        self._plotitem.hideAxis('right')

    def _pre_execution_checks(self, x):
        super(PGImageNode, self)._pre_execution_checks(x)
        if x.shape[0] > 1:
            raise mdp.NodeException("x.ndim should be 1, given %d."%(x.shape[0]))

    def _update_pg_plots(self, x):
        x = x.reshape(self.img_xy[0], self.img_xy[1], x.shape[1]/(self.img_xy[0]*self.img_xy[1]))
        self._img.setImage(x)
