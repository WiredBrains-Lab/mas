"""MEG-only FIF signal viewer."""

import numpy as np
import pyqtgraph as pg
from qtpy import QtWidgets

from usercontrols.checkable_treecombobox import CheckableTreeComboBox


class MEGSignalViewer:
    """Display a selectable window of MEG channels from a FIF file."""

    def __init__(self, c, chs_view=None, meg_3d=None, filtering=None):
        self.c = c
        self.chs_view = chs_view
        self.meg_3d = meg_3d
        self.filtering = filtering
        self.fs = 1000
        self.length = 5
        self.num_chs_to_plot = 10
        self.ph_diode = False
        self.pics = False
        self.channel_dict = {}
        self._plots = []
        self._curves = []

        self.widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(self.widget)
        controls = QtWidgets.QHBoxLayout()
        self.channel_combo = CheckableTreeComboBox(c=self.c)
        controls.addWidget(QtWidgets.QLabel("Channels:"))
        controls.addWidget(self.channel_combo, 1)

        back_btn = QtWidgets.QPushButton("Previous")
        back_btn.clicked.connect(lambda: self.c.sig_view_nav.emit("back"))
        controls.addWidget(back_btn)
        next_btn = QtWidgets.QPushButton("Next")
        next_btn.clicked.connect(lambda: self.c.sig_view_nav.emit("forward"))
        controls.addWidget(next_btn)
        layout.addLayout(controls)

        self.graphics = pg.GraphicsLayoutWidget()
        layout.addWidget(self.graphics, 1)
        if self.chs_view is not None:
            self.chs_view.addWidget(self.widget)

    def update_channel_list(self, channel_dict):
        """Replace the selectable MEG channel tree."""
        self.channel_dict = channel_dict or {}
        self.channel_combo.addItems(self.channel_dict)

    def _ensure_plots(self, count, labels):
        if len(self._plots) == count:
            for plot, label in zip(self._plots, labels):
                plot.setLabel("left", label)
            return
        self.graphics.clear()
        self._plots = []
        self._curves = []
        for row, label in enumerate(labels):
            plot = self.graphics.addPlot(row=row, col=0)
            plot.setLabel("left", label)
            plot.showGrid(x=True, y=True, alpha=0.15)
            if row < count - 1:
                plot.hideAxis("bottom")
            else:
                plot.setLabel("bottom", "Time", units="s")
            if self._plots:
                plot.setXLink(self._plots[0])
            self._plots.append(plot)
            self._curves.append(plot.plot(pen=pg.mkPen(width=1)))

    def update_vis_buffer(self, vis_buffer):
        """Render a window emitted by the MEG FIF loader."""
        data = np.asarray(vis_buffer.get("data", []))
        if data.ndim != 2 or data.shape[0] == 0:
            return
        time = np.asarray(vis_buffer.get("t", np.arange(data.shape[1]) / float(self.fs)))
        labels = list(vis_buffer.get("disp_chs", []))
        if len(labels) != data.shape[0]:
            labels = [f"MEG {index + 1}" for index in range(data.shape[0])]
        self._ensure_plots(data.shape[0], labels)
        for curve, values in zip(self._curves, data):
            curve.setData(time, values, skipFiniteCheck=True)

