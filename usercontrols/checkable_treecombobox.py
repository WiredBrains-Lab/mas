"""Hierarchical multi-select combo box for MEG channels and report fields."""

from qtpy.QtCore import Qt
from qtpy.QtGui import QStandardItem, QStandardItemModel
from qtpy.QtWidgets import QComboBox, QTreeView


class CheckableTreeComboBox(QComboBox):
    def __init__(self, c=None, parent=None):
        super().__init__(parent)
        self.c = c
        self.channel_dict = {}
        self.setEditable(True)
        self.lineEdit().setReadOnly(True)
        self.view = QTreeView(self)
        self.view.setHeaderHidden(True)
        self.setView(self.view)
        self.model = QStandardItemModel(self)
        self.setModel(self.model)
        self.model.itemChanged.connect(self._item_changed)

    def clear(self):
        self.model.clear()
        self.channel_dict = {}
        self.lineEdit().clear()

    def addItems(self, channel_dict, parent=None):
        if parent is None:
            self.clear()
            self.channel_dict = channel_dict
            parent = self.model.invisibleRootItem()
        self.model.blockSignals(True)
        try:
            for label, value in channel_dict.items():
                checked, children = value
                item = QStandardItem(str(label))
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
                parent.appendRow(item)
                if isinstance(children, dict) and children:
                    self.addItems(children, item)
        finally:
            self.model.blockSignals(False)
        self.updateText()

    def _item_changed(self, item):
        self.model.blockSignals(True)
        try:
            if item.hasChildren():
                self._set_children(item, item.checkState())
            self._update_parent(item.parent())
        finally:
            self.model.blockSignals(False)
        self.channel_dict = self._as_dict(self.model.invisibleRootItem())
        self.updateText()
        signal = getattr(self.c, "sig_view_acq_channels_selection", None)
        if signal is not None:
            signal.emit(self.channel_dict)

    def _set_children(self, item, state):
        for row in range(item.rowCount()):
            child = item.child(row)
            child.setCheckState(state)
            self._set_children(child, state)

    def _update_parent(self, parent):
        if parent is None:
            return
        states = [parent.child(row).checkState() for row in range(parent.rowCount())]
        if all(state == Qt.Checked for state in states):
            parent.setCheckState(Qt.Checked)
        elif all(state == Qt.Unchecked for state in states):
            parent.setCheckState(Qt.Unchecked)
        else:
            parent.setCheckState(Qt.PartiallyChecked)
        self._update_parent(parent.parent())

    def _as_dict(self, parent):
        result = {}
        for row in range(parent.rowCount()):
            item = parent.child(row)
            result[item.text()] = (item.checkState(), self._as_dict(item))
        return result

    def getCheckedItems(self, parent=None):
        parent = parent or self.model.invisibleRootItem()
        labels = []
        for row in range(parent.rowCount()):
            item = parent.child(row)
            if item.hasChildren():
                labels.extend(self.getCheckedItems(item))
            elif item.checkState() == Qt.Checked:
                labels.append(item.text())
        return labels

    def updateText(self):
        self.lineEdit().setText(", ".join(self.getCheckedItems()))
