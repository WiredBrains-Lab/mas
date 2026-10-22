"""Flat multi-select combo box used by the anatomical label controls."""

from qtpy.QtCore import QEvent, Qt
from qtpy.QtGui import QStandardItemModel
from qtpy.QtWidgets import QComboBox, QStyledItemDelegate


class CheckableItemDelegate(QStyledItemDelegate):
    def editorEvent(self, event, model, option, index):
        if event is not None and event.type() == QEvent.MouseButtonRelease:
            if index.isValid() and model is not None:
                item = model.itemFromIndex(index)
                if item is not None and (item.flags() & Qt.ItemIsUserCheckable):
                    new_state = Qt.Unchecked if item.checkState() == Qt.Checked else Qt.Checked
                    item.setCheckState(new_state)
                    return True
        return super().editorEvent(event, model, option, index)


class CheckableComboBox(QComboBox):
    def __init__(self, parent=None, width=140):
        super().__init__(parent)
        self.setModel(QStandardItemModel(self))
        self.setMinimumWidth(width)
        self._delegate = CheckableItemDelegate(self)
        self.setItemDelegate(self._delegate)

    def handleItemPressed(self, index):
        item = self.model().itemFromIndex(index)
        if item is not None:
            state = Qt.Unchecked if item.checkState() == Qt.Checked else Qt.Checked
            item.setCheckState(state)

    def addItem(self, text, userData=None):
        super().addItem(text, userData)
        item = self.model().item(self.count() - 1)
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Unchecked)
