from ikomia import core, dataprocess
from ikomia.utils import pyqtutils, qtconversion
from train_yolo_26_classification.train_yolo_26_classification_process import TrainYolo26ClassificationParam

from PyQt5.QtWidgets import *


class TrainYolo26ClassificationWidget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = TrainYolo26ClassificationParam()
        else:
            self.parameters = param

        self.grid_layout = QGridLayout()

        self.combo_model = pyqtutils.append_combo(
            self.grid_layout, "Model name")
        self.combo_model.addItem("yolo26n-cls")
        self.combo_model.addItem("yolo26s-cls")
        self.combo_model.addItem("yolo26m-cls")
        self.combo_model.addItem("yolo26l-cls")
        self.combo_model.addItem("yolo26x-cls")

        self.combo_model.setCurrentText(self.parameters.cfg["model_name"])

        self.browse_dataset_folder = pyqtutils.append_browse_file(
            self.grid_layout, label="Dataset folder",
            path=self.parameters.cfg["dataset_folder"],
            tooltip="Select folder",
            mode=QFileDialog.Directory
        )

        self.spin_epochs = pyqtutils.append_spin(
            self.grid_layout, "Epochs", self.parameters.cfg["epochs"])

        self.spin_batch = pyqtutils.append_spin(
            self.grid_layout, "Batch size", self.parameters.cfg["batch_size"])

        custom_hyp = bool(self.parameters.cfg["config_file"])
        self.check_hyp = QCheckBox("Custom hyper-parameters")
        self.check_hyp.setChecked(custom_hyp)
        self.grid_layout.addWidget(
            self.check_hyp, self.grid_layout.rowCount(), 0, 1, 2)
        self.check_hyp.stateChanged.connect(self.on_custom_hyp_changed)

        self.label_hyp = QLabel("Hyper-parameters file")
        self.browse_hyp_file = pyqtutils.BrowseFileWidget(path=self.parameters.cfg["config_file"],
                                                          tooltip="Select file",
                                                          mode=QFileDialog.ExistingFile)

        row = self.grid_layout.rowCount()
        self.grid_layout.addWidget(self.label_hyp, row, 0)
        self.grid_layout.addWidget(self.browse_hyp_file, row, 1)

        self.label_hyp.setVisible(custom_hyp)
        self.browse_hyp_file.setVisible(custom_hyp)

        self.browse_out_folder = pyqtutils.append_browse_file(
            self.grid_layout, label="Output folder",
            path=self.parameters.cfg["output_folder"],
            tooltip="Select folder",
            mode=QFileDialog.Directory
        )

        layout_ptr = qtconversion.PyQtToQt(self.grid_layout)
        self.set_layout(layout_ptr)

    def on_custom_hyp_changed(self, int):
        self.label_hyp.setVisible(self.check_hyp.isChecked())
        self.browse_hyp_file.setVisible(self.check_hyp.isChecked())

    def on_apply(self):
        self.parameters.cfg["model_name"] = self.combo_model.currentText()
        self.parameters.cfg["dataset_folder"] = self.browse_dataset_folder.path
        self.parameters.cfg["epochs"] = self.spin_epochs.value()
        self.parameters.cfg["batch_size"] = self.spin_batch.value()
        self.parameters.cfg["output_folder"] = self.browse_out_folder.path
        if self.check_hyp.isChecked():
            self.parameters.cfg["config_file"] = self.browse_hyp_file.path
        self.parameters.update = True

        self.emit_apply(self.parameters)


class TrainYolo26ClassificationWidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        self.name = "train_yolo_26_classification"

    def create(self, param):
        return TrainYolo26ClassificationWidget(param, None)
