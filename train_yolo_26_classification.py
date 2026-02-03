"""
Main Ikomia plugin module.
Ikomia Studio and Ikomia API use it to load algorithms dynamically.
"""
from ikomia import dataprocess
from train_yolo_26_classification.train_yolo_26_classification_process import TrainYolo26ClassificationFactory
from train_yolo_26_classification.train_yolo_26_classification_process import TrainYolo26ClassificationParamFactory


class IkomiaPlugin(dataprocess.CPluginProcessInterface):
    """
    Interface class to integrate the process with Ikomia application.
    Inherits PyDataProcess.CPluginProcessInterface from Ikomia API.
    """
    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def get_process_factory(self):
        """Instantiate process object."""
        return TrainYolo26ClassificationFactory()

    def get_widget_factory(self):
        """Instantiate associated widget object."""
        from train_yolo_26_classification.train_yolo_26_classification_widget import TrainYolo26ClassificationWidgetFactory
        return TrainYolo26ClassificationWidgetFactory()

    def get_param_factory(self):
        """Instantiate algorithm parameters object."""
        return TrainYolo26ClassificationParamFactory()
