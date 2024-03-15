from .mobilenet import MobileNet
from .resnet50 import Resnet50
from .resnet18 import Resnet18
from .densenet import DenseNet
from .efficientnet import EfficientNet

MODELS = dict(resnet50=Resnet50,
              resnet18=Resnet18,
              mobilenet=MobileNet,
              densenet=DenseNet,
              efficientnet=EfficientNet

              # other models

              )


def load_model(model_name, **kwargs):
    """Get models"""
    return MODELS[model_name](**kwargs).get_model()

