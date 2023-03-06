"""PyTorch Hub models

Usage:
        import torch
        model = torch.hub.load('repo','model')
"""

import torch
from pathlib import Path

dependencies = ['torch', 'yaml']

def _delv(model, pretrained=True, channels=3, classes=4, autoshape=True):
    valid_models = ('dv1-100', 'dv1-200', 'dv1a1-100', 'dv1a1-200', 'dv1a1-500')
    if model not in valid_models:
        raise ValueError("{} is not a valid model name.".format(model))
    # Build the file name.
    file = Path(model).with_suffix('pt')
    # If the file doesn't exist, download it.
    if not file.exists():
        url = f"https://github.com/chrisgilldc/delv/releases/download/0.2-alpha/{file}"
        torch.hub.download_url_to_file(url, file)
    # Load the mode with those
    model = torch.hub.load('WongKinYiu/yolov7', 'custom', path_or_model=str(file))
    return model

def dv1_100(pretrained=True, channels=3, classes=4, autoshape=True):
    return _delv('dv1_100', pretrained, channels, classes, autoshape)

def dv1_200(pretrained=True, channels=3, classes=4, autoshape=True):
    return _delv('dv1_200', pretrained, channels, classes, autoshape)

def dv1a1_100(pretrained=True, channels=3, classes=4, autoshape=True):
    return _delv('dv1a1_100', pretrained, channels, classes, autoshape)

def dv1a1_200(pretrained=True, channels=3, classes=4, autoshape=True):
    return _delv('dv1a1_200', pretrained, channels, classes, autoshape)

def dv1a1_500(pretrained=True, channels=3, classes=4, autoshape=True):
    return _delv('dv1a1_500', pretrained, channels, classes, autoshape)