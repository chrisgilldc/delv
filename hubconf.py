"""PyTorch Hub models

Usage:
        import torch
        model = torch.hub.load('repo','model')
"""

import torch
from pathlib import Path
import zipfile
import sys

dependencies = ['torch', 'yaml']

def _delv(model, pretrained=True, channels=3, classes=4, autoshape=True):
    valid_models = ('dv1_100', 'dv1_200', 'dv1a1_100', 'dv1a1_200', 'dv1a1_500')
    if model not in valid_models:
        raise ValueError("{} is not a valid model name.".format(model))
    # Build the file name.
    dv_file = Path(model).with_suffix('.pt')
    # If the file doesn't exist, download it.
    if not dv_file.exists():
        dv_url = f"https://github.com/chrisgilldc/delv/releases/download/0.3-alpha/{dv_file}"
        torch.hub.download_url_to_file(dv_url, dv_file)
    # Download the yolov7 archive.
    yolov7_url = "https://github.com/WongKinYiu/yolov7/archive/refs/heads/main.zip"
    torch.hub.download_url_to_file(yolov7_url, "yolov7.zip")
    zipfile.ZipFile("yolov7.zip").extractall()
    # Extend the system search path.
    sys.path.append(str(Path.cwd() / "yolov7-main"))
    print("New system path: {}".format(sys.path))
    # Now we can import the Yolo model.
    from models.yolo import Model

    # Load the pickled model
    dv_model = torch.load(dv_file, map_location=torch.device('cpu')).model
    return dv_model


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