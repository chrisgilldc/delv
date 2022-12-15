"""PyTorch Hub models

Usage:
        import torch
        model = torch.hub.load('repo','model')
"""

import torch
from pathlib import Path

dependencies = ['torch', 'yaml']

def delv(pretrained=True, channels=3, classes=4, autoshape=True):
        # Download the model file.
        file = Path("dv2aug2.pt")
        if not file.exists():
                url = f"https://github.com/chrisgilldc/delv/releases/download/0.1-alpha/{file}"
                torch.hub.download_url_to_file(url, file)
        # Create Yolov7 with the custom model.
        model = torch.hub.load('WongKinYiu/yolov7','custom',path_or_model='./dv2aug2.pt')
        return model
