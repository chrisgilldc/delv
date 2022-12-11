"""PyTorch Hub models

Usage:
	import torch
	model = torch.hub.load('repo','model')
"""

# Learned straight from YoloV7's hubconf.

from pathlib import Path
import torch

from utils import attempt_download
from utils import date_modified
from utils import git_describe
from utils import select_device

dependencies = ['torch', 'yaml']

def create(name, pretrained, channels, classes, autoshape):
	"""Creates a specified model
		Arguments:
		name (str): name of model, i.e. 'yolov7'
		pretrained (bool): load pretrained weights into the model
		channels (int): number of input channels
		classes (int): number of model classes

	Returns:
		pytorch model
	"""
	try:
		# Set the device to use, GPU if available, CPU otherwise.
		device = select_device('0' if torch.cuda.is_available() else 'cpu')
		# Create the baseline Yolov7 model.
		model = torch.hub.load('WongKinYiu/yolov7', 'yolov7', classes=classes, channels=channels)
		# Load the custom weights.
		fname = f'{name}.pt'  # checkpoint filename
		attempt_download(fname, repo="chrisgilldc/delv")  # download if not found locally
		ckpt = torch.load(fname, map_location=torch.device('cpu'))
		msd = model.state_dict()  # model state_dict
		csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
		csd = {k: v for k, v in csd.items() if msd[k].shape == v.shape}  # filter
		model.load_state_dict(csd, strict=False)  # load
		if len(ckpt['model'].names) == classes:
			model.names = ckpt['model'].names  # set class names attribute
		if autoshape:
			model = model.autoshape()  # for file/URI/PIL/cv2/np inputs and NMS
		return model.to(device)

	except Exception as e:
		s = 'Cache maybe be out of date, try force_reload=True.'
		raise Exception(s) from e


def custom(path_or_model='path/to/model.pt', autoshape=True):
	"""custom mode
	Arguments (3 options):
		path_or_model (str): 'path/to/model.pt'
		path_or_model (dict): torch.load('path/to/model.pt')
		path_or_model (nn.Module): torch.load('path/to/model.pt')['model']

	Returns:
		pytorch model
	"""
	model = torch.load(path_or_model) if isinstance(path_or_model, str) else path_or_model  # load checkpoint
	if isinstance(model, dict):
		model = model['ema' if model.get('ema') else 'model']  # load model

	hub_model = Model(model.yaml).to(next(model.parameters()).device)  # create
	hub_model.load_state_dict(model.float().state_dict())  # load state_dict
	hub_model.names = model.names  # class names
	if autoshape:
		hub_model = hub_model.autoshape()  # for file/URI/PIL/cv2/np inputs and NMS
	device = select_device('0' if torch.cuda.is_available() else 'cpu')  # default to GPU if available
	return hub_model.to(device)

def delv(pretrained=True, channels=3, classes=4, autoshape=True):
    return create('dv2aug2', pretrained, channels, classes, autoshape)
