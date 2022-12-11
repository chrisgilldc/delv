####
# Delivery Vehicles - Hubconf)
####

# Model to identify delivery vehicles!

dependencies = ['torch']

import torch
import pickle

def delivery_vehicles(verbose=True,device=None):
	""" Creates a Delivery Vehicle model
	This is a YOLOv5 based model to detect various types of delivery vehicles - UPS, US Postal Service, etc
	"""

	from utils.general import intersect_dicts, set_logging
	from utils.torch_utils import select_device

	# 3 channel, RGB images
	channels = 3
	# Classes for the model. Currently 4 - Amazon, Fedex, UPS and USPS
	classes = 4

	# Try to create the model
	try:
		# Do we use a CUDA device?
		device = select_device(('0' if torch.cuda.is_available() else 'cpu') if device is None else device)
		# Set up the base yolov5 model
		model = torch.hub.load('WongKinYiu/yolov7','yolov7',classes=classes,device=device)
		# Fetch the custom weights
		checkpoint_url = "https://www.jumpbeacon.net/dv_model/dv_latest.pt"
		ckpt = torch.hub.load_state_dict_from_url(checkpoint_url,map_location=device)
		# Checkpoint state_dict as FP32
		csd = ckpt['model'].float().state_dict()
		# Merge the new weights with the existing model state_dict
		csd = intersect_dicts(csd, model.state_dict(), exclude=['anchors'])
		# Load in the merged state
		model.load_state_dict(csd, strict=False)
		if len(ckpt['model'].names) == classes:
			model.names = ckpt['model'].names # Set the class names attribute
		return model.to(device)

	except Exception as e:
		raise e
