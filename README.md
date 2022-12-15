# Delivery Vehicle Detection Model

Custom-trained YOLOV7 object detection model for detecting delivery trucks.

I built this to run camera feeds through to identify when certain delivery trucks are near my home. Currently detets Amazon, FedEx, UPS and USPS.

I use DOODS2 to run detection on video feeds, which doesn't support custom weight loading directly, hence the need for a torch hub friendly wrapper.

When used with DOODS2, also requires manually installing scipy into the container.
