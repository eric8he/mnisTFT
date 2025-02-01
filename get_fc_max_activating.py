# Script to generate 28x28 binarized images that maximally activate each neuron in the FC layer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

# ---------------------
# Example Binarize Transform
# ---------------------
class BinarizeTransform:
    def __init__(self, threshold=0.2):
        self.threshold = threshold

    def __call__(self, img_tensor):
        # img_tensor is assumed to be in [0,1], shape [1, 28, 28]
        return (img_tensor > self.threshold).float()

# ---------------------
# Load model
# ---------------------
model = torch.load('small_mnist_cnn_binarized.pt')
model.eval()

size = 28


def visualize(layer, filter, lr=0.1, opt_steps=20, blur=None):
    sz = self.size
    img = np.uint8(np.random.uniform(150, 180, (sz, sz, 3)))/255  # generate random image
    activations = SaveFeatures(list(self.model.children())[layer])  # register hook

    for _ in range(self.upscaling_steps):  # scale the image up upscaling_steps times
        train_tfms, val_tfms = tfms_from_model(vgg16, sz)
        img_var = V(val_tfms(img)[None], requires_grad=True)  # convert image to Variable that requires grad
        optimizer = torch.optim.Adam([img_var], lr=lr, weight_decay=1e-6)
        for n in range(opt_steps):  # optimize pixel values for opt_steps times
            optimizer.zero_grad()
            self.model(img_var)
            loss = -activations.features[0, filter].mean()
            loss.backward()
            optimizer.step()
        img = val_tfms.denorm(img_var.data.cpu().numpy()[0].transpose(1,2,0))
        self.output = img
        sz = int(self.upscaling_factor * sz)  # calculate new image size
        img = cv2.resize(img, (sz, sz), interpolation = cv2.INTER_CUBIC)  # scale image up
        if blur is not None: img = cv2.blur(img,(blur,blur))  # blur image to reduce high frequency patterns
    self.save(layer, filter)
    activations.close()