from torchvision import transforms
import torch
from PIL import Image
import numpy as np
from torch.autograd import Variable
import cv2

#####################################################################
# About PredictionGenerator
# 
# Provides functionality related to the predictions and 
# class activation map creation
#
#####################################################################

# Normalization settings used by model (Standard ImageNet normalization)
normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)

# Transformations used for test images
transform = transforms.Compose([
   transforms.Grayscale(3),
   transforms.Resize((299,299)),
   transforms.ToTensor(),
   normalize
])

# Initial checkpoint is loaded on the CPU
checkpoint = torch.load('checkpoint', map_location=torch.device('cpu'))
model = checkpoint['model']
model.eval()

# Stores the outputs from the model layer which the activations are 
# extracted from
features_blobs = []

# 14 disease classes used for labelling prediction results
classes = ['Atelectasis',
          'Cardiomegaly',
          'Effusion',
          'Infiltration',
          'Mass',
          'Nodule',
          'Pneumonia',
          'Pneumothorax',
          'Consolidation',
          'Edema',
          'Emphysema',
          'Fibrosis',
          'Pleural_Thickening',
          'Hernia']

# Updates the model being used to a user uploaded version
def set_model(checkpoint):
  checkpoint = torch.load('checkpoint', map_location=torch.device('cpu'))
  model = checkpoint['model']
  model.eval()

# Passes the image through the model and obtains predictions
def predict(image_path):
    img = Image.open(image_path)
    batch_t = torch.unsqueeze(transform(img), 0)
    out = model(batch_t)

    h_x = torch.sigmoid(out).data.squeeze()
    probs, idx = h_x.sort(0, True)

    return [(classes[i], probs[i].item()) for i in idx]

# Creates class activation mapping of correct size which will be 
# overlayed on the initial chest x-ray image
def return_CAM(feature_conv, weight_softmax, class_idx):
    size_upsample = (299, 299)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

# Hook added to the model layer (last normal cell) to obtain the model outputs
def hook_feature(module, input, output):
    features_blobs.append(output[0].data.numpy())

# Creates the class activation map image
def generate_CAM(image_path):
	model._modules.get('cell5_normal1').register_forward_hook(hook_feature)

	params = list(model.parameters())
	weight_softmax = np.squeeze(params[-2].data.numpy())

	img_pil = Image.open(image_path)
	img_pil.save('temp.jpg')

	img_tensor = transform(img_pil)
	img_variable = Variable(img_tensor.unsqueeze(0))

	logit = model(img_variable)

	h_x = torch.sigmoid(logit).data.squeeze()
	probs, idx = h_x.sort(0, True)
	probs = probs.numpy()
	idx = idx.numpy()

	# generate class activation mapping for the top1 prediction
	CAMs = return_CAM(features_blobs[0], weight_softmax, [idx[0]])

	img = cv2.imread('temp.jpg')
	height, width, _ = img.shape
	heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
	result = heatmap * 0.3 + img * 0.5
	cv2.imwrite('CAM.jpg', result)

	return [(classes[i], probs[i].item()) for i in idx], Image.open('CAM.jpg')
