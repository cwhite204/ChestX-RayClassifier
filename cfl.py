from torchvision import models, transforms
import torch
from PIL import Image
import numpy as np
from torch.autograd import Variable
import cv2

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
transform = transforms.Compose([
   transforms.Grayscale(3),
   transforms.Resize((299,299)),
   transforms.ToTensor(),
   normalize
])

checkpoint = torch.load('checkpoint', map_location=torch.device('cpu'))
model = checkpoint['model']
model.eval()

features_blobs = []

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

def predict(image_path):
    img = Image.open(image_path)
    batch_t = torch.unsqueeze(transform(img), 0)
    out = model(batch_t)

    h_x = torch.sigmoid(out).data.squeeze()
    probs, idx = h_x.sort(0, True)

    return [(classes[i], probs[i].item()) for i in idx]

def returnCAM(feature_conv, weight_softmax, class_idx):
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


def hook_feature(module, input, output):
    features_blobs.append(output[0].data.numpy())

def generateCAM(image_path):
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
	CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])

	img = cv2.imread('temp.jpg')
	height, width, _ = img.shape
	heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
	result = heatmap * 0.3 + img * 0.5
	cv2.imwrite('CAM.jpg', result)

	return [(classes[i], probs[i].item()) for i in idx], Image.open('CAM.jpg')




