from PredictionGenerator import *

# Ensure model uploaded is used
def test_set_model():
	temp_model = model

	set_model('checkpoint')
	assert model == temp_model

# Ensure highest probability class predicted by model is infiltration
def test_predict():
	results = predict('temp.jpg')
	assert results[0][0] == 'Infiltration'

# Test return cam function
def test_return_cam():
	model._modules.get('cell5_normal1').register_forward_hook(hook_feature)

	params = list(model.parameters())
	weight_softmax = np.squeeze(params[-2].data.numpy())

	img_pil = Image.open('temp.jpg')

	img_tensor = transform(img_pil)
	img_variable = Variable(img_tensor.unsqueeze(0))

	logit = model(img_variable)

	h_x = torch.sigmoid(logit).data.squeeze()
	probs, idx = h_x.sort(0, True)
	probs = probs.numpy()
	idx = idx.numpy()

	CAMs = return_CAM(features_blobs[0], weight_softmax, [idx[0]])

	img = cv2.imread('temp.jpg')
	height, width, _ = img.shape
	heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
	result = heatmap * 0.3 + img * 0.5

	cv2.imwrite('temp_CAM.jpg', result)

	assert Image.open('temp_CAM.jpg') == Image.open('CAM.jpg')


# Test generate cam function
def test_generate_cam():
	results = generate_CAM('temp.jpg')
	predict_results = predict('temp.jpg')

	assert(results[0][0][0] == predict_results[0][0])