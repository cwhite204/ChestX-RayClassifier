import streamlit as st
from PIL import Image
from PredictionGenerator import predict, generate_CAM, set_model
import os

#####################################################################
# About ClassifyView
# 
# Responsible for creating the user interface and implementing the logic
# required to allow expected user actions such as uploading an image and
# obtaining predictions.
#
#####################################################################

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Configure page layout
st.set_page_config(page_title="X-Ray Classifier", page_icon=None, layout='centered', initial_sidebar_state='auto')

st.set_option('deprecation.showfileUploaderEncoding', False)

# Creates main title of the system
st.title("Chest X-ray Diagnosis Classifier")
st.write("")

# Allow for a model to be uploaded
st.sidebar.markdown('*Upload model - Default Model is AmoebaNet-D*')
checkpoint_up = st.sidebar.file_uploader('Upload AmoebaNet Model')

st.sidebar.markdown('*Choose threshold value and add chest x-ray image to make prediction*')

st.sidebar.write("")

# Threshold value used to hide/show prediction results
threshold = st.sidebar.slider("Pick threshold value", 0.0, 1.0, 0.1)

# Will store the path of the user uploaded image
file_up = st.file_uploader('Upload an image', type="png")

left_column, right_column = st.beta_columns(2)

if checkpoint_up is not None:
	# Set new model if user has uploaded a model
	set_model(checkpoint_up)

if file_up is not None:
	# Once image is uploaded, predictions and CAM are created
	with st.spinner("Making Prediction..."):
		image = Image.open(file_up)
		st.write('')

		labels, camImage = generate_CAM(file_up)

		left_column.image(image, caption='Original Image', use_column_width=True)
		right_column.image(camImage, caption='Class activation map', use_column_width=True)

		show_above = st.checkbox('Show above threshold')

		# Logic to show predictions on screen
		if show_above:
			st.header('Predictions above threshold:')
			count = 0;
			
			for i in labels:
				if i[1] > threshold:
					count += 1;
					st.write(i[0], '	', i[1])
			if count == 0:
				st.write("None above threshold")
		else:
			st.header('Top 3 Predictions')
			for i in range(0,3):
				st.write(labels[i][0], '	', labels[i][1])

			st.header('Rest Of The Predictions')
			for i in range(3,len(labels)):
				st.write(labels[i][0], '	', labels[i][1])

