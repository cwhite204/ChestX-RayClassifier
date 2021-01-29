import streamlit as st
from PIL import Image
from cfl import predict, generateCAM
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

st.set_page_config(page_title="X-Ray Classifier", page_icon=None, layout='centered', initial_sidebar_state='auto')

st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("Chest X-ray Diagnosis Application")
st.write("")

st.sidebar.markdown('*Choose threshold value and add chest x-ray image to make prediction*')

st.sidebar.write("")

threshold = st.sidebar.slider("Pick threshold value", 0.0, 1.0, 0.1)

file_up = st.file_uploader('Upload an image', type="png")

left_column, right_column = st.beta_columns(2)

if file_up is not None:
	with st.spinner("Making Prediction..."):
		image = Image.open(file_up)
		st.write('')

		labels, camImage = generateCAM(file_up)

		left_column.image(image, caption='Original Image', use_column_width=True)
		right_column.image(camImage, caption='Class activation map', use_column_width=True)
		labels.sort(key = lambda i: i[1], reverse=True)

		show_above = st.checkbox('Show above threshold')


		if show_above:
			st.header('Predcitions above threshold:')
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

