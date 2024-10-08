# Import libraries
import os
import cv2
import numpy as np
from flask import flash
def extract_faces(faces_path):
	# Define paths
	base_dir = os.path.dirname(__file__)

	prototxt_path = os.path.join(base_dir + '/models/deploy.prototxt.txt')
	caffemodel_path = os.path.join(base_dir + '/models/res10_300x300_ssd_iter_140000.caffemodel')
	# Read the model
	model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

	#Create directory 'faces' if it does not exist
	if not os.path.exists('faces'):
		flash("New directory created")
		os.makedirs(base_dir + '/faces')
	# save_dir = faces_path.split(os.sep)[-2]
	# os.makedirs(faces_path.split(os.sep)[-2])
	# Loop through all images and strip out faces
	count = 0
	#for file in os.listdir(base_dir + '/all_classes/20_classes_30pic/train_dir/zuganov'):
	for file in os.listdir(faces_path):
		file_name, file_extension = os.path.splitext(file)
		if (file_extension in ['.png', '.jpg']):
			#image = cv2.imread(base_dir + '/all_classes/20_classes_30pic/train_dir/zuganov/' + file)
			image = cv2.imread(faces_path + '/' + file)

			(h, w) = image.shape[:2]
			blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

			model.setInput(blob)
			detections = model.forward()

			# Identify each face
			for i in range(0, detections.shape[2]):
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				confidence = detections[0, 0, i, 2]

				# If confidence > 0.5, save it as a separate file
				if (confidence > 0.5):
					count += 1
					frame = image[startY:endY, startX:endX]
					cv2.imwrite(base_dir + '/faces/' + str(i) + '_' + file, frame)

	flash("Extracted " + str(count) + " faces from all images")

