# USAGE
# python train.py
# import the necessary packages
from bbox_regressor import ObjectDetector
from custom_tensor_dataset import CustomTensorDataset
import config
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn import CrossEntropyLoss
from torch.nn import MSELoss
from torch.optim import Adam
from torchvision.models import resnet50
from sklearn.model_selection import train_test_split
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
import time
import cv2
import os
import re

if __name__ == "__main__":
	# check to see if we are using a macOS system with GPU support

	print("Starting training for image resolution: ", config.DESIRED_RES)
	# Loop over all CSV files in the annotations directory
for csvPath in paths.list_files(config.ANNOTS_PATH, validExts=(".csv")):

	# Extract the resolution folder name from the CSV file's directory.
	# For example, if csvPath is ".../annotations/imgs/256/image_0_0.csv", then res will be "256".
	pattern = r'(\d+)\.csv$'
	res = re.search(pattern, csvPath)

	if res is None:
		print(f"[WARNING] Could not extract resolution from {csvPath}. Skipping.")
		continue

	if res.group(1) != config.DESIRED_RES:
		# Skip CSV files that do not match the desired resolution.
		continue
	# initialize the list of data (images), class labels, target bounding
	# box coordinates, and image paths
	print("[INFO] loading dataset...")
	data = []
	labels = []
	bboxes = []
	imagePaths = []

	# load the contents of the current CSV annotations file
	rows = open(csvPath).read().strip().split("\n")

	# loop over the rows
	for row in rows:
		row = row.split(",")
		try:
			(filename, width, height, label, startX, startY, endX, endY) = row
		except ValueError as e:
			print(f"[ERROR] Unable to unpack row {row} in file {csvPath}: {e}")
			continue

		# Instead of skipping, assign default bounding box values if missing
		if startX == "" or startY == "" or endX == "" or endY == "":
			# Assign default values indicating no bounding box.
			startX = startY = endX = endY = "0"

		# Use the label (e.g., 'waldo' or 'notwaldo') as the subfolder name.
		subfolder = label.strip().lower()
		# Build the image path by including the resolution folder.
		imagePath = os.path.sep.join([config.IMAGES_PATH, f"chopped-{res.group(1)}", subfolder, filename])
		
		# Check if the image exists
		if not os.path.exists(imagePath):
			print(f"[ERROR] File does not exist: {imagePath}")
			continue

		image = cv2.imread(imagePath)
		if image is None:
			print(f"[ERROR] Unable to load image {imagePath}.")
			continue

		(h, w) = image.shape[:2]
		try:
			startX = float(startX) / w
			startY = float(startY) / h
			endX = float(endX) / w
			endY = float(endY) / h
		except Exception as e:
			print(f"[ERROR] Conversion error for {filename}: {e}")
			continue

		# Proceed with the rest of your processing...
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = cv2.resize(image, (224, 224))
		data.append(image)
		labels.append(label)
		bboxes.append((startX, startY, endX, endY))
		imagePaths.append(imagePath)


	# convert the data, class labels, bounding boxes, and image paths to
	# NumPy arrays
	data = np.array(data, dtype="float32")
	labels = np.array(labels)
	bboxes = np.array(bboxes, dtype="float32")
	imagePaths = np.array(imagePaths)
	# perform label encoding on the labels
	le = LabelEncoder()
	labels = le.fit_transform(labels)
	# partition the data into training and testing splits using 80% of
	# the data for training and the remaining 20% for testing
	split = train_test_split(data, labels, bboxes, imagePaths,
		test_size=0.20, random_state=42, stratify=labels)
	# unpack the data split
	(trainImages, testImages) = split[:2]
	(trainLabels, testLabels) = split[2:4]
	(trainBBoxes, testBBoxes) = split[4:6]
	(trainPaths, testPaths) = split[6:]

	# convert NumPy arrays to PyTorch tensors
	(trainImages, testImages) = torch.tensor(trainImages),\
		torch.tensor(testImages)
	(trainLabels, testLabels) = torch.tensor(trainLabels),\
		torch.tensor(testLabels)
	(trainBBoxes, testBBoxes) = torch.tensor(trainBBoxes),\
		torch.tensor(testBBoxes)
	# define normalization transforms
	train_transforms = transforms.Compose([
		transforms.ToPILImage(),
		transforms.ToTensor(),
		transforms.Normalize(mean=config.MEAN, std=config.STD)
	])

	# convert NumPy arrays to PyTorch datasets
	trainDS = CustomTensorDataset((trainImages, trainLabels, trainBBoxes),
		transforms=train_transforms)
	testDS = CustomTensorDataset((testImages, testLabels, testBBoxes),
		transforms=train_transforms)
	print("[INFO] total training samples: {}...".format(len(trainDS)))
	print("[INFO] total test samples: {}...".format(len(testDS)))
	# calculate steps per epoch for training and validation set
	trainSteps = len(trainDS) // config.BATCH_SIZE
	valSteps = len(testDS) // config.BATCH_SIZE
	# create data loaders
	trainLoader = DataLoader(trainDS, batch_size=config.BATCH_SIZE,
		shuffle=True, num_workers=0, pin_memory=config.PIN_MEMORY)
	testLoader = DataLoader(testDS, batch_size=config.BATCH_SIZE,
		num_workers=0, pin_memory=config.PIN_MEMORY)

	# write the testing image paths to disk so that we can use then
	# when evaluating/testing our object detector
	print("[INFO] saving testing image paths...")
	f = open(config.TEST_PATHS, "w")
	f.write("\n".join(testPaths))
	f.close()
	# load the ResNet50 network
	resnet = resnet50(pretrained=True)
	# freeze all ResNet50 layers so they will *not* be updated during the
	# training process
	for param in resnet.parameters():
		param.requires_grad = False
		
	# create our custom object detector model and flash it to the current
	# device
	objectDetector = ObjectDetector(resnet, len(le.classes_))
	objectDetector = objectDetector.to(config.DEVICE)
	# define our loss functions
	classLossFunc = CrossEntropyLoss()
	bboxLossFunc = MSELoss()
	# initialize the optimizer, compile the model, and show the model
	# summary
	opt = Adam(objectDetector.parameters(), lr=config.INIT_LR)
	print(objectDetector)
	# initialize a dictionary to store training history
	H = {"total_train_loss": [], "total_val_loss": [], "train_class_acc": [],
		"val_class_acc": []}
	

	# loop over epochs
	print("[INFO] training the network...")
	startTime = time.time()
	for e in tqdm(range(config.NUM_EPOCHS)):
		# set the model in training mode
		objectDetector.train()
		# initialize the total training and validation loss
		totalTrainLoss = 0
		totalValLoss = 0
		# initialize the number of correct predictions in the training
		# and validation step
		trainCorrect = 0
		valCorrect = 0
		
		# loop over the training set
		for (images, labels, bboxes) in trainLoader:
			# send the input to the device
			(images, labels, bboxes) = (images.to(config.DEVICE),
				labels.to(config.DEVICE), bboxes.to(config.DEVICE))
			# perform a forward pass and calculate the training loss
			predictions = objectDetector(images)
			bboxLoss = bboxLossFunc(predictions[0], bboxes)
			classLoss = classLossFunc(predictions[1], labels)
			totalLoss = (config.BBOX * bboxLoss) + (config.LABELS * classLoss)
			# zero out the gradients, perform the backpropagation step,
			# and update the weights
			opt.zero_grad()
			totalLoss.backward()
			opt.step()
			# add the loss to the total training loss so far and
			# calculate the number of correct predictions
			totalTrainLoss += totalLoss
			trainCorrect += (predictions[1].argmax(1) == labels).type(
				torch.float).sum().item()
				# switch off autograd
		with torch.no_grad():
			# set the model in evaluation mode
			objectDetector.eval()
			# loop over the validation set
			for (images, labels, bboxes) in testLoader:
				# send the input to the device
				(images, labels, bboxes) = (images.to(config.DEVICE),
					labels.to(config.DEVICE), bboxes.to(config.DEVICE))
				# make the predictions and calculate the validation loss
				predictions = objectDetector(images)
				bboxLoss = bboxLossFunc(predictions[0], bboxes)
				classLoss = classLossFunc(predictions[1], labels)
				totalLoss = (config.BBOX * bboxLoss) + \
					(config.LABELS * classLoss)
				totalValLoss += totalLoss
				# calculate the number of correct predictions
				valCorrect += (predictions[1].argmax(1) == labels).type(
					torch.float).sum().item()
				
		# calculate the average training and validation loss
		avgTrainLoss = totalTrainLoss / trainSteps
		avgValLoss = totalValLoss / valSteps
		# calculate the training and validation accuracy
		trainCorrect = trainCorrect / len(trainDS)
		valCorrect = valCorrect / len(testDS)

		# update our training history
		H["total_train_loss"].append(avgTrainLoss.cpu().detach().numpy())
		H["train_class_acc"].append(trainCorrect)
		H["total_val_loss"].append(avgValLoss.cpu().detach().numpy())
		H["val_class_acc"].append(valCorrect)
		# print the model training and validation information
		print("[INFO] EPOCH: {}/{}".format(e + 1, config.NUM_EPOCHS))
		print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
			avgTrainLoss, trainCorrect))
		print("Val loss: {:.6f}, Val accuracy: {:.4f}".format(
			avgValLoss, valCorrect))
	endTime = time.time()
	print("[INFO] total time taken to train the model: {:.2f}s".format(
		endTime - startTime))

	# serialize the model to disk
	print("[INFO] saving object detector model...")
	torch.save(objectDetector, config.MODEL_PATH)
	# serialize the label encoder to disk
	print("[INFO] saving label encoder...")
	f = open(config.LE_PATH, "wb")
	f.write(pickle.dumps(le))
	f.close()
	# plot the training loss and accuracy
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(H["total_train_loss"], label="total_train_loss")
	plt.plot(H["total_val_loss"], label="total_val_loss")
	plt.plot(H["train_class_acc"], label="train_class_acc")
	plt.plot(H["val_class_acc"], label="val_class_acc")
	plt.title("Total Training Loss and Classification Accuracy on Dataset")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="lower left")
	# save the training plot
	plotPath = os.path.sep.join([config.PLOTS_PATH, f"{res.group(1)}-training.png"])
	plt.savefig(plotPath)