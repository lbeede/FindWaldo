# FindWaldo
Developing object detector to find various images of Waldo in PyTorch. See ```image_processing_process.ipynb``` for how we preprocessed the images and ```findingwaldo.ipynb``` on how to run this for yourself.

## Dataset
We were tasked with using images of Waldo from a Kaggle dataset by Bilogur (2018). However, this dataset did not have annotations for the locations of Waldo, so we retrieved images and their corresponding bounding boxes with Waldo's coordinates from Kenstler (2017). We independently chopped the images from Kenstler to match the original provided tasks by creating our own functions for the chopping and subsequently assigning the bounding boxes coordinates for each chopped image.

## Methods
Code primarily follows the methods described in Chakraborty (2021) in creating own object detector. The model was trained on MPS instead of CPU, but it can easily be swtiched to CPU or GPU in the ```config.py``` file.
### ```config.py```
**REQUIRED CONFIGURATIONS:**
- ```BASE_PATH``` –– define the base location for data and images
- ```BASE_OUTPUT``` –– define the location for results to be saved  
_Optional Configurations:_
- ```MEAN```
- ```STD```
- ```INIT_LR```
- ```NUM_EPOCHS```
- ```BATCH_SIZE```
- ```LABELS```
- ```BBOX```
- ```DESIRED_RES``` –– define the image resolution (256x256, 128x128, 64x64)
### ```bbox_regressor.py```
- Custom model, ```ObjectDetector```
- Regressor –– produce 4 separate values
- Classifier for object label
- ```forward``` –– output of base model passed through regressor and classifier
### ```custom_tensor_dataset.py```
- A custom class for data preparation
- Created by Chakraborty (2021)
### ```image_processing.py```
- Functions to preprocess the original image data
### ```train.py```
- Preprocesses the data
  - Unpacks the CSV files
  - Train/test split in 80/20 ratio
  - Normalize the data according to values defined in ```config.py```
- Using ```resnet50```
- Using Cross-Entropy loss for classifier
- Using MSE for regressor
- Object detection optimizer: Adam
### ```predict.py```
- The final step to our process is predicting Waldo's location in each image
- From the test set, we implement our trained object detector

## Results
For each image resolution, we used a different number of epochs. We used 15 epochs for 256x256 images, 10 epochs for 128x128 images, and 3 epochs for 64x64 images. This was a choice because the object detector was exhibiting high accuracy and low loss very early on as shown below. For 128x128 images, we could've even used only 5 epochs seeing that the model stopped learning a significant amount.

![Figure 1. Accuracy and loss plot for 256x256](/output/plots/256-training.png)
![Figure 2. Accuracy and loss plot for 128x128](/output/plots/128-training.png)
![Figure 3. Accuracy and loss plot for 64x64](/output/plots/64-training.png)

# References
1. Bilogur, Aleksky. 2018. Where's Waldo [Dataset]. Kaggle. Retrieved March 28, 2025, from https://www.kaggle.com/datasets/residentmario/wheres-waldo/data
2. Kenstler, B. 2017. There's Waldo: A fully-convolutional DenseNet approach to solving "Where's Waldo?" GitHub repository. Retrieved March 31, 2025, from https://github.com/bckenstler/TheresWaldo
3. Chakraborty, Devjyoti. (2021, November 1). Training an object detector from scratch in PyTorch. PyImageSearch. https://pyimagesearch.com/2021/11/01/training-an-object-detector-from-scratch-in-pytorch/