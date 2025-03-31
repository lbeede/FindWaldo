# FindWaldo
Developing object detector to find various images of Waldo in PyTorch. See ```image_processing_process.ipynb``` for how we preprocessed the images and ```findingwaldo.ipynb``` on how to run this for yourself.

## Dataset
We were tasked with using images of Waldo from a Kaggle dataset by Bilogur (2018). However, this dataset did not have annotations for the locations of Waldo, so we retrieved images and their corresponding bounding boxes with Waldo's coordinates from Kenstler (2017). We independently chopped the images from Kenstler to match the original provided tasks by creating our own functions for the chopping and subsequently assigning the bounding boxes coordinates for each chopped image.

## Methods
Code primarily follows the methods described in Chakraborty (2021) in creating own object detector.
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
### ```bbox_regressor.py```
- Custom model, ```ObjectDetector```
- Regressor –– produce 4 separate values
- Classifier for object label
- ```forward``` –– output of base model passed through regressor and classifier
### ```xml_to_csv.py```
- Converts the bounding boxes in XML files from [3] into CSV files for our use case here
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

# References
1. Bilogur, Aleksky. 2018. Where's Waldo [Dataset]. Kaggle. Retrieved March 28, 2025, from https://www.kaggle.com/datasets/residentmario/wheres-waldo/data
2. Kenstler, B. 2017. There's Waldo: A fully-convolutional DenseNet approach to solving "Where's Waldo?" GitHub repository. Retrieved March 31, 2025, from https://github.com/bckenstler/TheresWaldo
3. Chakraborty, Devjyoti. (2021, November 1). Training an object detector from scratch in PyTorch. PyImageSearch. https://pyimagesearch.com/2021/11/01/training-an-object-detector-from-scratch-in-pytorch/