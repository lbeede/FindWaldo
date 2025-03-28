# FindWaldo
Developing object detector to find various images of Waldo in PyTorch.

## Dataset
Exported various images of Waldo from a Kaggle dataset by Bilogur (2018). 

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

# References
1. Bilogur, Aleksky. 2018. Where's Waldo [Dataset]. Kaggle. Retrieved March 28, 2025, from https://www.kaggle.com/datasets/residentmario/wheres-waldo/data

2. Chakraborty, Devjyoti. (2021, November 1). Training an object detector from scratch in PyTorch. PyImageSearch. https://pyimagesearch.com/2021/11/01/training-an-object-detector-from-scratch-in-pytorch/