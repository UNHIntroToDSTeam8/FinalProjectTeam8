# Gender and Age Detection

# Objective:
To build a gender and age detector that can approximately guess the gender and age of a person (face) in a picture or through a webcam.

# About the Project:
This Python project uses deep learning to accurately identify the gender and age of a person from a single image of a face. The models trained by <a href="https://www.kaggle.com/datasets/ttungl/adience-benchmark-gender-and-age-classification">Tal Hassner and Gil Levi</a> are utilized. The predicted gender may be one of 'Male' and 'Female', and the predicted age may fall into the following ranges: (0 – 2), (4 – 6), (8 – 12), (15 – 20), (25 – 32), (38 – 43), (48 – 53), (60 – 100) (8 nodes in the final softmax layer). Predicting an exact age from a single image is challenging due to factors like makeup, lighting, obstructions, and facial expressions. Therefore, the problem is formulated as a classification task.

# Dataset:
For this Python project, the Adience dataset is used and is available in the public domain. You can find the dataset here. The Adience dataset serves as a benchmark for face photos and includes various real-world imaging conditions such as noise, lighting, pose, and appearance. The images are collected from Flickr albums and distributed under the Creative Commons (CC) license. The dataset contains a total of 26,580 photos of 2,284 subjects in eight age ranges, making it about 1GB in size. The models used in this project have been trained on this dataset.

# Additional Python Libraries Required:
pip install opencv-python
pip install gradio

# The Contents of This Project:
opencvFaceDetector.pbtxt
opencvFaceDetectorUint8.pb
ageDeploy.prototxt
ageNet.caffemodel
genderDeploy.prototxt
genderNet.caffemodel
A few pictures to try the project on
detect.py
For face detection, a .pb file is used for the graph definition and trained weights, and .pbtxt holds it in text format. These are TensorFlow files. For age and gender, the .prototxt files describe the network configuration, and the .caffemodel file defines the internal states of the parameters of the layers.

# Usage:

Download the repository.

Open your Command Prompt or Terminal and change the directory to the folder where all the files are present.

Detecting Gender and Age of face in an Image:

python genderandagedetection.py --image <image_name> OR Run the GenderAgeDetection.ipynb and upload the image and click on Submit.
