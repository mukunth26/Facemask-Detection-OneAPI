# Facemask-Detection-OneAPI

# INSPIRATION  <img src="https://user-images.githubusercontent.com/72274851/219947466-3dd124b6-ce28-493b-b4bd-157fa25fd093.png" width="60" height="60"> 

In the wake of the COVID-19 pandemic, wearing masks has become an essential practice to help prevent the spread of the virus. However, ensuring compliance with mask-wearing guidelines in public places can be challenging, especially in high-traffic areas. A machine learning model that can accurately detect whether a person is wearing a mask or not can help enforce compliance with these guidelines and improve public health outcomes.The model should be able to handle various scenarios, such as different types of masks, different lighting conditions, and different angles and poses of the person in the image

# PROJECT DESCRIPTION :memo:
The machine learning project aims to implement machine learning algorithms  to detect whether a person in a given image is wearing a mask or not.The algorithm used in this project is Convolutional Neural Network (CNN) which is a type of neural network that is specifically designed to process and analyze visual data such as images or videos.The dataset used for this project consists of images of people wearing masks and people not wearing masks: https://www.kaggle.com/datasets/omkargurav/face-mask-dataset.

The project will be useful in various situations such as:

:round_pushpin: Public places such as airports, hospitals, and shopping malls to enforce mask-wearing policies

:round_pushpin: Educational institutions to enforce mask-wearing policies among students

:round_pushpin: Industries such as construction and manufacturing to ensure the safety of workers in hazardous areas.



# WORKING  <img src="https://user-images.githubusercontent.com/72274851/222216353-58874ba5-d9cc-4298-baab-4255bbdb0193.png" width="60" height="60"> 
:arrow_right: The objective of the project is to accurately identify whether the person in an input image is wearing a mask or not.To achieve this, we will use a Convolutional Neural Network (CNN) architecture that is very effective for image classification tasks.

:arrow_right: The preprocessing stage of the project includes steps such as labellng the images with mask and without mask, resizing and converting the images into numpy arrays which will be stored in a list which will be used to train the CNN model.

:arrow_right: The dataset will be split into training, validation, and testing sets. The training and validation sets will be used to train and tune the CNN model, respectively. The testing set will be used to evaluate the performance of the trained model on new data.

:arrow_right:The CNN model is designed and trained to classify images as either containing a person wearing a mask or not.The model includes 2 convolutional layers, 2 max-pooling layers, and 2 fully dense layers

:arrow_right:The output layer has 2 neurons (one for each class). The output will be a probability distribution over the 2 classes, with which we will arrive at a binary classification decision that indicates whether a mask is present in the image or not.

# OPTIMIZATION USING ONEAPI
![neww](https://user-images.githubusercontent.com/130204205/230727046-eea03a8a-83d8-4890-ae96-87eacb43ac54.jpeg) ![images](https://user-images.githubusercontent.com/130204205/230724985-8cb85f10-a326-4758-893e-3fa49f3a53f6.jpeg)



oneDNN  which is a part of OneAPI is an open-source library developed by Intel that provides optimized implementations of deep learning primitives and algorithms such as convolutional neural networks (CNNs) and recurrent neural networks (RNNs). It supports a wide range of deep learning frameworks and can be integrated with various programming languages.
This project uses oneDNN to optimize the performance and efficiency of the CNN model. The tensorflow framework which is integrated with OneDNN is used to achieve lower memory consumption,higher accuracy,faster training times and better utilization of hardware resource.

<img width="762" alt="Screenshot 2023-04-08 at 7 45 08 PM" src="https://user-images.githubusercontent.com/130204205/230725954-72d3d17e-9c11-4974-b94b-5935319c41c4.png">

Intel(R) Extension for Scikit-learn is also used which provides a seamless way to speed up the Scikit-learn application.

<img width="745" alt="Screenshot 2023-04-08 at 7 48 34 PM" src="https://user-images.githubusercontent.com/130204205/230726131-cf899d57-f52d-4077-8d78-fc781e26d53f.png">

The test accuracy achieved after the optimizations made by the OneDNN on the CNN model is 93.44%

Overall, integrating OneDNN with TensorFlow can provide significant performance benefits for deep learning algorithms, making it a powerful tool for developing high-performance deep learning applications.

# RESULT AND LEARNINGS

A Simple but effective Interface with Streamlit was created which allows the user to either paste the URL of the image or upload the image. Then the input image,probability distribution for the image and the final prediction made by the model will be displayed.

:arrow_right: Giving URL of the image as input to the interface

<img width="1221" alt="Screenshot 2023-04-09 at 12 11 40 AM" src="https://user-images.githubusercontent.com/130204205/230738475-32276862-d925-4cbc-8085-44f06fde2855.png">


:arrow_right: Display of the prediction made by the CNN model

<img width="1414" alt="Screenshot 2023-04-09 at 12 27 02 AM" src="https://user-images.githubusercontent.com/130204205/230738494-8edab987-e1b6-4899-b57c-50cc8490cad8.png">


The learnings from this project are:

:white_check_mark: Building application using intel oneAPI interface , mainly OneDNN library.

:white_check_mark: Understanding different machine learning algorithms and selection of algorithm for the specific problem.

:white_check_mark: Understanding of the data preprocessing stage for images and using them to train the CNN model

:white_check_mark: Understanding how to design and train CNN model for image classification tasks


