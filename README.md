# Behavioral Cloning Project

This project involves training a deep learning model to drive a car in a simulator using behavioral cloning. The model learns from human driving data and predicts steering angles based on the input images.

## Project Overview

The project consists of two main parts:
1. **Training the Model**: A Convolutional Neural Network (CNN) is trained on images captured by the car's camera along with the corresponding steering angles.
2. **Running the Model**: The trained model is deployed in a simulator where it controls the car based on the camera feed in real-time.

## File Structure

- `behavioural_Cloning.ipynb`: Jupyter Notebook containing the model architecture, training process, and data preprocessing steps.
- `drive.py`: Python script used to interface with the simulator and drive the car using the trained model.
- `model.h5`: The trained model file (not included, must be generated from the notebook or downloaded).

## Dependencies

To run this project, you will need the following dependencies:

- Python 3.7+
- TensorFlow 2.x
- Keras
- Flask
- eventlet
- numpy
- Pillow
- OpenCV
- socketio

## Model Training

The model is trained using a Convolutional Neural Network (CNN) architecture, similar to NVIDIA's model for self-driving cars. The following preprocessing steps are applied to the images:

- **Cropping**: The image is cropped to remove irrelevant parts, such as the sky and car hood.
- **Color Space Conversion**: The image is converted to the YUV color space, which is more suitable for computer vision tasks.
- **Gaussian Blur**: A Gaussian blur is applied to the image to reduce noise and smooth the image.
- **Resizing**: The image is resized to 200x66 pixels, a size compatible with the input layer of the CNN.
- **Normalization**: The image pixel values are normalized to a range between 0 and 1, improving the model's convergence during training.

### Training Process

To train the model, run the notebook `behavioural_Cloning.ipynb`. The notebook contains the following key sections:

1. **Data Loading and Augmentation**: Load the training data and apply augmentation techniques such as flipping and shifting to increase the dataset's diversity.
2. **Model Architecture Definition**: Define the CNN architecture, including convolutional layers, dropout layers, and dense layers.
3. **Model Training and Validation**: Train the model on the preprocessed data and validate its performance using a separate validation set.

## Driving the Car

Once the model is trained, it can be used to drive the car in the simulator. The `drive.py` script is responsible for this task. It loads the trained model, preprocesses the incoming images from the simulator, and sends the predicted steering angle back to the simulator.

### Running the Model

To drive the car using the trained model, execute the following command in your terminal:

```bash
python drive.py model.h5
