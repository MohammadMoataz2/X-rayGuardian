# Chest X-Ray Pneumonia Detection

This project focuses on detecting Pneumonia in chest X-ray images or video using artificial intelligence and deep learning techniques. The solution involves preprocessing the data, training a model using TensorFlow, and developing a desktop application to classify new X-ray images or video frames as either normal or showing signs of Pneumonia.



![image](https://github.com/MohammadMoataz2/Weather-Forecasting-Application/assets/123085286/b3292a4a-a77a-400b-a38a-d4fa5b2f1107)

## Project Overview

### Dataset
The dataset consists of chest X-ray images categorized into two classes:
- Normal
- Pneumonia

### Model
The model is built using TensorFlow and involves:
- Data preprocessing
- Creating a pipeline for data handling
- Utilizing a pre-trained VGG16 model
- Applying transfer learning and fine-tuning

### Application
A desktop application is developed using Tkinter, allowing users to upload and analyze X-ray images or videos. The application provides the following functionalities:
- Select the path of the photo or video
- Analyze the selected file to classify it as normal or Pneumonia

For images, the application uses OpenCV to display the label directly on the image within the Tkinter window and also saves the labeled image externally. For videos, the application processes each frame, labels it, and displays the result.


![image](https://github.com/MohammadMoataz2/Weather-Forecasting-Application/assets/123085286/013a0f32-0582-4793-a862-6acb5b47cf21)




## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/chest-xray-pneumonia-detection.git
    cd chest-xray-pneumonia-detection
    ```

2. Create and activate a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```


## Usage


1. **Running the Application**:
    - Start the desktop application:
        ```sh
        python main.py
        ```
    - Use the interface to select an X-ray image or video for analysis.
    - Click the button to start the analysis and view the results.


![image](https://github.com/MohammadMoataz2/Weather-Forecasting-Application/assets/123085286/e75e16ef-5f28-4645-98cb-36e76d84a4a9)


## Features

- **Image Analysis**:
  - Upload an X-ray image.
  - The application displays the image with the label (Normal or Pneumonia) overlaid.

- **Video Analysis**:
  - Upload a video containing X-ray frames.
  - The application processes each frame and labels it accordingly.

## Dependencies

- TensorFlow
- OpenCV
- Tkinter

![image](https://github.com/MohammadMoataz2/Weather-Forecasting-Application/assets/123085286/68e744d1-c767-4c13-be42-87aec50dda21)




