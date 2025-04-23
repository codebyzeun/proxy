# Proxy - AI Project

## Overview

Proxy is a versatile AI project developed using Python, leveraging the power of OpenCV, TensorFlow, and Keras. This project integrates computer vision and natural language processing.  It is designed to:

* Detect moods from visual input (images or video).
* Generate text using a character-level Recurrent Neural Network (RNN).

## Features

* **Mood Detection:**
    * Analyzes visual input to detect the predominant mood or emotion expressed.
    * Supports processing of both still images and video streams.
    * happy, sad, angry, neutral, surprise, fear
* **Character-Level RNN Text Generation:**
    * Generates text sequences, learning the structure and patterns of language at the character level.
    * Allows for generating text with varying styles and content, depending on the training data.
    * Provides control over the text generation process through parameters like seed text and generation length.
* **OpenCV Integration:**
    * Utilizes OpenCV for image and video capture, pre-processing, and display.
    * Enables real-time mood analysis from video streams.
* **TensorFlow and Keras:**
    * Employs TensorFlow for building and training machine learning models.
    * Uses Keras as a high-level API for defining and training neural networks, including CNNs for mood detection and RNNs for text generation.

## Core Technologies

* Python
* OpenCV
* TensorFlow
* Keras

## Project Structure

Based on the file listing, the repository has the following structure:

```
proxy/
├── .idea/                 # PyCharm project settings
├── models/                # Directory for trained models
├── venv/                  # Python virtual environment
├── main.py                # Main application entry point
├── run_vision.py          # Running computer vision tasks
├── test_text.txt          # Text file testing text generation
├── text_generation.py     # Character-level RNN text generation
├── vision_ai.py           # Core computer vision AI logic
└── vision_dashboard.py    # Vision-related dashboard or UI
```

## Setup and Installation

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/codebyzeun/proxy.git](https://github.com/codebyzeun/proxy.git)
    cd proxy
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python3 -m venv venv  # or py -3 -m venv venv on Windows
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt # Preferred method, if available
    # OR, if you don't have a requirements.txt:
    pip install opencv-python tensorflow keras
    #  Install any other dependencies.  For example:
    # pip install numpy  # If used
    ```

    * **Note:** It's highly recommended to create a `requirements.txt` file to ensure consistent dependencies.  You can generate it using `pip freeze > requirements.txt` after installing the necessary packages in your virtual environment.

4.  **Configuration:**

    * **Trained Models:** Ensure that the trained models for mood detection and text generation are located in the `models/` directory.  The scripts assume a specific file naming convention (e.g., `mood_detection_model.h5`, `text_generation_model.h5`).  If you have different names or locations, you'll need to modify the corresponding script files (`run_vision.py`, `text_generation.py`).
    * **OpenCV Setup:** OpenCV should be installed correctly on your system.  If you encounter issues with video capture, ensure that your camera or video source is properly configured.
    * **Resource Paths:** If the scripts use specific file paths (e.g., for loading images, videos, or text files), verify that these paths are correct for your system.  You might need to adjust them in the script files.
    * **GPU Configuration (Optional):** If you have a GPU, configure TensorFlow to use it for accelerated training and inference.  This typically involves installing the `tensorflow-gpu` package (instead of the regular `tensorflow`) and installing the necessary NVIDIA drivers and CUDA/cuDNN libraries.

## Usage

1.  **Running the main application:**

    ```bash
    python main.py
    ```

    * This script likely orchestrates the overall workflow of Proxy.  You'll need to examine `main.py` to understand its specific functionality.  It might involve running mood detection and/or text generation in a coordinated manner.

2.  **Running mood detection:**

    ```bash
    python run_vision.py --mode mood --input [input_source]
    ```

    * `--mode mood`:  Specifies that the script should perform mood detection.
    * `--input [input_source]`:  Specifies the source of the visual input.  This can be:
        * `image.jpg`:  A path to an image file.
        * `video.mp4`:  A path to a video file.
        * `camera`:  To use the default camera.
    * Example:
        * `python run_vision.py --mode mood --input image1.jpg`
        * `python run_vision.py --mode mood --input video.mp4`
        * `python run_vision.py --mode mood --input camera`
    * The script will output the detected mood (e.g., "happy," "sad") to the console.  If using a video stream, it might display the video with the mood overlaid.

3.  **Running character-level RNN text generation:**

    ```bash
    python text_generation.py --model [model_path] --seed "[seed_text]" --length [num_chars]
    ```

    * `--model [model_path]`:  Specifies the path to the trained character-level RNN model (e.g., `models/text_generation_model.h5`).
    * `--seed "[seed_text]"`:  Provides the initial text to start the generation (e.g., "The quick brown fox").  The model will continue this text.
    * `--length [num_chars]`:  Specifies the number of characters to generate.
    * Example:
        * `python text_generation.py --model models/text_generation_model.h5 --seed "The rain in Spain falls" --length 200`
    * The script will print the generated text to the console.

4.  **Running the vision dashboard:**

    ```bash
    python vision_dashboard.py
    ```

    * If applicable, this script will start a dashboard or user interface.  Provide details on how to use the dashboard (e.g., what it displays, what controls are available).  If you don't have a dashboard, you can remove this section.

## Example

* **Mood Detection:** "To detect the mood in a video from your webcam, run `python run_vision.py --mode mood --input camera`. The script will display the video feed with the detected mood overlaid in real-time."
* **Text Generation:** "To generate a 150-character continuation of the phrase "It was a dark and stormy night," use the command `python text_generation.py --model models/text_generation_model.h5 --seed "It was a dark and stormy night" --length 150`.  The output will be the generated text."

## Contributing

## License

[MIT]

## Contact

[Provide a way for people to contact you (e.g., email address, GitHub profile). If you have a website or blog, you can link to it here.]
