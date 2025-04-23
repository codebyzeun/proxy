# Proxy AI

Proxy AI is an advanced AI-powered system designed to streamline the integration of vision-based processing and text generation in real-time applications. The system utilizes computer vision algorithms to process frames from a live camera feed and generate contextually relevant descriptions or analyses based on the visual data.

## Features

- **Live Camera Feed**: Start/stop a real-time camera feed using OpenCV.
- **Image Processing Modes**:
  - **Normal**: Default mode for live camera feed.
  - **Grayscale**: Converts the feed to grayscale.
  - **Edges**: Applies edge detection algorithms to highlight outlines.
  - **Blur**: Applies a blur effect to the feed.
  - **Face Detection**: Identifies and highlights faces in the frame.
  - **Text Detection**: Recognizes text within the frame and processes it.
- **Text Generation**: Generates textual descriptions based on the processed camera feed using an AI model.
- **Model Loading**: Easily load pre-trained text generation models from a local directory.
- **Save Frames**: Save processed frames to disk in various formats (JPEG, PNG, etc.).
- **Adjustable Parameters**: Fine-tune the text generation settings such as temperature and length.

## Requirements

- Python 3.x
- OpenCV
- Tkinter
- Pillow (PIL)
- TensorFlow or PyTorch (depending on the model type)

### Install Dependencies

Before running the system, install the required dependencies by using `pip`:

```bash
pip install -r requirements.txt
```

## Installation and Setup

1. Clone the repository:

    ```bash
    git clone https://github.com/codebyzeun/proxy.git
    cd proxy
    ```

2. Ensure you have the necessary models in a `models` directory. You can place any pre-trained text generation models (like GPT, RNN, etc.) in the `models` folder.

3. Install dependencies using the command above.

## Usage

### Running the Dashboard

To run the AI Vision Dashboard, execute the following command:

```bash
python vision_dashboard.py --model "your_model_name"
```

Replace `"your_model_name"` with the name of the model you want to use. If no model is provided, the system will load a default one if available.

### Control Interface

- **Start Camera**: Starts the camera feed and begins processing frames.
- **Stop Camera**: Stops the camera feed.
- **Processing Modes**: Use the radio buttons to switch between processing modes like grayscale, edge detection, and face detection.
- **Generate Description**: Once a camera feed is active, click "Generate Description" to generate a textual description of the scene.
- **Save Frame**: Click "Save Frame" to save the current frame from the camera feed to your local machine.
- **Load Model**: Use the "Load Text Model" button to load a pre-trained text generation model for generating descriptions.

### Example Output

The output in the dashboard will display a processed camera feed on the left and a text box on the right. Depending on the chosen mode, the feed will update in real-time with effects like grayscale, edge detection, or face bounding boxes. The generated text will appear in the text area to describe the visual content.

## Contributing

Feel free to fork the repository, submit issues, or create pull requests for new features or bug fixes. Contributions are always welcome!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
