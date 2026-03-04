# **Qwen3-VL-Video-Grounding-Frame-Propagation**

Qwen3-VL-Video-Grounding-Frame-Propagation is a sophisticated AI application designed for advanced video frame processing and precise object detection using bounding boxes. Driven by the robust Qwen3-VL-4B-Instruct-Unredacted-MAX vision-language model, this tool allows users to accurately ground and propagate target objects across sequential video frames. By analyzing input descriptions and applying precise coordinate tracking, the application calculates and overlays scaled bounding boxes and exact center points on detected elements. Built completely in Python, it integrates seamlessly with OpenCV for video frame extraction and Gradio for providing an intuitive, interactive web interface. This makes complex frame propagation and visual grounding tasks highly accessible for developers and researchers without the need for manual frame-by-frame annotation.

## Features

* **Video Frame Extraction:** Utilizes OpenCV to efficiently process videos and extract frames for analysis.
* **Precise Object Grounding:** Implements advanced vision-language processing to detect elements based on textual descriptions and output scaled bounding box coordinates.
* **Point-Based Detection:** Includes precise pointing capabilities to identify the exact center coordinates of specific objects.
* **Interactive Interface:** Features a user-friendly web interface powered by Gradio for easy video upload and visualization of detected objects.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/PRITHIVSAKTHIUR/Qwen3-VL-Video-Grounding-Frame-Propagation.git
cd Qwen3-VL-Video-Grounding-Frame-Propagation
```

### 2. Install Pre-requirements

System-level dependencies should be installed first to ensure smooth operation:

```bash
pip install -r pre-requirements.txt
```

### 3. Install Standard Dependencies

Install the core Python packages, including PyTorch, Transformers, and Gradio:

```bash
pip install -r requirements.txt
```

## How to Run

Launch the application by running the main Python script:

```bash
python app.py
```

Once the model weights are loaded and the server starts, you will receive a local URL (typically `http://127.0.0.1:7860`). Open this link in your browser to interact with the application.

## Project Structure

* `app.py`: The main application script containing the Gradio interface, video processing logic, and model execution.
* `requirements.txt`: The primary list of Python dependencies necessary for running the environment.
* `pre-requirements.txt`: A list of essential preliminary dependencies.
* `examples/`: Directory containing sample data to test the application's capabilities.
* `examples-images/`: Directory with example images to demonstrate object grounding functionality.

## Workflow

1. Upload a video or image via the web application interface.
2. Provide a textual description of the object you wish to track or detect.
3. The underlying Qwen3-VL model analyzes the frames, calculating precise bounding boxes or center coordinates.
4. The application outputs the processed visuals, highlighting the detected elements across the frames.

## License

This project is open-source and licensed under the Apache License 2.0. Please refer to the `LICENSE.txt` file within the repository for full terms and conditions.

## Contributing

Contributions are welcomed. Feel free to open issues for bug reports or submit Pull Requests to suggest enhancements and improve the tool's grounding capabilities.
