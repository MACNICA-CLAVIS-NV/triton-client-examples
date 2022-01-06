# triton-client-examples
Example client applications for NVIDIA Triton Inference Server.

## Installation (Server Side)

Use the download_model.sh scripts in the application directories, to download the models to a server. 

```
download_model.sh [download_path]
```

If not specified *download_path*, the model will be downloaded to **./model_repository**.

## Installation (Client Side)

**These application are platform agnostic. You don't need GPU in your platform.**

1. Clone this repository.
    ```
    git clone https://github.com/MACNICA-CLAVIS-NV/triton-client-examples
    ```

1. Install the modules on which these applications depend.
    ```
    python3 -m pip install -r requirements.txt
    ```

## Usage

### Tiny YOLO v2
```
python3 main.py [-h] [--camera CAMERA_ID] [--width CAPTURE_WIDTH] [--height CAPTURE_HEIGHT] [--url SERVER_URL]

Triton Tiny YOLO v2 Demo

optional arguments:
-h, --help            show this help message and exit
--camera CAMERA_ID    Camera ID (Default: 0)
--width CAPTURE_WIDTH
                        Capture Width (Default: 640)
--height CAPTURE_HEIGHT
                        Capture Height (Default: 480)
--url SERVER_URL      Triton Inference Server URL (Default: localhost:8000)
```

### Densenet Classification
```
python3 main.py [-h] [--camera CAMERA_ID] [--width CAPTURE_WIDTH] [--height CAPTURE_HEIGHT] [--url SERVER_URL]
            [--count CLASS_COUNT]

Triton Tiny YOLO v2 Demo

optional arguments:
-h, --help            show this help message and exit
--camera CAMERA_ID    Camera ID (Default: 0)
--width CAPTURE_WIDTH
                        Capture Width (Default: 640)
--height CAPTURE_HEIGHT
                        Capture Height (Default: 480)
--url SERVER_URL      Triton Inference Server URL (Default: localhost:8000)
--count CLASS_COUNT   Class Count to Display (Default: 3)
```