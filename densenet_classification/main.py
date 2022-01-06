#!/usr/bin/env python

# MIT License
#
# Copyright (c) 2021 MACNICA-CLAVIS-NV
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import os
import cv2
import sys
import argparse
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))
import triton_client
import preprocess
import interval_counter


WINDOW_TITLE = 'Triton Image Classification Demo'
MODEL_NAME = 'densenet_onnx'
CAMERA_ID_DEFAULT = 0
CAPTURE_WIDTH_DEFAULT = 640
CAPTURE_HEIGHT_DEFAULT = 480
SERVER_URL_DEFAULT = 'localhost:8000'
CLASS_COUNT_DEFAULT = 3


def convert_results(output_array):
    ret_results = []
    for results in output_array:
        for result in results:
            if output_array.dtype.type == np.object_:
                cls = "".join(chr(x) for x in result).split(':')
            else:
                cls = result.split(':')
            ret_results.append("{} ({}) = {}".format(cls[0], cls[1], cls[2]))
    return ret_results


def write_results(frame, results, interval):
    row = 32
    results = convert_results(results)
    for result in results:
        cv2.putText(
            frame, result, (32, row),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (133, 15, 127), 2, cv2.LINE_AA
        )
        row += 40

    height, width, color = frame.shape
    frame_info = 'Size:{}x{}'.format(width, height)
    cv2.putText(
        frame, frame_info, (width - 280, height - 16),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (133, 15, 127), 2, cv2.LINE_AA
    )

    if interval is not None:
        fps = 1.0 / interval
        fpsInfo = '{0}{1:.2f}'.format('FPS:', fps)
        cv2.putText(
            frame, fpsInfo, (32, height - 16),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (133, 15, 127), 2, cv2.LINE_AA
        )


def main():
    # Parse the command line parameters
    parser = argparse.ArgumentParser(description='Triton Tiny YOLO v2 Demo')
    parser.add_argument('--camera',
        type=int, default=CAMERA_ID_DEFAULT, metavar='CAMERA_ID',
        help='Camera ID (Default: {})'.format(CAMERA_ID_DEFAULT))
    parser.add_argument('--width',
        type=int, default=CAPTURE_WIDTH_DEFAULT, metavar='CAPTURE_WIDTH',
        help='Capture Width (Default: {})'.format(CAPTURE_WIDTH_DEFAULT))
    parser.add_argument('--height',
        type=int, default=CAPTURE_HEIGHT_DEFAULT, metavar='CAPTURE_HEIGHT',
        help='Capture Height (Default: {})'.format(CAPTURE_HEIGHT_DEFAULT))
    parser.add_argument('--url',
        type=str, default=SERVER_URL_DEFAULT, metavar='SERVER_URL',
        help='Triton Inference Server URL (Default: {})'.format(SERVER_URL_DEFAULT)
    )
    parser.add_argument('--count',
        type=int, default=CLASS_COUNT_DEFAULT, metavar='CLASS_COUNT',
        help='Class Count to Display (Default: {})'.format(CLASS_COUNT_DEFAULT))
    args = parser.parse_args()

    # Create Triton client
    client = triton_client.TritonClient(url=args.url)

    # Load model
    try:
        client.load_model(model_name=MODEL_NAME)
    except triton_client.TritonClientError as e:
        print(e)
        sys.exit(-1)

    # Initialize camera device
    cam_id = args.camera
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print("Cannot open camera")
        sys.exit()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # Create interval counter to measure FPS
    fps_counter = interval_counter.IntervalCounter(10)

    # Define the function to detect window close event
    if os.name == 'nt': # Windows
        was_window_closed = lambda: cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1
    elif os.name == 'posix': # Linux
        was_window_closed = lambda: cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_AUTOSIZE) < 0
    else:
        was_window_closed = lambda: False

    while True:
        # Capture frame n
        ret, frame = cap.read()

        # Preprocess frame n for model spec
        target_image = preprocess.preprocess(
            frame, client.format, client.dtype,
            client.c, client.h, client.w, 'INCEPTION'
        )

        # Get inference results for frame n-1
        results = client.get_results()

        # Get interval value
        interval = fps_counter.measure()

        # Write results for frame n-1 to OSD
        if results is not None:
            write_results(frame, results, interval)

        # Show captured frame n
        cv2.imshow(WINDOW_TITLE, frame)

        # Check if ESC pressed
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break
        # Check if the window was closed
        if was_window_closed():
            break

        # Submit inference request for frame n
        client.infer(target_image, class_count=args.count)

    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    main()
