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

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import cv2
import sys
import os
import numpy as np
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))
import interval_counter


WINDOW_TITLE = 'Camera Test'
INFO_COLOR = (0, 255, 0)
CAMERA_ID_DEFAULT = 0
CAPTURE_WIDTH_DEFAULT = 640
CAPTURE_HEIGHT_DEFAULT = 480


def draw_info(frame, interval):

    height, width, color = frame.shape
    frame_info = 'Size:{}x{}'.format(width, height)
    cv2.putText(
        frame, frame_info, (width - 280, height - 16),
        cv2.FONT_HERSHEY_SIMPLEX, 1, INFO_COLOR, 2, cv2.LINE_AA
    )

    if interval is not None:
        fps = 1.0 / interval
        fpsInfo = 'FPS:{0:.2f}'.format(fps)
        cv2.putText(
            frame, fpsInfo, (32, height - 16),
            cv2.FONT_HERSHEY_SIMPLEX, 1, INFO_COLOR, 2, cv2.LINE_AA
        )


def main():
    # Parse the command line parameters
    parser = argparse.ArgumentParser(description='Camera Test')
    parser.add_argument('--camera',
        type=int, default=CAMERA_ID_DEFAULT, metavar='CAMERA_ID',
        help='Camera ID (Default: {})'.format(CAMERA_ID_DEFAULT))
    parser.add_argument('--width',
        type=int, default=CAPTURE_WIDTH_DEFAULT, metavar='CAPTURE_WIDTH',
        help='Capture Width (Default: {})'.format(CAPTURE_WIDTH_DEFAULT))
    parser.add_argument('--height',
        type=int, default=CAPTURE_HEIGHT_DEFAULT, metavar='CAPTURE_HEIGHT',
        help='Capture Height (Default: {})'.format(CAPTURE_HEIGHT_DEFAULT))
    args = parser.parse_args()

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

        # Get interval value
        interval = fps_counter.measure()

        draw_info(frame, interval)

        # Show captured frame n
        cv2.imshow(WINDOW_TITLE, frame)

        # Check if ESC pressed
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break
        # Check if the window was closed
        if was_window_closed():
            break

    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    main()
