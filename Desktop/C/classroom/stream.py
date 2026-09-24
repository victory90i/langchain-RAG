import os

import cv2


def choose_stream_source(default_ip_url="http://192.168.1.10:8080/video"):
    """Prompt the user for the source type before opening a video stream."""
    print("Choose input source:")
    print("1) Default camera")
    print("2) IP Webcam / phone camera")
    print("3) Local video file")
    choice = input("Enter 1, 2, or 3: ").strip()

    if choice == "2":
        ip_url = input(f"Enter IP webcam URL [{default_ip_url}]: ").strip() or default_ip_url
        return ip_url

    if choice == "3":
        while True:
            video_path = input("Enter path to the video file: ").strip().strip('"')
            if video_path and os.path.exists(video_path):
                return video_path
            print("File not found. Please try again.")

    return 0


class VideoStream:
    """Simple camera or video stream wrapper."""

    def __init__(self, source=0):
        self.source = source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source!r}")

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()

    @property
    def fps(self):
        return self.cap.get(cv2.CAP_PROP_FPS) or 30.0
