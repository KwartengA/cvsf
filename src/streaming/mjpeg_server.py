"""
mjpeg_server.py
----------------
Minimal MJPEG-over-HTTP server for showing the Pi's annotated camera feed
live in a browser. A background Flask server serves whatever frame was
last handed to it via update_frame() as a multipart/x-mixed-replace
stream -- browsers decode that natively in a plain <img> tag, no
WebRTC/signaling needed.

Usage from a capture loop:
    server = MjpegServer(port=5001)
    server.start()
    ...
    server.update_frame(annotated_frame)  # call once per processed frame
    ...
    server.stop()
"""

import threading
import time

import cv2
from flask import Flask, Response


class MjpegServer:
    def __init__(self, port: int = 5001, quality: int = 80, max_fps: float = 15.0):
        self.port = port
        self.quality = quality
        self.min_interval = 1.0 / max_fps
        self._frame = None
        self._lock = threading.Lock()
        self._app = Flask(__name__)
        self._app.add_url_rule("/stream.mjpg", "stream", self._stream)
        self._server_thread = None

    def update_frame(self, frame):
        """Store the latest annotated frame (BGR numpy array) to be served next."""
        with self._lock:
            self._frame = frame

    def _generate(self):
        boundary = b"--frame"
        last_sent = 0.0
        while True:
            now = time.time()
            wait = self.min_interval - (now - last_sent)
            if wait > 0:
                time.sleep(wait)

            with self._lock:
                frame = None if self._frame is None else self._frame.copy()
            if frame is not None:
                ok, jpeg = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality]
                )
                if ok:
                    last_sent = time.time()
                    yield (
                        boundary + b"\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                        + jpeg.tobytes() + b"\r\n"
                    )

    def _stream(self):
        return Response(
            self._generate(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    def start(self):
        self._server_thread = threading.Thread(
            target=lambda: self._app.run(
                host="0.0.0.0", port=self.port, threaded=True, debug=False, use_reloader=False
            ),
            daemon=True,
        )
        self._server_thread.start()
