import cv2
import threading
import queue
import time

class VideoStreamer:
    """
    Asynchronous video stream reader that handles both local files and live RTSP feeds.
    Implements a threaded ring buffer. For local files, it paces the reading to 
    match the video's original FPS and drops frames if the inference loop is slower,
    perfectly simulating a live real-time IP camera.
    """
    def __init__(self, source, name="Camera"):
        self.source = source
        self.name = name
        self.cap = cv2.VideoCapture(source)
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps == 0 or self.fps != self.fps: 
            self.fps = 30
            
        self.is_live = str(source).startswith("rtsp://") or str(source).startswith("http://")
        self.simulate_live = not self.is_live
        self.target_frame_time = 1.0 / self.fps
        
        # Small queue size so we always get the most recent frame
        self.q = queue.Queue(maxsize=5) 
        self.stopped = False

    def start(self):
        """Starts the background frame extraction thread."""
        t = threading.Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        """Worker thread loop for extracting frames."""
        while not self.stopped:
            t0 = time.time()
            ret, frame = self.cap.read()
            
            if not ret:
                self.q.put(None)
                self.stop()
                return
                
            # Always behave like a live buffer: drop oldest if full
            if self.q.full():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put(frame)

            # If it's a local file, pace the reading to its natural FPS
            if self.simulate_live:
                elapsed = time.time() - t0
                sleep_time = self.target_frame_time - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    def read(self):
        """Returns the next available frame from the buffer."""
        try:
            return self.q.get(timeout=1.0)
        except queue.Empty:
            return None

    def stop(self):
        """Signals the thread to terminate and releases resources."""
        self.stopped = True
        if self.cap.isOpened():
            self.cap.release()
