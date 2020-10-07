import threading
import numpy as np
import cv2

class Camera():
    def __init__(self):
        self.is_opened = False
        self.use_thread = False
        self.thread_running = False
        self.cap = None

    def open(self, video):
        self.video = video
        self.cap = cv2.VideoCapture(self.video)

    def start(self):
        assert not self.thread_running

        if self.use_thread:
            self.thread_running = True
            self.thread = threading.Thread(target=grab_img, args=(self,))
            self.thread.start()

    def stop(self):
        self.thread_running = False
        if self.use_thread:
            self.thread.join()

    def read(self):
        _, img = self.cap.read()

        if img is None:
            #logging.warning('grab_img(): cap.read() returns None...')
            # looping around
            self.cap.release()
        else:
            return img

    def release(self):
        assert not self.thread_running
        if self.cap != "OK":
            self.cap.release()
            