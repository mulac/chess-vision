""" Interface for grabbing frames from a camera or disk """

import abc
import logging
import pyrealsense2 as rs
import numpy as np
import cv2


class BaseCamera(metaclass=abc.ABCMeta):
    
    cancel_signal = True

    def loop(self, callback):
        """ Will continually pass frames to callback.  
        Stops when callback returns True (Camera.cancel_signal) """
        while callback(*self.fetch()) != self.cancel_signal: pass

    @abc.abstractmethod
    def fetch(self):
        """ Returns a tuple of frames """
        pass

    @abc.abstractmethod
    def close(self):
        """ After the camera is closed frames can no longer be fetched """
        pass


class Camera(BaseCamera):
    """ Default Camera using OpenCV VideoCapture

    Args:
        feed: (int) camera index || (str) video filepath
    """
    # TODO add camera parameters (fps)
    def __init__(self, feed=0):
        self.feed = feed
        self.capture = cv2.VideoCapture(feed)
        
    def fetch(self):
        ret, frame = self.capture.read()
        if not ret: logging.warning(f"Camera {self.feed} failed to fetch frame")
        return frame, None
        
    def close(self):
        self.capture.release()


class RealsenseCamera(BaseCamera):
    def __init__(self, pipeline=None):
        self.pipeline = self.setup_pipeline() if pipeline is None else pipeline
        self.resolution = self.width, self.height = 1920, 1080

    def setup_pipeline(self):
        # Configure depth and color streams
        pipeline, config = rs.pipeline(), rs.config()

        # Get device model for specs
        pipeline_profile = config.resolve(rs.pipeline_wrapper(pipeline))
        device = pipeline_profile.get_device()

        if not any(s.get_info(rs.camera_info.name) == 'RGB Camera' for s in device.sensors):
            raise EnvironmentError("The demo requires Depth camera with Color sensor")

        config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 10)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.align = rs.align(rs.stream.color)
        
        pipeline.start(config)
        # sensor = pipeline.get_active_profile().get_device().query_sensors()[1]
        # sensor.set_option(rs.option.exposure, 200000) # Set the exposure anytime during the operation
        device.hardware_reset()

        return pipeline

    def ndarray(cls, frame):
        return np.asanyarray(frame.get_data()).copy()

    def fetch(self):
        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)
        depth = frames.get_depth_frame()
        color = frames.get_color_frame()

        if not (color and depth):
            logging.warning(f"RealsenseCamera: failed to fetch frames")
        
        return self.ndarray(color), self.ndarray(depth)

    def close(self):
        self.pipeline.stop()