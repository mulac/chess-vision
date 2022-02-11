import pyrealsense2 as rs
import numpy as np
import pickle

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)
device.hardware_reset()

moves = []
picklefile = f'{input("Name of Game: ")}.pkl'

try:
    while True:
        input()
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        
        print(frames.get_frame_number(), int(frames.get_timestamp()))

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            print("ERROR: could not get both depth and color frame")
            continue

        moves.append({
            "color": np.asanyarray(color_frame.get_data()).copy(),
            "depth": np.asanyarray(depth_frame.get_data()).copy()
        })

finally:
    # Stop streaming
    pipeline.stop()
    with open(picklefile, "wb") as pkl_wb_obj:
        pickle.dump(moves, pkl_wb_obj)