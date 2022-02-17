import pyrealsense2 as rs
import numpy as np
import pickle
import cv2


def main():
    pipeline = setup_pipeline()
    picklefile = f'games/{input("Name of Game: ")}.pkl'

    moves = []

    try:
        while True:
            # wait for user input to trigger next capture
            if input() == 'q':
                break

            frames = pipeline.wait_for_frames()
            print(frames.get_frame_number(), int(frames.get_timestamp()))

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                print("ERROR: could not get both depth and color frame")
                continue

            cv2.imwrite("data/current.jpg", np.asanyarray(color_frame.get_data()).copy())

            moves.append({
                "color": np.asanyarray(color_frame.get_data()).copy(),
                "depth": np.asanyarray(depth_frame.get_data()).copy()
            })

    finally:
        # Stop streaming
        pipeline.stop()
        with open(picklefile, "wb") as pkl_wb_obj:
            pickle.dump(moves, pkl_wb_obj)


def setup_pipeline():
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
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)
    # # Get the sensor once at the beginning. (Sensor index: 1)
    # sensor = pipeline.get_active_profile().get_device().query_sensors()[1]
    # # Set the exposure anytime during the operation
    # sensor.set_option(rs.option.exposure, 200000)
    device.hardware_reset()
    

    return pipeline


if __name__ == '__main__':
    main()