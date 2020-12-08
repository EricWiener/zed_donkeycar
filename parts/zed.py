"""
Author: Eric Wiener
File: zed.py
Date: November 29 20202
Notes: Donkeycar part for the ZED depth camera
"""
import time
import logging
import math
import pyzed.sl as sl
import numpy as np


##
# Basic class to handle the timestamp of the different sensors to know if it is a new sensors_data or an old one
class TimestampHandler:
    def __init__(self):
        self.t_imu = sl.Timestamp()
        self.t_baro = sl.Timestamp()
        self.t_mag = sl.Timestamp()

    ##
    # check if the new timestamp is higher than the reference one, and if yes, save the current as reference
    def is_new(self, sensor):
        if (isinstance(sensor, sl.IMUData)):
            new_ = (sensor.timestamp.get_microseconds() >
                    self.t_imu.get_microseconds())
            if new_:
                self.t_imu = sensor.timestamp
            return new_
        elif (isinstance(sensor, sl.MagnetometerData)):
            new_ = (sensor.timestamp.get_microseconds() >
                    self.t_mag.get_microseconds())
            if new_:
                self.t_mag = sensor.timestamp
            return new_
        elif (isinstance(sensor, sl.BarometerData)):
            new_ = (sensor.timestamp.get_microseconds() >
                    self.t_baro.get_microseconds())
            if new_:
                self.t_baro = sensor.timestamp
            return new_


##
#  Function to display sensor parameters
def printSensorParameters(sensor_parameters):
    if sensor_parameters.is_available:
        print("*****************************")
        print("Sensor type: " + str(sensor_parameters.sensor_type))
        print("Max rate: " + str(sensor_parameters.sampling_rate) + " " +
              str(sl.SENSORS_UNIT.HERTZ))
        print("Range: " + str(sensor_parameters.sensor_range) + " " +
              str(sensor_parameters.sensor_unit))
        print("Resolution: " + str(sensor_parameters.resolution) + " " +
              str(sensor_parameters.sensor_unit))
        if not math.isnan(sensor_parameters.noise_density):
            print("Noise Density: " + str(sensor_parameters.noise_density) +
                  " " + str(sensor_parameters.sensor_unit) + "/√Hz")
        if not math.isnan(sensor_parameters.random_walk):
            print("Random Walk: " + str(sensor_parameters.random_walk) + " " +
                  str(sensor_parameters.sensor_unit) + "/s/√Hz")


class ZED(object):
    """
    Donkeycar part for the ZED depth camera.
    The ZED camera is a device which uses an imu, twin fisheye cameras
    """
    def __init__(self,
                 enable_rgb=True,
                 enable_depth=True,
                 enable_imu=False,
                 verbose=False):
        self.verbose = verbose
        self.enable_imu = enable_imu
        self.enable_rgb = enable_rgb
        self.enable_depth = enable_depth

        # Create a ZED camera object
        self.zed = sl.Camera()
        self.zed_info = self.zed.get_camera_information()

        # Used to check if the data is new
        self.zed_timestamp_handler = TimestampHandler()

        # Set configuration parameters
        init_params = sl.InitParameters()
        init_params.sdk_verbose = self.verbose  # Enable verbose logging

        if self.enable_rgb:
            init_params.camera_resolution = sl.RESOLUTION.HD1080
            init_params.camera_fps = 30

        if self.enable_depth:
            init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # Set the depth mode to performance (fastest)
            init_params.coordinate_units = sl.UNIT.MILLIMETER  # Use millimeter units

        if self.enable_imu:
            if self.zed_info.camera_model == sl.MODEL.ZED:
                print(
                    "IMU data is only available for ZED-M and ZED2. Disabling IMU data"
                )
                self.enable_imu = False

        # Open the camera
        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print("Unable to open ZED camera with error: {}".format(err))
            exit(1)

        # Check that we are connected
        # Get camera information (serial number)
        if self.verbose:
            zed_serial = self.zed.get_camera_information().serial_number
            print("Successfully connected to ZED {}".format(zed_serial))

        time.sleep(2.0)  # let camera warm up

        # initialize frame state
        self.color_image = None
        self.depth_image = None
        self.point_cloud = None
        self.imu_quaternion = None
        self.linear_acceleration = None
        self.angular_velocity = None
        self.magnetic_field = None
        self.barometer_pressure = None
        self.frame_count = 0
        self.start_time = time.time()
        self.frame_time = self.start_time

        self.running = True

    def get_zed_info(self):
        # Display camera information (model, serial number, firmware version)
        info = self.zed_info
        print("Camera model: {}".format(info.camera_model))
        print("Serial Number: {}".format(info.serial_number))
        print("Camera Firmware: {}".format(
            info.camera_configuration.firmware_version))
        print("Sensors Firmware: {}".format(
            info.sensors_configuration.firmware_version))

        # Display accelerometer sensor configuration
        sensor_parameters = info.sensors_configuration.accelerometer_parameters
        print("Sensor Type: {}".format(sensor_parameters.sensor_type))
        print("Sampling Rate: {}".format(sensor_parameters.sampling_rate))
        print("Range: {}".format(sensor_parameters.sensor_range))
        print("Resolution: {}".format(sensor_parameters.resolution))

        import math
        if math.isfinite(sensor_parameters.noise_density):
            print("Noise Density: {}".format(sensor_parameters.noise_density))
        if math.isfinite(sensor_parameters.random_walk):
            print("Random Walk: {}".format(sensor_parameters.random_walk))

    def _poll(self):
        last_time = self.frame_time
        self.frame_time = time.time() - self.start_time
        self.frame_count += 1

        #
        # get the frames
        #
        runtime_parameters = sl.RuntimeParameters()

        if self.enable_rgb:
            image = sl.Mat()

        if self.enable_depth:
            depth = sl.Mat()
            point_cloud = sl.Mat()

        # Grab an image, a RuntimeParameters object must be given to grab()
        if self.zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # A new image is available if grab() returns ERROR_CODE.SUCCESS

            if self.enable_rgb:
                # Get the left image
                self.zed.retrieve_image(image, sl.VIEW.LEFT)

                # Get an np array from ZED Matrix
                rgba_np_image = image.get_data()

                # Drop the alpha channel
                rgb_np_image = rgba_np_image[:, :, :-1]

                # Convert rgb to bgr
                self.color_image = rgb_np_image[:, :, ::-1]

                if self.verbose:
                    # Get the image timestamp
                    timestamp = self.zed.get_timestamp(sl.TIME_REFERENCE.IMAGE)
                    print("Image resolution: {0} x {1} || Image timestamp: {2}\n".
                          format(image.get_width(), image.get_height(),
                                timestamp.get_milliseconds()))

            if self.enable_depth:
                # Retrieve depth matrix. Depth is aligned on the left RGB image
                self.zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
                self.depth_image = depth.get_data()

                # Retrieve colored point cloud
                self.zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
                self.point_cloud = point_cloud.get_data()

                if self.verbose and self.depth_image is not None:
                    y, x = self.depth_image.shape
                    x = x // 2
                    y = y // 2
                    depth_value = self.depth_image[x, y]
                    print("Distance to Camera at ({0}, {1}): {2} mm".format(x, y, depth_value), end="\r")

            if self.enable_imu:
                sensors_data = sl.SensorsData()
                self.zed.get_sensors_data(sensors_data,
                                          sl.TIME_REFERENCE.CURRENT)

                if self.zed_timestamp_handler.is_new(sensors_data.get_imu_data()):
                    self.imu_quaternion = sensors_data.get_imu_data().get_pose().get_orientation().get()
                    self.linear_acceleration = sensors_data.get_imu_data().get_linear_acceleration()
                    self.angular_velocity = sensors_data.get_imu_data().get_angular_velocity()

                    if self.verbose:
                        print("IMU Orientation: {}".format(self.imu_quaternion))
                        print("IMU Acceleration: {} [m/sec^2]".format(self.linear_acceleration))
                        print("IMU Angular Velocity: {} [deg/sec]".format(self.angular_velocity))

                # Check if Magnetometer data has been updated
                if self.zed_timestamp_handler.is_new(sensors_data.get_magnetometer_data()):
                    self.magnetic_field = sensors_data.get_magnetometer_data().get_magnetic_field_calibrated()

                    if self.verbose:
                        print("Magnetometer Magnetic Field: {} [uT]".format(self.magnetic_field))

                # Check if Barometer data has been updated
                if self.zed_timestamp_handler.is_new(sensors_data.get_barometer_data()):
                    self.barometer_pressure = sensors_data.get_barometer_data().pressure

                    if self.verbose:
                        print("Barometer Atmospheric pressure: {} [hPa]".format(self.barometer_pressure))

    def update(self):
        """
        When running threaded, update() is called from the background thread
        to update the state.  run_threaded() is called to return the latest state.
        """
        while self.running:
            self._poll()

    def run_threaded(self):
        """
        Return the lastest state read by update().  This will not block.
        All 4 states are returned, but may be None if the feature is not enabled when the camera part is constructed.
        For gyroscope, x is pitch, y is yaw and z is roll.
        :return: (rbg_image: nparray, depth_image: nparray, point_cloud: nparray, 
                  imu_quaternion: nparray(x:float, y:float, z:float, w: float), linear_acceleration: float, 
                  angular_velocity: float, magnetic_field: float, barometer_pressure: float)
        """
        return self.color_image, self.depth_image, self.point_cloud, self.imu_quaternion, self.linear_acceleration, self.angular_velocity, self.magnetic_field, self.barometer_pressure

    def run(self):
        """
        Read and return frame from camera.  This will block while reading the frame.
        see run_threaded() for return types.
        """
        self._poll()
        return self.run_threaded()

    def shutdown(self):
        self.running = False
        time.sleep(2)  # give thread enough time to shutdown

        # done running
        self.zed.close()


#
# self test
#
if __name__ == "__main__":

    show_opencv_window = False  # True to show images in opencv window: note that default donkeycar environment is not configured for this.
    if show_opencv_window:
        import cv2

    enable_rgb = True
    enable_depth = True
    enable_imu = False

    profile_frames = 0  # set to non-zero to calculate the max frame rate using given number of frames

    try:
        #
        # for D435i, enable_imu can be True, for D435 enable_imu should be false
        #
        camera = ZED(
            enable_rgb=enable_rgb,
            enable_depth=enable_depth,
            enable_imu=enable_imu,
            verbose=True
        )

        frame_count = 0
        start_time = time.time()
        frame_time = start_time
        while True:
            #
            # read data from camera
            #
            color_image, depth_image, point_cloud, imu_quaternion, linear_acceleration, angular_velocity, magnetic_field, barometer_pressure = camera.run(
            )

            # maintain frame timing
            frame_count += 1
            last_time = frame_time
            frame_time = time.time()

            if enable_imu and not profile_frames:
                print(
                    "imu frame {} in {} seconds: \n\linear acceleration = {}, \n\tangular velocity = {}, \n\tquaternion = {}"
                    .format(
                        str(frame_count), str(frame_time - last_time),
                        str(linear_acceleration),
                        str(angular_velocity),
                        str(imu_quaternion)))

            # Show images
            if show_opencv_window and not profile_frames:
                # cv2.namedWindow('ZED', cv2.WINDOW_AUTOSIZE)
                if enable_rgb or enable_depth:
                    # make sure depth and color images have same number of channels so we can show them together in the window
                    depth_colormap = cv2.applyColorMap(
                        cv2.convertScaleAbs(depth_image, alpha=0.03),
                        cv2.COLORMAP_JET) if enable_depth else None
                
                    # Stack both images horizontally
                    images = None
                    if enable_rgb:
                        images = np.hstack(
                            (color_image,
                             depth_colormap)) if enable_depth else color_image
                    elif enable_depth:
                        images = depth_colormap

                    if images is not None:
                        cv2.imshow('ZED', images)

                # Press esc or 'q' to close the image window
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q') or key == 27:
                    cv2.destroyAllWindows()
                    break
            if profile_frames > 0:
                if frame_count == profile_frames:
                    print("Aquired {} frames in {} seconds for {} fps".format(
                        str(frame_count), str(frame_time - start_time),
                        str(frame_count / (frame_time - start_time))))
                    break
            else:
                time.sleep(0.05)
    finally:
        camera.shutdown()
