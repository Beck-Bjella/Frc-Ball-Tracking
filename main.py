from cscore import CameraServer
from networktables import NetworkTables
import numpy
import cv2
from enum import Enum


class GripPipeline:
    def __init__(self, hue_threshold, saturation_threshold, value_threshold):
        self.BlurType = Enum('self.BlurType', 'Box_Blur Gaussian_Blur Median_Filter Bilateral_Filter')
        """initializes all values to presets or None if need to be set
        """

        self.__resize_image_width = 160.0
        self.__resize_image_height = 90.0
        self.__resize_image_interpolation = cv2.INTER_CUBIC

        self.resize_image_output = None

        self.__blur_input = self.resize_image_output
        self.__blur_type = self.BlurType.Gaussian_Blur
        self.__blur_radius = 3.8472440561712458

        self.blur_output = None

        self.__hsv_threshold_input = self.blur_output
        self.__hsv_threshold_hue = hue_threshold
        self.__hsv_threshold_saturation = saturation_threshold
        self.__hsv_threshold_value = value_threshold

        self.hsv_threshold_output = None

    def process(self, source0):
        """
        Runs the pipeline and sets all outputs to new values.
        """
        # Step Resize_Image0:
        self.__resize_image_input = source0
        (self.resize_image_output) = self.__resize_image(self.__resize_image_input, self.__resize_image_width, self.__resize_image_height, self.__resize_image_interpolation)

        # Step Blur0:
        self.__blur_input = self.resize_image_output
        (self.blur_output) = self.__blur(self.__blur_input, self.__blur_type, self.__blur_radius)

        # Step HSV_Threshold0:
        self.__hsv_threshold_input = self.blur_output
        (self.hsv_threshold_output) = self.__hsv_threshold(self.__hsv_threshold_input, self.__hsv_threshold_hue, self.__hsv_threshold_saturation, self.__hsv_threshold_value)

    @staticmethod
    def __resize_image(input, width, height, interpolation):
        """Scales and image to an exact size.
        Args:
            input: A numpy.ndarray.
            Width: The desired width in pixels.
            Height: The desired height in pixels.
            interpolation: Opencv enum for the type fo interpolation.
        Returns:
            A numpy.ndarray of the new size.
        """
        return cv2.resize(input, ((int)(width), (int)(height)), 0, 0, interpolation)

    def __blur(self, src, type, radius):
        """Softens an image using one of several filters.
        Args:
            src: The source mat (numpy.ndarray).
            type: The self.BlurType to perform represented as an int.
            radius: The radius for the blur as a float.
        Returns:
            A numpy.ndarray that has been blurred.
        """
        if (type is self.BlurType.Box_Blur):
            ksize = int(2 * round(radius) + 1)
            return cv2.blur(src, (ksize, ksize))
        elif (type is self.BlurType.Gaussian_Blur):
            ksize = int(6 * round(radius) + 1)
            return cv2.GaussianBlur(src, (ksize, ksize), round(radius))
        elif (type is self.BlurType.Median_Filter):
            ksize = int(2 * round(radius) + 1)
            return cv2.medianBlur(src, ksize)
        else:
            return cv2.bilateralFilter(src, -1, round(radius), round(radius))

    @staticmethod
    def __hsv_threshold(input, hue, sat, val):
        """Segment an image based on hue, saturation, and value ranges.
        Args:
            input: A BGR numpy.ndarray.
            hue: A list of two numbers the are the min and max hue.
            sat: A list of two numbers the are the min and max saturation.
            lum: A list of two numbers the are the min and max value.
        Returns:
            A black and white numpy.ndarray.
        """
        out = cv2.cvtColor(input, cv2.COLOR_BGR2HSV)
        return cv2.inRange(out, (hue[0], sat[0], val[0]), (hue[1], sat[1], val[1]))


def main():
    team_color = "red"

    red_hue_threshold = [0.0, 22.0]
    red_saturation_threshold = [76.17461099255837, 254.73577810920378]
    red_value_threshold = [64.20863309352518, 254.14949555665729]

    blue_hue_threshold = [96.0, 109.0]
    blue_saturation_threshold = [76.17461099255837, 254.73577810920378]
    blue_value_threshold = [64.20863309352518, 254.14949555665729]

    left_threshold = 0.48
    right_threshold = 0.52

    bottom_threshold = 0.9

    screen_width = 160
    screen_height = 90

    # ----------------------------

    cserver = CameraServer()
    cserver.startAutomaticCapture()

    input_stream = cserver.getVideo()
    img = numpy.zeros(shape=(screen_width, screen_height, 3), dtype=numpy.uint8)

    NetworkTables.initialize(server='10.22.64.2')
    vision_nt = NetworkTables.getTable('Vision')

    if team_color == "red":
        pipeline = GripPipeline(red_hue_threshold, red_saturation_threshold, red_value_threshold)
    else:
        pipeline = GripPipeline(blue_hue_threshold, blue_saturation_threshold, blue_value_threshold)

    while True:
        frame_time, input_img = input_stream.grabFrame(img)

        pipeline.process(input_img)
        output_image = numpy.array(pipeline.hsv_threshold_output)

        circles = cv2.HoughCircles(output_image, cv2.HOUGH_GRADIENT, 1, minDist=10, param1=100, param2=10, minRadius=5, maxRadius=160)

        best_detection = None
        biggest_radius = 0

        if circles is not None:
            for circle in circles[0, :]:
                x, y, r = int(circle[0]), int(circle[1]), int(circle[2])

                if r > biggest_radius:
                    biggest_radius = r
                    best_detection = [x, y, r]

        heading = 0
        close_to_bottom = 0

        if best_detection:
            if 0 < best_detection[0] < (screen_width * left_threshold):
                heading = -1
            elif best_detection[0] > (screen_width * right_threshold):
                heading = 1
            else:
                heading = 2

            if best_detection[1] > (screen_height * bottom_threshold):
                close_to_bottom = 1

        vision_nt.putNumber('x', int(best_detection[0]))
        vision_nt.putNumber('y', int(best_detection[1]))
        vision_nt.putNumber('r', int(best_detection[2]))
        vision_nt.putNumber('bottom', close_to_bottom)
        vision_nt.putNumber('heading', heading)


if __name__ == '__main__':
    main()
