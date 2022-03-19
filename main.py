from cscore import CameraServer
from networktables import NetworkTables
import numpy
import cv2
from enum import Enum


class GripPipelineFinal:
    def __init__(self):
        self.BlurType = Enum('self.BlurType', 'Box_Blur Gaussian_Blur Median_Filter Bilateral_Filter')

        self.__resize_image_width = 480.0
        self.__resize_image_height = 360.0
        self.__resize_image_interpolation = cv2.INTER_CUBIC

        self.resize_image_output = None

        self.__hsv_threshold_input = self.resize_image_output
        self.__hsv_threshold_hue = [0.0, 180.0]
        self.__hsv_threshold_saturation = [134.37951681425253, 255.0]
        self.__hsv_threshold_value = [81.02518131407045, 255.0]

        self.hsv_threshold_output = None

        self.__cv_erode_src = self.hsv_threshold_output
        self.__cv_erode_kernel = None
        self.__cv_erode_anchor = (-1, -1)
        self.__cv_erode_iterations = 20.0
        self.__cv_erode_bordertype = cv2.BORDER_CONSTANT
        self.__cv_erode_bordervalue = (-1)

        self.cv_erode_output = None

        self.__cv_dilate_src = self.cv_erode_output
        self.__cv_dilate_kernel = None
        self.__cv_dilate_anchor = (-1, -1)
        self.__cv_dilate_iterations = 20.0
        self.__cv_dilate_bordertype = cv2.BORDER_CONSTANT
        self.__cv_dilate_bordervalue = (-1)

        self.cv_dilate_output = None

        self.__blur_input = self.cv_dilate_output
        self.__blur_type = self.BlurType.Gaussian_Blur
        self.__blur_radius = 6.006007151560739

        self.blur_output = None

        self.__find_contours_input = self.blur_output
        self.__find_contours_external_only = False

        self.find_contours_output = None

    def process(self, source0):
        # Step Resize_Image0:
        self.__resize_image_input = source0
        (self.resize_image_output) = self.__resize_image(self.__resize_image_input, self.__resize_image_width, self.__resize_image_height, self.__resize_image_interpolation)

        # Step HSV_Threshold0:
        self.__hsv_threshold_input = self.resize_image_output
        (self.hsv_threshold_output) = self.__hsv_threshold(self.__hsv_threshold_input, self.__hsv_threshold_hue, self.__hsv_threshold_saturation, self.__hsv_threshold_value)

        # Step CV_erode0:
        self.__cv_erode_src = self.hsv_threshold_output
        (self.cv_erode_output) = self.__cv_erode(self.__cv_erode_src, self.__cv_erode_kernel, self.__cv_erode_anchor, self.__cv_erode_iterations, self.__cv_erode_bordertype, self.__cv_erode_bordervalue)

        # Step CV_dilate0:
        self.__cv_dilate_src = self.cv_erode_output
        (self.cv_dilate_output) = self.__cv_dilate(self.__cv_dilate_src, self.__cv_dilate_kernel, self.__cv_dilate_anchor, self.__cv_dilate_iterations, self.__cv_dilate_bordertype, self.__cv_dilate_bordervalue)

        # Step Blur0:
        self.__blur_input = self.cv_dilate_output
        (self.blur_output) = self.__blur(self.__blur_input, self.__blur_type, self.__blur_radius)

        # Step Find_Contours0:
        self.__find_contours_input = self.blur_output
        (self.find_contours_output) = self.__find_contours(self.__find_contours_input, self.__find_contours_external_only)

    @staticmethod
    def __resize_image(input, width, height, interpolation):
        return cv2.resize(input, ((int)(width), (int)(height)), 0, 0, interpolation)

    @staticmethod
    def __hsv_threshold(input, hue, sat, val):
        out = cv2.cvtColor(input, cv2.COLOR_BGR2HSV)
        return cv2.inRange(out, (hue[0], sat[0], val[0]), (hue[1], sat[1], val[1]))

    @staticmethod
    def __cv_erode(src, kernel, anchor, iterations, border_type, border_value):
        return cv2.erode(src, kernel, anchor, iterations=(int)(iterations + 0.5),
                         borderType=border_type, borderValue=border_value)

    @staticmethod
    def __cv_dilate(src, kernel, anchor, iterations, border_type, border_value):
        return cv2.dilate(src, kernel, anchor, iterations=(int)(iterations + 0.5), borderType=border_type, borderValue=border_value)

    def __blur(self, src, type, radius):
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
    def __find_contours(input, external_only):
        if (external_only):
            mode = cv2.RETR_EXTERNAL
        else:
            mode = cv2.RETR_LIST
        method = cv2.CHAIN_APPROX_SIMPLE
        im2, contours, hierarchy = cv2.findContours(input, mode=mode, method=method)
        return contours


def main():
    left_threshold = 0.20
    right_threshold = 0.80

    screen_width = 480
    screen_height = 360

    # ----------------------------

    cserver = CameraServer()
    cserver.startAutomaticCapture()

    input_stream = cserver.getVideo()
    output = cserver.putVideo('Processed', width=screen_width, height=screen_height)

    img = numpy.zeros(shape=(screen_height, screen_width, 3), dtype=numpy.uint8)

    NetworkTables.initialize(server='10.22.64.2')
    vision_nt = NetworkTables.getTable('Vision')

    pipeline = GripPipelineFinal()

    while True:
        frame_time, frame = input_stream.grabFrame(img)

        pipeline.process(frame)
        output_data = pipeline.find_contours_output

        output_image = cv2.resize(frame, (screen_width, screen_height))

        detection_count = 0
        biggest_radius = 0
        best_detection = False

        if len(output_data) > 0:
            detection_count = len(output_data)

            for x in range(len(output_data)):
                (x, y), r = cv2.minEnclosingCircle(output_data[x])
                x = int(x)
                y = int(y)
                r = int(r)

                if r > biggest_radius:
                    biggest_radius = r
                    best_detection = {"x": x, "y": y, "r": r}

        if best_detection:
            if 0 < best_detection["x"] < (screen_width * left_threshold):
                heading = -1
            elif best_detection["x"] > (screen_width * right_threshold):
                heading = 1
            else:
                heading = 2

            vision_nt.putNumber('heading', heading)
            vision_nt.putNumber('x', best_detection["x"])
            vision_nt.putNumber('y', best_detection["y"])

            cv2.circle(img=output_image, center=(best_detection["x"], best_detection["y"]), radius=best_detection["r"], color=(0, 255, 0), thickness=5)
            print("x:", best_detection["x"], "y:", best_detection["y"], "r:", best_detection["r"], "heading:", heading)

        vision_nt.putNumber("detectionCount", detection_count)
        output.putFrame(output_image)


if __name__ == '__main__':
    main()
