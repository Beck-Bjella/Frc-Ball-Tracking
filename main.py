import numpy
import numpy as np
import cv2 as cv
from grip import GripPipeline


def render_frame(ball_x, ball_y):
    canvas = np.zeros((480, 640, 3), dtype="uint8")

    canvas = cv.circle(canvas, (ball_x, ball_y), int(w / 2), (0, 0, 255), -1)

    cv.imshow("Canvas", canvas)


if __name__ == '__main__':
    capture = cv.VideoCapture(0)
    if not capture.isOpened():
        raise IOError("Cannot open webcam")

    pipeline = GripPipeline()

    while True:
        ret, frame = capture.read()

        pipeline.process(frame)
        x, y, w, h = cv.boundingRect(numpy.array(pipeline.cv_erode_output))

        render_frame(x, y)

        c = cv.waitKey(1)
        if c == 27:
            break

    capture.release()
    cv.destroyAllWindows()
