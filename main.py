import numpy
import numpy as np
import cv2 as cv
from grip import GripPipeline

capture = cv.VideoCapture(0)

if not capture.isOpened():
    raise IOError("Cannot open webcam")

pipeline = GripPipeline()

while True:
    ret, frame = capture.read()

    pipeline.process(frame)
    x, y, w, h = cv.boundingRect(numpy.array(pipeline.cv_erode_output))

    canvas = np.zeros((480, 640, 3), dtype="uint8")

    canvas = cv.circle(canvas, (x, y), int(w / 2), (0, 0, 255), -1)

    cv.imshow("Canvas", canvas)
    c = cv.waitKey(1)
    if c == 27:
        break

capture.release()
cv.destroyAllWindows()
