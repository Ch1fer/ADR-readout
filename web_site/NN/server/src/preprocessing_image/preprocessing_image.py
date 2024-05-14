import os
import cv2 as cv
import numpy as np
from pathlib import Path
from web_site.NN.server.src.utils import get_file_path_in_module


def resize_image(image_path: Path) -> np.ndarray:
    image = cv.imread(cv.samples.findFile(image_path), cv.IMREAD_COLOR)
    resized_image = cv.resize(image, (500, 500))
    return resized_image


def get_median_filter(resized_image: np.ndarray) -> np.ndarray:
    gray_scaled_image = cv.cvtColor(resized_image, cv.COLOR_BGR2GRAY)
    median_filter = cv.medianBlur(gray_scaled_image, 5)
    return median_filter


def get_formed_clock_image(image_path: Path) -> np.ndarray:
    image = resize_image(image_path)
    median_filter = get_median_filter(image)
    rows = median_filter.shape[0]
    circles = cv.HoughCircles(median_filter, cv.HOUGH_GRADIENT, 1, rows / 8,
                              param1=100, param2=30,
                              minRadius=200, maxRadius=250)

    edges = cv.Canny(median_filter, 50, 150)
    lines = cv.HoughLines(edges, 1.5, np.pi / 180, 200)

    height, width = 500, 500
    result_image = np.zeros((height, width, 3), dtype=np.uint8)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            cv.circle(image, center, 5, (0, 100, 100), 3)
            cv.circle(result_image, center, 5, (0, 100, 100), 3)
            radius = i[2]
            cv.circle(image, center, radius, (255, 255, 255), 3)
            cv.circle(result_image, center, radius, (255, 255, 255), 3)

    if lines is not None:
        for line in lines:
            rho = 1
            theta = np.pi / 180
            threshold = 10
            min_line_length = 100
            max_line_gap = 20
            lines = cv.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

            for line in lines:
                x1, y1, x2, y2 = line[0]
                dist1 = np.sqrt((x1 - 250) ** 2 + (y1 - 250) ** 2)
                dist2 = np.sqrt((x2 - 250) ** 2 + (y2 - 250) ** 2)
                if dist1 <= 30 or dist2 <= 30:
                    cv.line(image, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    cv.line(result_image, (x1, y1), (x2, y2), (255, 255, 255), 2)

    directory = Path("./preprocessing_output")
    output_image_path = directory / "preprocessing_output_image.jpg"
    cv.imwrite(str(output_image_path), result_image)



if __name__ == "__main__":
    print(get_formed_clock_image(os.path.abspath(get_file_path_in_module("clock4.jpg", Path(__file__)))))
