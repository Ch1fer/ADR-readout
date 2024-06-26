import os
import cv2 as cv
import numpy as np


def draw(file_path):
    extension = file_path.split('.')[1]
    
    # Loads an image
    src = cv.imread(cv.samples.findFile(file_path), cv.IMREAD_COLOR)
    src = cv.resize(src, (500, 500))

    ## [convert_to_gray]
    # Convert it to gray
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    ## [convert_to_gray]

    ## [reduce_noise]
    # Reduce the noise to avoid false circle detection
    gray = cv.medianBlur(gray, 5)
    ## [reduce_noise]

    ## [houghcircles]
    rows = gray.shape[0]
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8,
                               param1=100, param2=30,
                               minRadius=200, maxRadius=250)
    ## [houghcircles]
    edges = cv.Canny(gray, 50,150)
    lines = cv.HoughLines(edges, 1.5, np.pi / 180, 200)

    height, width = 500, 500
    result_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    ## [draw]
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv.circle(src, center, 5, (0, 100, 100), 3)
            cv.circle(result_image, center, 5, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv.circle(src, center, radius, (255, 255, 255), 3)
            cv.circle(result_image, center, radius, (255, 255, 255), 3)

    ## [draw]
    if lines is not None:
               
        for line in lines:
            
            # Convert polar coordinates to Cartesian coordinates
            rho = 1
            theta = np.pi / 180
            threshold = 10
            min_line_length = 100
            max_line_gap = 20
            lines = cv.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

            # Draw lines on the original image
            for line in lines:
                x1, y1, x2, y2 = line[0]
                dist1 = np.sqrt((x1 - 250) ** 2 + (y1 - 250) ** 2)
                dist2 = np.sqrt((x2 - 250) ** 2 + (y2 - 250) ** 2)
                if dist1 <= 30 or dist2 <= 30:
                    cv.line(src, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    cv.line(result_image, (x1, y1), (x2, y2), (255, 255, 255), 2)
    ## [display]
    # cv.imshow("detected circles", src)
    # cv.imshow("mask", result_image)
    # cv.waitKey(0)
    
    output_image_path = f"photo_preprocessing/result_image/output_image.{extension}"
    cv.imwrite(output_image_path, result_image)
    
    # print(output_image_path)
    return output_image_path


# example
if __name__ == "__main__":
    draw("photo_preprocessing/assets/clock1.jpg")