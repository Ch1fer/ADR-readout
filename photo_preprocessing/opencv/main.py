import sys
import cv2 as cv
import numpy as np



def main(argv):
    ## [load]
    default_file = 'assets/clock2.jpg'
    filename = argv[0] if len(argv) > 0 else default_file

    # Loads an image
    src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_COLOR)
    src = cv.resize(src, (500, 500))
    
    # src = cv.resize(src, (0, 0), fx = 0.4, fy = 0.4)
    # Check if image is loaded fine
    if src is None:
        print ('Error opening image!')
        print ('Usage: hough_circle.py [image_name -- default ' + default_file + '] \n')
        return -1
    ## [load]

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

    ## [draw]
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv.circle(src, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv.circle(src, center, radius, (255, 0, 255), 3)

    ## [draw]
    if lines is not None:
        for line in lines:
            rho = 1
            theta = np.pi/180
            # Convert polar coordinates to Cartesian coordinates
            
            rho = 1
            theta = np.pi / 180
            threshold = 15
            min_line_length = 100
            max_line_gap = 10
            lines = cv.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

            # Draw lines on the original image
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv.line(src, (x1, y1), (x2, y2), (255, 0, 0), 2)
    ## [display]
    cv.imshow("detected circles", src)
    cv.waitKey(0)
    ## [display]

    return 0


if __name__ == "__main__":
    main(sys.argv[1:])