import cv2 as cv
import numpy as np
import sys


def draw_circles(original, image):
    # perform Canny edge detection image should be gray
    edges = cv.Canny(image, 50, 150)
    # find contours in the edges
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # find the minimum enclosing circle for each contour
    count = 0
    for c in contours:
        ((x, y), radius) = cv.minEnclosingCircle(c)
        if radius > 20:
            cv.circle(original, (int(x), int(y)), int(radius), (0, 255, 0), 2)
            count += 1

    return original, count


if __name__ == '__main__':

    img = cv.imread('photos/eyes.jpg')

    if img is None:
        sys.exit("Could not read the image.")

    cv.imshow("Oh-riginal", img)
    cv.waitKey(0)

    # convert the image to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray_blur = cv.GaussianBlur(gray, (5, 5), 0)
    cv.imshow("Display blur", gray_blur)
    cv.waitKey(0)

    # perform thresholding using morphological operations
    _, morph = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)
    # perform erosion
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    eroded = cv.erode(morph, kernel, iterations=4)
    # perform dilation
    dilated = cv.dilate(eroded, kernel, iterations=2)
    cv.imshow("Morphed", dilated)
    cv.waitKey(0)

    # find contours in the image
    contours, _ = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # draw the contours on a blank image with the same size as the input image
    contours_image = np.zeros_like(img)
    cv.drawContours(contours_image, contours, -1, (0, 255, 0), 2)
    # show the image with the contours
    cv.imshow("Contours", contours_image)
    cv.waitKey(0)

    # perform Hough Circle Transform draw circles
    img_circle, count = draw_circles(img, dilated)
    cv.imshow("circles", img_circle)
    cv.waitKey(0)

    cv.destroyAllWindows()

    # print the number of objects found
    print("== Analysis ==")
    # print("Number of contoured objects:", len(contours))
    print("Number of circled objects:", count)
