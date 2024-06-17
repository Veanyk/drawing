import time
import numpy as np
import cv2


def negative(image: np.array):
    return 255 - image


def to_black_white(image: np.array):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def change_brightness_and_contrast(image: np.array, contrast: float = 1, brightness: int = 0):
    return cv2.addWeighted(image, contrast, np.zeros(image.shape, image.dtype), 0, brightness)


def sharpen_image(image: np.array, kernel: np.array = None):
    if not kernel:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)


def sharpen_laplacian(image: np.array):
    return cv2.Laplacian(image, cv2.CV_64F)


def blur_image(image: np.array, blur: int = 1):
    return cv2.medianBlur(image, blur)


def remove_noise_gaussian(image: np.array, size_of_kernel: tuple = None, kernel_shift: int = 0):
    if size_of_kernel is None:
        size_of_kernel = (5, 5)
    return cv2.GaussianBlur(image, size_of_kernel, kernel_shift)


def change_colors(image: np.array):
    pass


def dilate(image: np.array, kernel: np.array = None):
    if not kernel:
        kernel = np.ones((3, 3))
    dilate_img = cv2.dilate(image, kernel=kernel, iterations=1)
    return dilate_img


def erode(image: np.array, kernel: np.array = None):
    if not kernel:
        kernel = np.ones((3, 3))
    erode_img = cv2.erode(image, kernel=kernel, iterations=1)
    return erode_img


if __name__ == "__main__":
    mans_picture = cv2.resize(cv2.imread("2024-06-17_16-07-15.jpg"), (1920, 1080))

    cv2.imshow("Testing window", negative(mans_picture))
    if cv2.waitKey(1) & 0xFF == "q":
        exit(1)

    time.sleep(2)

    cv2.imshow("Testing window", to_black_white(mans_picture))
    if cv2.waitKey(1) & 0xFF == "q":
        exit(1)

    time.sleep(2)

    cv2.imshow("Testing window", change_brightness_and_contrast(mans_picture, 2, 10))
    if cv2.waitKey(1) & 0xFF == "q":
        exit(1)

    time.sleep(2)

    cv2.imshow("Testing window", sharpen_image(mans_picture))
    if cv2.waitKey(1) & 0xFF == "q":
        exit(1)

    time.sleep(2)

    cv2.imshow("Testing window", sharpen_laplacian(mans_picture))
    if cv2.waitKey(1) & 0xFF == "q":
        exit(1)

    time.sleep(2)

    cv2.imshow("Testing window", blur_image(mans_picture, 5))
    if cv2.waitKey(1) & 0xFF == "q":
        exit(1)

    time.sleep(2)

    cv2.imshow("Testing window", erode(mans_picture))
    if cv2.waitKey(1) & 0xFF == "q":
        exit(1)

    time.sleep(2)

    cv2.imshow("Testing window", dilate(mans_picture))
    if cv2.waitKey(1) & 0xFF == "q":
        exit(1)

    time.sleep(2)

    cv2.imshow("Testing window", remove_noise_gaussian(mans_picture))
    if cv2.waitKey(1) & 0xFF == "q":
        exit(1)

    time.sleep(2)

    cv2.destroyAllWindows()
