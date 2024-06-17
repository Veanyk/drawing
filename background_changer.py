import time

import numpy as np
import cv2
from cvzone.SelfiSegmentationModule import SelfiSegmentation


class BackgroundSubtractor:
    def fit(self, background_image: np.array):
        pass

    def subtract_background(self, image: np.array):
        pass

    def change_background(self, image: np.array, chosen_background: np.array):
        pass


class MySubtractor(BackgroundSubtractor):
    def __init__(self):
        self.background_image = None

    def fit(self, background_image: np.array):
        self.background_image = background_image

    def subtract_background(self, image: np.array, threshold: int = 5):
        diff = cv2.absdiff(self.background_image, image)
        mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        imask = mask > threshold

        transformed_image = np.zeros_like(image, np.uint8)
        transformed_image[imask] = image[imask]

        return transformed_image

    def change_background(self, image: np.array, chosen_background: np.array):
        my_image = self.subtract_background(image)
        mask = cv2.cvtColor(my_image, cv2.COLOR_BGR2GRAY)
        imask = mask == 0
        my_image[imask] = chosen_background[imask]
        return my_image


class CvSubtractor(BackgroundSubtractor):
    def __init__(self):
        self.model = cv2.createBackgroundSubtractorKNN()
        self.background_image = None

    def fit(self, background_image: np.array):
        self.background_image = background_image
        for _ in range(50):
            self.model.apply(background_image)

    def subtract_background(self, image: np.array):
        transformed_image = self.model.apply(image)
        for _ in range(50):
            self.model.apply(self.background_image)
        return transformed_image

    def change_background(self, image: np.array, chosen_background: np.array):
        my_image = self.subtract_background(image)
        # mask = cv2.cvtColor(my_image, cv2.COLOR_BGR2GRAY)
        imask = my_image == 0
        image[imask] = chosen_background[imask]
        return image


class CvzoneSubtractor(BackgroundSubtractor):
    def __init__(self):
        self.model = SelfiSegmentation()

    def fit(self, background_image: np.array):
        pass

    def subtract_background(self, image: np.array, threshold: float = 0.2):
        img_out = self.model.removeBG(image, (0, 0, 0), cutThreshold=threshold)
        return img_out

    def change_background(self, image: np.array, chosen_background: np.array, threshold: float = 0.2):
        img_out = self.model.removeBG(image, chosen_background, cutThreshold=threshold)
        return img_out


if __name__ == "__main__":
    background_picture = cv2.resize(cv2.imread("2024-06-17_16-06-49.jpg"), (1920, 1080))
    mans_picture = cv2.resize(cv2.imread("2024-06-17_16-07-15.jpg"), (1920, 1080))
    new_background_picture = cv2.resize(cv2.imread("New_background.jpg"), (1920, 1080))

    my_subtractor = MySubtractor()
    cv_subtractor = CvSubtractor()
    cvzone_subtractor = CvzoneSubtractor()

    my_subtractor.fit(background_picture)
    cvzone_subtractor.fit(background_picture)
    cv_subtractor.fit(background_picture)

    my_subtractor_image = my_subtractor.change_background(mans_picture, new_background_picture)
    cv_subtractor_image = cv_subtractor.change_background(mans_picture, new_background_picture)
    cvzone_subtractor_image = cvzone_subtractor.change_background(mans_picture, new_background_picture)

    cv2.imshow("Testing window", my_subtractor_image)
    if cv2.waitKey(1) & 0xFF == "q":
        exit(1)

    time.sleep(2)

    cv2.imshow("Testing window", cv_subtractor_image)
    if cv2.waitKey(1) & 0xFF == "q":
        exit(1)

    time.sleep(2)

    cv2.imshow("Testing window", cvzone_subtractor_image)
    if cv2.waitKey(1) & 0xFF == "q":
        exit(1)

    time.sleep(2)

    cv2.destroyAllWindows()
