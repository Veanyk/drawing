"""
Файл с классами, реализующими выделение и замену фона изображения
"""
import time
import numpy as np
import cv2
from cvzone.SelfiSegmentationModule import SelfiSegmentation


class BackgroundSubtractor:
    """
    Основной (абстрактный) класс для отделения фона от переднего плана
    """
    def fit(self, background_image: np.ndarray) -> None:
        """
        Метод, применяющийся для запоминания фона без изображений на переднем плане
        :param background_image: Изображение фона
        :return: None
        """
        pass

    def subtract_background(self, image: np.ndarray) -> None:
        """
        Метод, применяющийся для отделения фона от изображения
        :param image: Изображение
        :return: Изображение "без" фона
        """
        pass

    def change_background(self, image: np.array, chosen_background: np.array) -> None:
        """
        Метод, заменяющий фон изображения на переданный
        :param image: Изображение
        :param chosen_background: Новый фон
        :return: Изображение с измененным фоном
        """
        pass


class MySubtractor(BackgroundSubtractor):
    """
    Класс, реализующий "собственную" реализацию определения фона изображения. На данный момент в испытаниях показал себя хуже всего
    """
    def __init__(self):
        self.background_image = None

    def fit(self, background_image: np.ndarray) -> None:
        self.background_image = background_image

    def subtract_background(self, image: np.ndarray, threshold: int = 5) -> np.ndarray:
        """
        Метод, применяющийся для отделения фона от изображения
        :param image: Изображение
        :param threshold: Пороговое значение, при котором пиксель считается причастным к фону. Чем выше значение - тем больше пикселей будут отнесены к фону
        :return:
        """
        diff = cv2.absdiff(self.background_image, image)
        mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        imask = mask > threshold

        transformed_image = np.zeros_like(image, np.uint8)
        transformed_image[imask] = image[imask]

        return transformed_image

    def change_background(self, image: np.ndarray, chosen_background: np.ndarray) -> np.ndarray:
        my_image = self.subtract_background(image)
        mask = cv2.cvtColor(my_image, cv2.COLOR_BGR2GRAY)
        imask = mask == 0
        my_image[imask] = chosen_background[imask]
        return my_image


class CvSubtractor(BackgroundSubtractor):
    """
    Класс, реализующий разделение фона и изображение на основе модели (KNN) из библиотеки opencv. На данный момент в испытаниях показал среднее качество
    """
    def __init__(self):
        self.model = cv2.createBackgroundSubtractorKNN()
        self.background_image = None

    def fit(self, background_image: np.ndarray) -> None:
        self.background_image = background_image
        for _ in range(50):
            self.model.apply(background_image)

    def subtract_background(self, image: np.ndarray) -> np.ndarray:
        transformed_image = self.model.apply(image)
        for _ in range(50):
            self.model.apply(self.background_image)
        return transformed_image

    def change_background(self, image: np.ndarray, chosen_background: np.ndarray) -> np.ndarray:
        my_image = self.subtract_background(image)
        # mask = cv2.cvtColor(my_image, cv2.COLOR_BGR2GRAY)
        imask = my_image == 0
        image[imask] = chosen_background[imask]
        return image


class CvzoneSubtractor(BackgroundSubtractor):
    """
    класс, реализующий разделение фона и изображения на основе модели из библиотеки cvzone. На данный момент в испытаниях показал наилучшую производительность.
    """
    def __init__(self):
        self.model = SelfiSegmentation()

    def fit(self, background_image: np.ndarray = None) -> None:
        """
        Метод, применяющийся для запоминания фона без изображений на переднем плане. Для данного класса может не применяться, оставлен для совместимости
        :param background_image: Изображение фона
        :return: None
        """
        pass

    def subtract_background(self, image: np.ndarray, threshold: float = 0.2) -> np.ndarray:
        """
        Метод, применяющийся для отделения фона от изображения
        :param image: Изображение
        :param threshold: Пороговое значение, при котором пиксель считается причастным к фону. Чем выше значение - тем больше пикселей будут отнесены к фону
        :return:
        """
        img_out = self.model.removeBG(image, (0, 0, 0), cutThreshold=threshold)
        return img_out

    def change_background(self, image: np.ndarray, chosen_background: np.ndarray, threshold: float = 0.2) -> np.ndarray:
        """
        Метод, заменяющий фон изображения на переданный
        :param image: Изображение
        :param chosen_background: Новый фон
        :param threshold: Пороговое значение, при котором пиксель считается причастным к фону. Чем выше значение - тем больше пикселей будут отнесены к фону
        :return: Изображение с измененным фоном
        """
        img_out = self.model.removeBG(image, chosen_background, cutThreshold=threshold)
        return img_out


if __name__ == "__main__":
    """
    Если запустить этот файл, а не импортировать, то будут проведен ряд тестов.
    """
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
