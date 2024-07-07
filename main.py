import cv2
import mediapipe as mp
import numpy as np
from cvzone.SelfiSegmentationModule import SelfiSegmentation
from PIL import Image

# Инициализация mediapipe для отслеживания рук и лица
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation


# Функция для вычисления координат указательного пальца
def get_index_finger_tip(landmarks, image):
    return (int(landmarks[8].x * image.shape[1]), int(landmarks[8].y * image.shape[0]))

# Функция для определения, сжата ли рука в кулак
def is_fist(landmarks):
    for i in [8, 12, 16, 20]:  # Кончики пальцев
        if landmarks[i].y < landmarks[i - 3].y:
            return False
    return True

# Функция для обновления положения колеса
def update_wheel_position(position, velocity, positions):
    position += velocity
    position %= positions  # Всего позиций на колесе
    return position

# Функция для отрисовки вертикального колеса
def draw_vertical_wheel(image, position, positions):
    center_index = int(position) % len(positions)
    wheel_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i in range(-1, 2):
        pos_index = (center_index + i) % len(positions)
        text = positions[pos_index]
        x_offset = 30
        if i == 0:
            color = (0, 255, 0, 255)
            scale = 1.5
            thickness = 3
        else:
            color = (128, 128, 128, 255)
            scale = 1
            thickness = 1

        y_pos = image.shape[0] // 3 + i * 60
        x_pos = x_offset
        cv2.putText(wheel_image, text, (x_pos, y_pos), font, scale, color, thickness)

    return wheel_image

# Функция для загрузки и масштабирования изображения
def load_image(path):
    image = Image.open(path)
    # Преобразование изображения в 4-канальный формат с альфа-каналом
    image = image.convert("RGBA")
    np.array(image)
    # Преобразование изображения в массив NumPy
    image_array = np.array(image)
    # Перестановка каналов для получения формата BGRA
    image_array = image_array[..., [2, 1, 0, 3]]
    return image_array

# Функция для уменьшения изображения для колеса
def resize_image(image, width, height):
    pil_image = Image.fromarray(image)
    pil_image = pil_image.resize((width, height), Image.Resampling.LANCZOS)
    return np.array(pil_image)

# Функция для отрисовки горизонтального колеса с изображениями
def draw_wheel(image, position, images):
    center_index = int(position) % len(images)
    wheel_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)

    for i in range(-2, 3):
        pos_index = (center_index + i) % len(images)
        icon_image = images[pos_index]
        if i == 0:
            scale = 1.0
            y_offset = 60
        elif abs(i) == 1:
            scale = 1 / 1.5
            y_offset = 50
        else:
            scale = 1 / (1.5 * 1.5)
            y_offset = 40

        x_offset = 50
        y_pos = y_offset
        x_pos = x_offset + image.shape[1] // 2 + i * 100

        wheel_icon_resized = cv2.resize(icon_image, (0, 0), fx=scale, fy=scale)
        ih, iw, _ = wheel_icon_resized.shape
        x1, y1 = x_pos - iw // 2, y_pos - ih // 2
        x2, y2 = x1 + iw, y1 + ih

        # Проверка и добавление альфа-канала, если его нет
        if wheel_icon_resized.shape[2] == 3:
            wheel_icon_resized = cv2.cvtColor(wheel_icon_resized, cv2.COLOR_BGR2BGRA)

        # Добавляем прозрачность к изображению колеса
        alpha_foreground = wheel_icon_resized[:, :, 3] / 255.0
        alpha_background = 1.0 - alpha_foreground

        for c in range(0, 3):
            wheel_image[y1:y2, x1:x2, c] = (
                alpha_foreground * wheel_icon_resized[:, :, c] +
                alpha_background * wheel_image[y1:y2, x1:x2, c]
            )

        wheel_image[y1:y2, x1:x2, 3] = alpha_foreground * 255

    return wheel_image

# Функция для альфа-блендинга
def alpha_blend(foreground, background, x_offset, y_offset):
    fg_h, fg_w, _ = foreground.shape
    bg_h, bg_w, _ = background.shape

    if x_offset + fg_w > bg_w or y_offset + fg_h > bg_h:
        raise ValueError("Изображение переднего плана выходит за границы фона")

    alpha_foreground = foreground[:, :, 3] / 255.0
    alpha_background = 1.0 - alpha_foreground

    for c in range(0, 3):
        background[y_offset:y_offset+fg_h, x_offset:x_offset+fg_w, c] = (
            alpha_foreground * foreground[:, :, c] +
            alpha_background * background[y_offset:y_offset+fg_h, x_offset:x_offset+fg_w, c]
        )

    return background

# Функция для наложения панели с альфа-блендингом
def overlay_panel(frame, panel_img, x, y, scale=1):
    panel_img_resized = cv2.resize(panel_img, (0, 0), fx=scale, fy=scale)
    panel_h, panel_w, _ = panel_img_resized.shape

    # Проверка и добавление альфа-канала, если его нет
    if frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

    return alpha_blend(panel_img_resized, frame, x, y)

# Функция для наложения изображения шляпы без обрезания
def overlay_hat(image, hat_image, x, y):
    ih, iw, _ = hat_image.shape
    bg_h, bg_w, _ = image.shape

    if x < 0 or y < 0 or x + iw > bg_w or y + ih > bg_h:
        return image  # если изображение шляпы выходит за границы кадра, не накладывать

    alpha_hat = hat_image[:, :, 3] / 255.0
    alpha_image = 1.0 - alpha_hat

    for c in range(0, 3):
        image[y:y+ih, x:x+iw, c] = (alpha_hat * hat_image[:, :, c] +
                                    alpha_image * image[y:y+ih, x:x+iw, c])

    return image

# Поворот шляпы
def rotate_image(image, angle):
    return image.rotate(angle, expand=True)

# Функция для обработки шляп
def handle_face_and_second_wheel(frame_copy, face_results, selected_image, img_width, img_height,images):
    if face_results.multi_face_landmarks and selected_image is not None and int(second_wheel_position) % len(images) != 0:
        for face_landmarks in face_results.multi_face_landmarks:
            # Координаты ключевых точек
            chin_landmark = face_landmarks.landmark[152]  # Подбородок
            nose_landmark = face_landmarks.landmark[1]  # Нос
            left_ear_landmark = face_landmarks.landmark[234]  # Левое ухо
            right_ear_landmark = face_landmarks.landmark[454]  # Правое ухо
            left_eye_landmark = face_landmarks.landmark[33]  # Левый глаз
            right_eye_landmark = face_landmarks.landmark[263]  # Правый глаз

            # Преобразование относительных координат в абсолютные
            chin_x = int(chin_landmark.x * img_width)
            chin_y = int(chin_landmark.y * img_height)
            nose_x = int(nose_landmark.x * img_width)
            nose_y = int(nose_landmark.y * img_height)
            left_ear_x = int(left_ear_landmark.x * img_width)
            right_ear_x = int(right_ear_landmark.x * img_width)
            left_eye_x = int(left_eye_landmark.x * img_width)
            left_eye_y = int(left_eye_landmark.y * img_height)
            right_eye_x = int(right_eye_landmark.x * img_width)
            right_eye_y = int(right_eye_landmark.y * img_height)

            # Определение верхушки головы
            dx = nose_x - chin_x
            dy = nose_y - chin_y
            if int(second_wheel_position) % len(images) == 2:
                head_top_x = chin_x + int(2 * dx)
                head_top_y = chin_y + int(2 * dy)
            elif int(second_wheel_position) % len(images) == 3:
                head_top_x = chin_x + int(2.5 * dx)
                head_top_y = chin_y + int(2.5 * dy)
            else:
                head_top_x = chin_x + int(2.25 * dx)
                head_top_y = chin_y + int(2.25 * dy)

            # Масштабирование шляпы до ширины головы
            if int(second_wheel_position) % len(images) == 2:
                hat_width = int((right_ear_x - left_ear_x) * 1.2)
            elif int(second_wheel_position) % len(images) == 4:
                hat_width = int((right_ear_x - left_ear_x) * 1.5)
            else:
                hat_width = int((right_ear_x - left_ear_x) * 1.35)
            hat_height = int(selected_image.shape[0] * (hat_width / selected_image.shape[1]))

            # Проверка на то, что ширина и высота не нулевые или отрицательные
            if hat_width > 0 and hat_height > 0:
                # Изменение размера шляпы с использованием Pillow
                hat_image_pil = Image.fromarray(cv2.cvtColor(selected_image, cv2.COLOR_BGRA2RGBA))
                hat_resized_pil = hat_image_pil.resize((hat_width, hat_height), Image.Resampling.LANCZOS)

                # Вычисление угла наклона шляпы
                eye_dx = right_eye_x - left_eye_x
                eye_dy = right_eye_y - left_eye_y
                angle = -np.degrees(np.arctan2(eye_dy, eye_dx))

                # Поворот шляпы с использованием функции rotate_image
                rotated_hat_pil = rotate_image(hat_resized_pil, angle)
                rotated_hat = cv2.cvtColor(np.array(rotated_hat_pil), cv2.COLOR_RGBA2BGRA)

                # Координаты для размещения шляпы
                hat_x1 = head_top_x - rotated_hat.shape[1] // 2
                hat_y1 = head_top_y - rotated_hat.shape[0]

                # Сдвиг шляпы в зависимости от угла поворота
                if angle > 0:
                    hat_x1 -= int(hat_width * 0.035)  # Сдвиг вправо
                elif angle < 0:
                    hat_x1 += int(hat_width * 0.035)  # Сдвиг влево

                # Наложение шляпы на изображение
                frame_copy = overlay_hat(frame_copy, rotated_hat, hat_x1, hat_y1)
            else:
                print(f"Invalid hat dimensions: width = {hat_width}, height = {hat_height}")

        return frame_copy

# Функция для удаления фона
def remove_background(image):
    mp_selfie_segmentation = mp.solutions.selfie_segmentation

    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
        # Преобразование изображения из BGR в RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = selfie_segmentation.process(rgb_image)

        # Получение маски сегментации
        mask = results.segmentation_mask

        # Улучшение маски
        mask = (mask > 0.1).astype(np.uint8)  # Бинаризация маски
        mask = cv2.GaussianBlur(mask, (5, 5), 0)  # Размытие маски для сглаживания краев

        # Преобразование изображения в формат RGBA
        rgba_image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

        # Создание нового изображения с черным фоном (4 канала)
        bg_image = np.zeros(rgba_image.shape, dtype=np.uint8)

        # Маскирование изображения
        for c in range(4):  # Применение маски к каждому каналу
            rgba_image[:, :, c] = rgba_image[:, :, c] * mask

        # Объединение исходного изображения и фона
        transparent_background = np.where(np.stack((mask,) * 4, axis=-1) > 0, rgba_image, bg_image)

        return transparent_background

# Функция для замены фона
def overlay_background(image, background_image):
    segmentor = SelfiSegmentation()

    # Изменяем размер background_image до размера image
    background_image_resized = cv2.resize(background_image, (image.shape[1], image.shape[0]))

    # Убедимся, что у обоих изображений три канала
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if background_image_resized.shape[2] == 4:
        background_image_resized = cv2.cvtColor(background_image_resized, cv2.COLOR_BGRA2BGR)

    img_out = segmentor.removeBG(image, background_image_resized)
    return img_out

# Функции для изменения яркости и контрастности
def change_brightness(image, value):
    value -= 40
    value = max(-50, min(value * 0.5, 50))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    v = np.clip(v, 0, 255)
    final_hsv = cv2.merge((h, s, v))
    image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return image

def change_contrast(image, value):
    value -= 40
    value = max(-50, min(value * 0.5, 50))
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.add(l, value)
    l = np.clip(l, 0, 255)
    final_lab = cv2.merge((l, a, b))
    image = cv2.cvtColor(final_lab, cv2.COLOR_LAB2BGR)
    return image

def draw_scale_with_point(scale_image, value):
    point_radius = 8
    scale_height, scale_width, _ = scale_image.shape
    position = (value + 150) / 300
    point_x = int(scale_width * position)
    point_x = int(max(0.322 * scale_width, min(point_x, 0.938 * scale_width)))
    point_y = int(scale_height * 0.846)
    cv2.circle(scale_image, (point_x, point_y), point_radius, (0, 0, 0), -1)
    return scale_image

def apply_edge_blur(image, blur_radius):
    if blur_radius <= 0:
        return image

    # Создание маски с такими же размерами, как и изображение
    mask = np.zeros_like(image)

    # Определение центра и радиуса круга
    center = (mask.shape[1] // 2, mask.shape[0] // 2)
    radius = min(center[0], center[1]) - blur_radius

    # Рисование белого круга в середине маски
    cv2.circle(mask, center, radius, (255, 255, 255), -1)

    # Инвертирование маски
    inverted_mask = cv2.bitwise_not(mask)

    # Размытие всего изображения
    blurred_image = cv2.GaussianBlur(image, (2*blur_radius+1, 2*blur_radius+1), 0)

    # Комбинирование исходного изображения и размытого изображения с использованием маски
    frame_copy = cv2.bitwise_and(image, mask) + cv2.bitwise_and(blurred_image, inverted_mask)

    return frame_copy

# Загрузка изображения
frame = cv2.imread('screenshot.png', cv2.IMREAD_UNCHANGED)

# Проверяем, что изображение было успешно загружено
if frame is None:
    print("Изображение не найдено")
    exit()

if frame.shape[2] == 4:
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

# Инициализация камеры
cap = cv2.VideoCapture(0)

# Переменные для отслеживания положения и скорости движения
prev_position = None
wheel_position = 0
velocity = 0
second_wheel_position = 0
third_wheel_position = 0
fourth_wheel_position = 0
show_second_wheel = False
show_third_wheel = False
show_fourth_wheel = False
selected_image = None

brightness_value = 40  # Текущая яркость
contrast_value = 40  # Текущая контрастность
blur_radius = 1  # Радиус размытия

positions = ['Hats', 'Bg', 'Effects', 'Result']
# Загрузка изображений высокого разрешения
image_folder = 'head'
image_paths = [f'{image_folder}/1.png', f'{image_folder}/2.png', f'{image_folder}/3.png', f'{image_folder}/4.png', f'{image_folder}/5.png']
loaded_images = [load_image(path) for path in image_paths]

resized_images = [
    resize_image(loaded_images[0], 50, 50),
    resize_image(loaded_images[1], 50, 50),
    resize_image(loaded_images[2], 75, 50),
    resize_image(loaded_images[3], 50, 50),
    resize_image(loaded_images[4], 50, 50)
]

image_folder = 'effects'
image_paths = [f'{image_folder}/1.png', f'{image_folder}/2.png', f'{image_folder}/3.png']
loaded_effect_images = [load_image(path) for path in image_paths]

effect_images = [
    resize_image(loaded_effect_images[0], 50, 50),
    resize_image(loaded_effect_images[1], 50, 50),
    resize_image(loaded_effect_images[2], 50, 75)
]

image_folder = 'background'
image_paths = [f'{image_folder}/0.png', f'{image_folder}/1.jpg', f'{image_folder}/2.jpg', f'{image_folder}/3.jpg']
loaded_bg_images = [load_image(path) for path in image_paths]

bg_images = [
    resize_image(loaded_bg_images[0], 50, 50),
    resize_image(loaded_bg_images[1], 50, 50),
    resize_image(loaded_bg_images[2], 50, 50),
    resize_image(loaded_bg_images[3], 50, 50)
]

scale_image = load_image('scale.png')
resized_scale_image = resize_image(scale_image, 400, 15)

# Инициализация MediaPipe для отслеживания рук и лица
with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as hands,mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Камера не найдена")
            continue

        # Отзеркаливание изображения по горизонтали
        image = cv2.flip(image, 1)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(image)
        face_results = face_mesh.process(frame)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        img_height, img_width, _ = image.shape

        # Копируем оригинальный кадр для обновления изображения
        frame_copy = frame.copy()

        frame_copy = change_brightness(frame_copy, brightness_value)
        frame_copy = change_contrast(frame_copy, contrast_value)

        if int(fourth_wheel_position) % len(loaded_bg_images) != 0:
            frame_copy = overlay_background(frame_copy,
                                            loaded_bg_images[int(fourth_wheel_position) % len(loaded_bg_images)])

        frame_copy = apply_edge_blur(frame_copy, blur_radius)

        handle_face_and_second_wheel(frame_copy, face_results, selected_image, img_width, img_height,
                                     resized_images)

        # Отображение точки на кончике указательного пальца
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                point_coords = (int(index_tip.x * img_width), int(index_tip.y * img_height))
                cv2.circle(frame_copy, point_coords, 10, (0, 0, 255), -1)  # Красная точка на кончике пальца

                # Проверяем, сжата ли рука в кулак и находится ли рука слева
                if is_fist(hand_landmarks.landmark):
                    print("Кулак обнаружен")
                    if point_coords[0] < frame.shape[1] // 4:
                        if prev_position is not None:
                            # формула для вычисления скорости
                            velocity = (point_coords[1] - prev_position[1]) / -100.0
                            wheel_position = update_wheel_position(wheel_position, velocity, len(positions))
                        prev_position = point_coords
                    elif show_second_wheel and point_coords[1] < frame.shape[0] // 4:
                        if prev_position is not None:
                            # формула для вычисления скорости
                            velocity = (point_coords[0] - prev_position[0]) / -100.0
                            second_wheel_position = update_wheel_position(second_wheel_position, velocity, len(loaded_images))
                        prev_position = point_coords
                        # Запоминаем выбранное изображение для отображения
                        selected_image = loaded_images[int(second_wheel_position) % len(loaded_images)]
                    elif show_third_wheel and point_coords[1] < frame.shape[0] // 4:
                        if prev_position is not None:
                            # формула для вычисления скорости
                            velocity = (point_coords[0] - prev_position[0]) / -100.0
                            third_wheel_position = update_wheel_position(third_wheel_position, velocity, len(effect_images))
                        prev_position = point_coords
                    elif show_fourth_wheel and point_coords[1] < frame.shape[0] // 4:
                        if prev_position is not None:
                            # формула для вычисления скорости
                            velocity = (point_coords[0] - prev_position[0]) / -100.0
                            fourth_wheel_position = update_wheel_position(fourth_wheel_position, velocity, len(bg_images))
                        prev_position = point_coords
                    else:
                        prev_position = None
                        velocity = 0

        # Отображение первого колеса
        wheel_image = draw_vertical_wheel(frame_copy, wheel_position, positions)
        frame_copy = overlay_panel(frame_copy, wheel_image, 0, 0, 1)

        # Если выбран режим Hats
        if int(wheel_position) % len(positions) == 0:
            show_second_wheel = True
            second_wheel_image = draw_wheel(frame_copy, second_wheel_position, resized_images)
            frame_copy = overlay_panel(frame_copy, second_wheel_image, 0, 0, 1)
        else:
            show_second_wheel = False

        # Если выбран режим Bg
        if int(wheel_position) % len(positions) == 1:
            show_fourth_wheel = True
            fourth_wheel_image = draw_wheel(frame_copy, fourth_wheel_position, bg_images)
            frame_copy = overlay_panel(frame_copy, fourth_wheel_image, 0, 0, 1)
        else:
            show_fourth_wheel = False

        # Если выбран режим Effects
        if int(wheel_position) % len(positions) == 2:
            show_third_wheel = True
            third_wheel_image = draw_wheel(frame_copy, third_wheel_position, effect_images)
            frame_copy = overlay_panel(frame_copy, third_wheel_image, 0, 0, 1)
            # Добавление изображения scale.png, если третее колесо открыто на 0 или 1
            if int(third_wheel_position) % len(effect_images) in [0, 1, 2]:
                frame_copy = overlay_panel(frame_copy, resized_scale_image, 200, 400, 1)
                # Получение положения руки для изменения параметров
                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        if is_fist(hand_landmarks.landmark) and point_coords[1] > 3 * frame.shape[0] // 4:
                            hand_position = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
                            if int(third_wheel_position) % len(effect_images) == 0:
                                brightness_value = int((hand_position * 2 - 1) * 150)
                            elif int(third_wheel_position) % len(effect_images) == 1:
                                contrast_value = int((hand_position * 2 - 1) * 150)
                            elif int(third_wheel_position) % len(effect_images) == 2:
                                blur_radius = int((hand_position * 2 - 1) * 150)
            if int(third_wheel_position) % len(effect_images) == 0:
                frame_copy = draw_scale_with_point(frame_copy, brightness_value)
            elif int(third_wheel_position) % len(effect_images) == 1:
                frame_copy = draw_scale_with_point(frame_copy, contrast_value)
            elif int(third_wheel_position) % len(effect_images) == 2:
                frame_copy = draw_scale_with_point(frame_copy, blur_radius)
        else:
            show_third_wheel = False

        cv2.imshow('view', image)
        cv2.imshow('Drawing_project', frame_copy)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()