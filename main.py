import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# Инициализация mediapipe для отслеживания рук и лица
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

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
def update_wheel_position(position, velocity):
    position += velocity
    position %= 4  # Всего позиций на колесе
    return position

# Функция для отрисовки вертикального колеса
def draw_vertical_wheel(image, position):
    positions = ['Hats', 'Bg', 'Effects', 'Result']
    center_index = int(position) % len(positions)
    wheel_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i in range(-2,3):
        pos_index = (center_index + i) % len(positions)
        text = positions[pos_index]
        if i == 0:
            color = (0, 255, 0, 255)
            scale = 1.5
            thickness = 3
            x_offset = 70
        elif abs(i) == 1:
            color = (128, 128, 128, 255)
            scale = 1.25
            thickness = 2
            x_offset = 50
        else:
            color = (128, 128, 128, 255)
            scale = 1
            thickness = 1
            x_offset = 30

        y_pos = image.shape[0] // 2 + i * 80
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

        y_pos = y_offset
        x_pos = image.shape[1] // 2 + i * 100

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
def handle_face_and_second_wheel(combined_image, face_results, selected_image, img_width, img_height,images):
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

            # Вычисление расстояния между подбородком и носом
            distance = np.sqrt((chin_x - nose_x) ** 2 + (chin_y - nose_y) ** 2)

            # Определение верхушки головы
            dx = nose_x - chin_x
            dy = nose_y - chin_y
            if int(second_wheel_position) % len(images) == 2:
                head_top_x = chin_x + int(1.75 * dx)
                head_top_y = chin_y + int(1.75 * dy)
            elif int(second_wheel_position) % len(images) == 4:
                head_top_x = chin_x + int(2.25 * dx)
                head_top_y = chin_y + int(2.25 * dy)
            else:
                head_top_x = chin_x + int(2 * dx)
                head_top_y = chin_y + int(2 * dy)

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
                combined_image = overlay_hat(combined_image, rotated_hat, hat_x1, hat_y1)
            else:
                print(f"Invalid hat dimensions: width = {hat_width}, height = {hat_height}")

        return combined_image

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
show_second_wheel = False
display_large = False
selected_image = None

# Загрузка изображений высокого разрешения
image_folder = 'head'
image_paths = [f'{image_folder}/1.png', f'{image_folder}/2.png', f'{image_folder}/3.png', f'{image_folder}/4.png', f'{image_folder}/5.png']
loaded_images = [load_image(path) for path in image_paths]

# Создание уменьшенных изображений для колеса
resized_images = [
    resize_image(loaded_images[0], 50, 50),
    resize_image(loaded_images[1], 50, 50),
    resize_image(loaded_images[2], 75, 50),
    resize_image(loaded_images[3], 50, 50),
    resize_image(loaded_images[4], 50, 50)
]

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
                            wheel_position = update_wheel_position(wheel_position, velocity)
                        prev_position = point_coords
                    else:
                        if show_second_wheel and point_coords[1] < img_height // 4:
                            if prev_position is not None:
                                # формула для вычисления скорости
                                velocity = (point_coords[0] - prev_position[0]) / -100.0
                                second_wheel_position = update_wheel_position(second_wheel_position, velocity)
                            prev_position = point_coords
                            # Запоминаем выбранное изображение для отображения
                            selected_image = loaded_images[int(second_wheel_position) % len(loaded_images)]
                            display_large = True
                else:
                    prev_position = None
                    velocity = 0

        # Отображение первого колеса
        wheel_image = draw_vertical_wheel(frame_copy, wheel_position)
        combined_image = overlay_panel(frame_copy, wheel_image, 0, 0, 1)

        # Если выбрана единица, отображаем второе колесо
        if int(wheel_position) % len(resized_images) == 0:
            show_second_wheel = True
            second_wheel_image = draw_wheel(frame_copy, second_wheel_position, resized_images)
            combined_image = overlay_panel(combined_image, second_wheel_image, 0, 0, 1)
        else:
            show_second_wheel = False

        handle_face_and_second_wheel(combined_image, face_results, selected_image, img_width, img_height, resized_images)

        cv2.imshow('view', image)
        cv2.imshow('Drawing_project', combined_image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()