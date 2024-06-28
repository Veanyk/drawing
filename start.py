import cv2
import mediapipe as mp
import time
import subprocess
import os
import sys

# Загрузка изображения панельки
panel_img = cv2.imread('panel.png', cv2.IMREAD_UNCHANGED)

# Функция для наложения панельки
def overlay_panel(frame, panel_img, x, y, scale=1):
    panel_img_resized = cv2.resize(panel_img, (0, 0), fx=scale, fy=scale)
    panel_h, panel_w, panel_c = panel_img_resized.shape

    if panel_c == 4:
        alpha_channel = panel_img_resized[:, :, 3]
        rgb_channels = panel_img_resized[:, :, :3]
        alpha_mask = cv2.merge([alpha_channel, alpha_channel, alpha_channel])

        roi = frame[y:y+panel_h, x:x+panel_w]
        masked_frame = cv2.bitwise_and(roi, cv2.bitwise_not(alpha_mask))
        masked_panel = cv2.bitwise_and(rgb_channels, alpha_mask)
        overlay = cv2.add(masked_frame, masked_panel)
        frame[y:y+panel_h, x:x+panel_w] = overlay
    else:
        frame[y:y+panel_h, x:x+panel_w] = panel_img_resized
    return frame

# Инициализация mediapipe для отслеживания рук
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Функция для вычисления расстояния между двумя точками
def calculate_distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

# Определение прямоугольника
def is_rectangle(landmarks_left, landmarks_right):
    left_index_tip = landmarks_left[8]  # Кончик указательного пальца левой руки
    left_thumb_tip = landmarks_left[4]  # Кончик большого пальца левой руки
    right_index_tip = landmarks_right[8]  # Кончик указательного пальца правой руки
    right_thumb_tip = landmarks_right[4]  # Кончик большого пальца правой руки

    width = calculate_distance(left_index_tip, right_thumb_tip)
    height = calculate_distance(right_index_tip, left_thumb_tip)

    return width < 25 and height < 25

# Функция для определения руки относительно кругов
def is_hand_in_circle(landmarks, circle_center, circle_radius):
    index_tip = landmarks[8]
    distance = calculate_distance(index_tip, circle_center)
    return distance < circle_radius

# Функция для закрашивания круга снизу вверх
def draw_count(image, center, radius, elapsed_time, max_time, colors):
    if elapsed_time >= max_time:
        cv2.circle(image, center, radius, colors[-1], thickness=-1)
    else:
        num_colors = len(colors)
        sector_time = max_time / num_colors
        current_color_index = int(elapsed_time // sector_time)
        angle = int(((elapsed_time % sector_time) / sector_time) * 360)
        axes = (radius, radius)

        for i in range(current_color_index):
            cv2.ellipse(image, center, axes, 0, 0, 360, colors[i], thickness=-1)

        cv2.ellipse(image, center, axes, 0, 0, angle, colors[current_color_index], thickness=-1)

# Функция для закрашивания круга
def draw_filled_circle(image, center, radius, elapsed_time, max_time, color):
    if elapsed_time >= max_time:
        cv2.circle(image, center, radius, color, thickness=-1)
    else:
        angle = int((elapsed_time / max_time) * 360)
        axes = (radius, radius)
        cv2.ellipse(image, center, axes, 0, 0, angle, color, thickness=-1)

# Инициализация камеры
cap = cv2.VideoCapture(0)
countdown = 5
start_time = None
screenshot_taken = False
frame = None
left_circle_start_time = None
right_circle_start_time = None
cooldown_start_time = None

# Цвета для градиентной заливки круга
cooldown_colors = [(255, 255, 255), (128, 128, 128), (255, 255, 255), (128, 128, 128), (255, 255, 255)]

with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Камера не найдена")
            continue

        # Отзеркаливание изображения по горизонтали
        image = cv2.flip(image, 1)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        img_height, img_width, _ = image.shape

        # Положение панельки
        panel_center_left = (85, img_height - 62)
        panel_center_right = (553, img_height - 62)
        circle_radius = 47

        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2 and not screenshot_taken:
            landmarks_left = [(lm.x * img_width, lm.y * img_height) for lm in results.multi_hand_landmarks[0].landmark]
            landmarks_right = [(lm.x * img_width, lm.y * img_height) for lm in results.multi_hand_landmarks[1].landmark]

            if is_rectangle(landmarks_left, landmarks_right):
                if start_time is None:
                    start_time = time.time()
                    print("Прямоугольник обнаружен")

        if start_time is not None and not screenshot_taken:
            elapsed_time = time.time() - start_time
            remaining_time = countdown - elapsed_time
            if remaining_time > 0:
                draw_count(image, (img_width // 2, 70), circle_radius, elapsed_time, countdown,
                                   cooldown_colors)
                cv2.putText(image, f"{int(remaining_time) + 1}", (img_width // 2 - 20, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
            else:
                ret, frame = cap.read()
                # Отзеркаливание изображения по горизонтали
                frame = cv2.flip(frame, 1)
                if ret:
                    cv2.imwrite("screenshot.png", frame)
                    print("Фотография сохранена")
                    screenshot_taken = True
                    start_time = None
                    cooldown_start_time = time.time()
                    countdown = 5

        if screenshot_taken:
            # Копируем оригинальный кадр для обновления изображения
            frame_copy = frame.copy()

            # Отображение панели внизу изображения
            panel_h, panel_w, _ = panel_img.shape
            image = overlay_panel(frame_copy, panel_img, 0, img_height - panel_h, scale=1)

            # Отображение точки на кончике указательного пальца
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    point_coords = (int(index_tip.x * img_width), int(index_tip.y * img_height))
                    cv2.circle(image, point_coords, 10, (0, 0, 255), -1)  # Красная точка на кончике пальца

        if results.multi_hand_landmarks and screenshot_taken:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = [(lm.x * img_width, lm.y * img_height) for lm in hand_landmarks.landmark]

                if is_hand_in_circle(landmarks, panel_center_left, circle_radius):
                    if left_circle_start_time is None:
                        left_circle_start_time = time.time()
                    elapsed_left_time = time.time() - left_circle_start_time
                    draw_filled_circle(image, panel_center_left, circle_radius, elapsed_left_time, 2, (0, 0, 255))
                else:
                    left_circle_start_time = None

                if is_hand_in_circle(landmarks, panel_center_right, circle_radius):
                    if right_circle_start_time is None:
                        right_circle_start_time = time.time()
                    elapsed_right_time = time.time() - right_circle_start_time
                    draw_filled_circle(image, panel_center_right, circle_radius, elapsed_right_time, 2, (0, 255, 0))
                else:
                    right_circle_start_time = None

                if left_circle_start_time is not None and (time.time() - left_circle_start_time) > 2:
                    screenshot_taken = False
                    print("Перезапуск программы")
                    left_circle_start_time = None
                    break
                elif right_circle_start_time is not None and (time.time() - right_circle_start_time) > 2:
                    print("Запуск основной программы.")
                    cap.release()
                    cv2.destroyAllWindows()
                    venv_python = os.path.join(sys.prefix, 'Scripts', 'python.exe')
                    subprocess.run([venv_python, "main.py"])
                    exit()

            # Отображение панели внизу изображения
            image = overlay_panel(frame_copy, panel_img, 0, img_height - panel_h, scale=1)

        cv2.imshow('Drawing_project', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
