import cv2
import mediapipe as mp
import time
import subprocess
import os
import sys

# Инициализация mediapipe для отслеживания рук
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Функция для вычисления расстояния между двумя точками
def calculate_distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

# Определение прямоугольного жеста
def is_rectangle(landmarks_left, landmarks_right):
    left_index_tip = landmarks_left[8]  # Кончик указательного пальца левой руки
    left_thumb_tip = landmarks_left[4]  # Кончик большого пальца левой руки
    right_index_tip = landmarks_right[8]  # Кончик указательного пальца правой руки
    right_thumb_tip = landmarks_right[4]  # Кончик большого пальца правой руки

    width = calculate_distance(left_index_tip, right_thumb_tip)
    height = calculate_distance(right_index_tip, left_thumb_tip)

    return width < 25 and height < 25

# Определение жеста для пересъемки (рука слева или справа)
def is_hand_on_left_or_right(landmarks, img_width):
    index_tip_x = landmarks[8][0]
    if index_tip_x < img_width * 0.1:
        return "left"
    elif index_tip_x > img_width * 0.9:
        return "right"
    return None

# Функция для отображения текста с фоном
def draw_text(image, text, position, font_scale=1, color=(0, 0, 255), thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x, text_y = position
    box_coords = ((text_x - 5, text_y + 5), (text_x + text_size[0] + 5, text_y - text_size[1] - 5))
    cv2.rectangle(image, box_coords[0], box_coords[1], (255, 255, 255), cv2.FILLED)
    cv2.putText(image, text, position, font, font_scale, color, thickness)

# Инициализация камеры
cap = cv2.VideoCapture(0)
# Счетчик для отсчета времени перед скриншотом
countdown = 5
start_time = None
screenshot_taken = False
hand_start_time = None
reshoot_start_time = None
frame = None

with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        img_height, img_width, _ = image.shape

        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2 and not screenshot_taken:
            landmarks_left = [(lm.x * img_width, lm.y * img_height) for lm in results.multi_hand_landmarks[0].landmark]
            landmarks_right = [(lm.x * img_width, lm.y * img_height) for lm in results.multi_hand_landmarks[1].landmark]

            if is_rectangle(landmarks_left, landmarks_right):
                if start_time is None:
                    start_time = time.time()
                    print("Rectangle gesture detected! Countdown started.")

        if start_time is not None and not screenshot_taken:
            elapsed_time = time.time() - start_time
            remaining_time = countdown - int(elapsed_time)
            if remaining_time > 0:
                print(f"Countdown: {remaining_time}")
                draw_text(image, f"Time left: {remaining_time}s", (50, 50))
            else:
                ret, frame = cap.read()
                if ret:
                    cv2.imwrite("screenshot.png", frame)
                    print("Screenshot saved.")
                    screenshot_taken = True
                    start_time = None
                    countdown = 5

        if screenshot_taken:
            image = frame

        if results.multi_hand_landmarks and screenshot_taken:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = [(lm.x * img_width, lm.y * img_height) for lm in hand_landmarks.landmark]

                hand_position = is_hand_on_left_or_right(landmarks, img_width)
                if hand_position == "right":
                    if reshoot_start_time is None:
                        reshoot_start_time = time.time()
                    else:
                        elapsed_reshoot_time = time.time() - reshoot_start_time
                        draw_text(image, f"Reshoot: {elapsed_reshoot_time:.1f}s", (50, 100))
                        if elapsed_reshoot_time > 2:
                            screenshot_taken = False
                            print("Reshoot gesture detected. Ready for new screenshot.")
                            reshoot_start_time = None
                            break

                elif hand_position == "left":
                    if hand_start_time is None:
                        hand_start_time = time.time()
                    else:
                        elapsed_hand_time = time.time() - hand_start_time
                        draw_text(image, f"Exit: {elapsed_hand_time:.1f}s", (50, 150))
                        if elapsed_hand_time > 2:
                            print("Send screenshot gesture detected. Exiting program.")
                            cap.release()
                            cv2.destroyAllWindows()
                            venv_python = os.path.join(sys.prefix, 'Scripts', 'python.exe')
                            subprocess.run([venv_python, "main.py"])
                            exit()
                else:
                    hand_start_time = None

        cv2.imshow('MediaPipe Hands', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
