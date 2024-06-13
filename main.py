import cv2
import mediapipe as mp

# Инициализация mediapipe для отслеживания рук
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


# Функция для вычисления координат указательного пальца
def get_index_finger_tip(landmarks):
    return (int(landmarks[8].x * image.shape[1]), int(landmarks[8].y * image.shape[0]))


# Чтение сохраненного изображения
image = cv2.imread('screenshot.png')
drawing_image = image.copy()  # Копия для рисования

# Инициализация камеры (если необходимо для рисования в реальном времени)
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as hands:
    while True:
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Переводим изображение в RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame)

        # Возвращаем изображение в BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                index_finger_tip = get_index_finger_tip(hand_landmarks.landmark)

                # Рисование на изображении
                cv2.circle(drawing_image, index_finger_tip, 5, (0, 255, 0), -1)  # Зеленый кружок

        # Отображение изображения с рисованием
        cv2.imshow('Drawing', drawing_image)

        if cv2.waitKey(1) & 0xFF == 27:  # Нажмите ESC для выхода
            break

cap.release()
cv2.destroyAllWindows()
