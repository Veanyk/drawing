import cv2
import mediapipe as mp
import numpy as np

# Load the ears image with alpha channel
ears_image_path = '1.png'
ears_img = cv2.imread(ears_image_path, cv2.IMREAD_UNCHANGED)

# Check if ears image was loaded successfully
if ears_img is None:
    print(f"Error: could not load ears image from {ears_image_path}")
    exit()

# Initialize MediaPipe for face mesh
mp_face_mesh = mp.solutions.face_mesh

# Function to overlay image with alpha channel
def overlay_image_alpha(img, img_overlay, x, y, alpha_mask):
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return img

    img_crop = img[y1:y2, x1:x2]
    img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
    alpha = alpha_mask[y1o:y2o, x1o:x2o, np.newaxis] / 255.0

    img_crop[:] = alpha * img_overlay_crop + (1 - alpha) * img_crop
    return img

# Load the screenshot image
screenshot_path = 'screenshot.png'
screenshot = cv2.imread(screenshot_path)

# Check if screenshot image was loaded successfully
if screenshot is None:
    print(f"Error: could not load screenshot image from {screenshot_path}")
    exit()

# Function to get ear coordinates
def get_ears_coordinates(face_landmarks, img_width, img_height):
    left_ear_top = face_landmarks[127]
    right_ear_top = face_landmarks[356]

    left_ear_top = (int(left_ear_top.x * img_width), int(left_ear_top.y * img_height))
    right_ear_top = (int(right_ear_top.x * img_width), int(right_ear_top.y * img_height))

    return left_ear_top, right_ear_top

# Initialize MediaPipe Face Mesh
with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
    screenshot_rgb = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(screenshot_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            img_height, img_width, _ = screenshot.shape

            left_ear_top, right_ear_top = get_ears_coordinates(face_landmarks.landmark, img_width, img_height)

            ear_width = int(0.5 * (right_ear_top[0] - left_ear_top[0]))
            ear_height = int(ears_img.shape[0] * (ear_width / ears_img.shape[1]))

            resized_ears_img = cv2.resize(ears_img, (ear_width, ear_height), interpolation=cv2.INTER_AREA)

            alpha_mask = resized_ears_img[:, :, 3]
            ears_img_rgb = resized_ears_img[:, :, :3]

            # Positioning the ears image
            y_offset = int(left_ear_top[1] - ear_height / 2)
            x_offset = left_ear_top[0] - int(ear_width / 2)

            screenshot = overlay_image_alpha(screenshot, ears_img_rgb, x_offset, y_offset, alpha_mask)

        cv2.imwrite('result.png', screenshot)
        print("Result saved as final_result.png")

cv2.imshow('Ears Overlay', screenshot)
cv2.waitKey(0)
cv2.destroyAllWindows()
