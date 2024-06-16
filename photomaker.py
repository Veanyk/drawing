import time
import numpy as np
import cv2
from cvzone.SelfiSegmentationModule import SelfiSegmentation

MAX_HEIGHT = 1080
MAX_WIDTH = 1920


if __name__ == "__main__":
    capture = cv2.VideoCapture(0)
    ret, frame = capture.read()
    if not ret:
        print("Couldn't open the camera")
        exit(-1)
    # background_picture = cv2.resize(frame, (MAX_WIDTH, MAX_HEIGHT))
    # photo = np.full((MAX_HEIGHT, MAX_WIDTH, 3), 255, dtype=np.uint8)
    seconds_before_first_background_picture = 5
    seconds_before_foreground_picture = 3
    seconds_between_background_pictures = 60
    timestamp_enter = timestamp_last_background_picture = time.time()
    timestamp_got_photo = -1
    got_first_background_picture = False
    got_photo = False
    backSub = cv2.createBackgroundSubtractorKNN()
    mycode = False
    cvcv = False
    segmentor = SelfiSegmentation()

    while True:
        ret, frame = capture.read()
        if not ret:
            print("Couldn't open the camera")
            exit(-1)

        if mycode:
            if time.time() - timestamp_enter > seconds_before_first_background_picture and not got_first_background_picture:
                # background_picture = cv2.resize(frame, (MAX_WIDTH, MAX_HEIGHT))
                background_picture = frame
                # kernel = np.ones((5, 5), np.uint8)
                # cv2.dilate(background_picture, kernel, iterations=1)
                # kernel = np.ones((5, 5), np.uint8)
                # cv2.erode(background_picture, kernel, iterations=1)
                # cv2.medianBlur(background_picture, 3)
                got_first_background_picture = True
                timestamp_last_background_picture = time.time()
                # background_picture[0][0][0] = 255
                print("Made background photo")
                print("back: ", background_picture)
                # cv2.imshow("Back", background_picture)

            if (got_first_background_picture and not got_photo and
                    time.time() - timestamp_last_background_picture > seconds_before_foreground_picture):
                # photo = cv2.resize(frame, (MAX_WIDTH, MAX_HEIGHT))
                photo = frame
                # kernel = np.ones((5, 5), np.uint8)
                # cv2.dilate(photo, kernel, iterations=1)
                # kernel = np.ones((5, 5), np.uint8)
                # cv2.erode(photo, kernel, iterations=1)
                # cv2.medianBlur(photo, 3)
                # photo[-1][-1][0] = 255
                timestamp_got_photo = time.time()
                got_photo = True
                # transformed_photo = np.bitwise_xor(photo, background_picture)
                th = 5
                diff = cv2.absdiff(background_picture, photo)
                # diff_mask = diff > th
                # new_diff = np.zeros_like(photo, np.uint8)
                # new_diff[diff_mask] = diff[diff_mask]
                # mask = cv2.cvtColor(new_diff, cv2.COLOR_BGR2GRAY)

                mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                imask = mask > th

                transformed_photo = np.zeros_like(photo, np.uint8)
                transformed_photo[imask] = photo[imask]
                print("photo: ", photo)
                print("mask: ", mask)
                print("imask: ", imask)
                print("transform: ", transformed_photo)
                # cv2.imshow("Fore", photo)

            if not got_photo:
                # cv2.imshow("Drawing", cv2.resize(frame, (MAX_WIDTH, MAX_HEIGHT)))
                cv2.imshow("Drawing", frame)
            else:
                if time.time() - timestamp_got_photo < 2:
                    cv2.imshow("Drawing", photo)
                else:
                    cv2.imshow("Drawing", transformed_photo)
        else:
            if cvcv:
                if time.time() - timestamp_enter < 5:
                    fgMask = backSub.apply(frame)

                if time.time() - timestamp_enter > 5 and not got_first_background_picture:
                    got_first_background_picture = True
                    timestamp_last_background_picture = time.time()
                    print("Got model")
                if time.time() - timestamp_last_background_picture > 3 and not got_photo and got_first_background_picture:
                    # backSub2 = backSub
                    fgMask = backSub.apply(frame)
                    got_photo = True
                cv2.imshow('Frame', frame)
                cv2.imshow('FG Mask', fgMask)
            else:
                img_Out = segmentor.removeBG(frame, (255, 255, 255), cutThreshold=0.2)
                cv2.imshow('Frame', img_Out)

        if cv2.waitKey(1) & 0xFF == "q":
            break

    cv2.destroyAllWindows()
    capture.release()
