import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
accumulator_frame = None
capture_image = False
image_counter = 0

while True:
    ret, frame = cap.read()
    fs = frame.shape
    flipped = cv.flip(frame, 1)

    # Convert to RGB color space
    rgb = cv.cvtColor(flipped, cv.COLOR_BGR2RGB)

    red_mask = cv.inRange(rgb, (150, 0, 0), (255, 100, 50))  # Threshold for red color

    if accumulator_frame is None:
        accumulator_frame = np.zeros(fs, dtype=np.uint8)

    contours, _ = cv.findContours(red_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if len(contours) >= 2:
        cnt1, cnt2 = contours[:2]
        moments1 = cv.moments(cnt1)
        moments2 = cv.moments(cnt2)

        if moments1['m00'] > 0 and moments2['m00'] > 0:
            cx1 = int(moments1['m10'] / moments1['m00'])
            cy1 = int(moments1['m01'] / moments1['m00'])
            cx2 = int(moments2['m10'] / moments2['m00'])
            cy2 = int(moments2['m01'] / moments2['m00'])

            dist = np.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2)

            if dist < 50:
                capture_image = True

    if capture_image:
        cv.imwrite(f'finger_image_{image_counter}.jpg', flipped)
        print(f"Image saved: finger_image_{image_counter}.jpg")
        image_counter += 1
        capture_image = False

    cv.drawContours(accumulator_frame, contours, -1, (0, 255, 255), thickness=2)
    cv.imshow('Input', flipped)
    cv.imshow('Red Mask', red_mask)
    cv.imshow('Time-Lapse', accumulator_frame)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
