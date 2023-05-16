import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
zoom_factor = 1.0

while True:
    ret, frame = cap.read()
    fs = frame.shape
    flipped = cv.flip(frame, 1)

    # Convert to RGB color space
    rgb = cv.cvtColor(flipped, cv.COLOR_BGR2RGB)

    red_mask = cv.inRange(rgb, (120, 0, 0), (255, 70, 100))  # Threshold for red color

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

            # Adjust zoom factor based on distance
            zoom_factor = max(1.0, dist / 100)

    # Resize frame based on zoom factor
    resized_frame = cv.resize(flipped, None, fx=zoom_factor, fy=zoom_factor)

    cv.imshow('Zoom', resized_frame)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
