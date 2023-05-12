import cv2
import numpy as np

# Define green color in BGR
green = (0, 255, 0)

# Initialize camera
cap = cv2.VideoCapture(0)

while True:
    # Read frame from camera
    ret, frame = cap.read()

    # Convert to grayscale and apply Gaussian blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Thresholding to segment hand region
    threshold_value, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Apply morphological operations
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(thresh, kernel, iterations=1)
    dilation = cv2.dilate(erosion, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find largest contour, which corresponds to the hand
    if len(contours) > 0:
        hand_contour = max(contours, key=cv2.contourArea)

        # Find convex hull and defects
        hull = cv2.convexHull(hand_contour, returnPoints=False)
        defects = cv2.convexityDefects(hand_contour, hull)

        # Find two farthest points in convex hull
        farthest_points = []
        for i in range(defects.shape[0]):
            s, e, d = defects[i, 0][0], defects[i, 0][1], defects[i, 0][3]
            start = tuple(hand_contour[s][0])
            end = tuple(hand_contour[e][0])
            if d > 10000:  # Only consider defects with depth > 10000
                farthest_points.append(start)
                farthest_points.append(end)

        if len(farthest_points) > 1:
            # Draw green line between two farthest points
            cv2.line(frame, farthest_points[0], farthest_points[1], green, thickness=2)

    # Display video stream
    cv2.imshow('Green Line', frame)

    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
