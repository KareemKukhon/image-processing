import cv2

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    x, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY_INV)
    contours, y = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        hand_contour = max(contours, key=cv2.contourArea)
        topmost = tuple(hand_contour[hand_contour[:,:,1].argmin()][0])
        if frame[topmost[1], topmost[0], 2] > 110 and frame[topmost[1], topmost[0], 1] < 150 and frame[topmost[1], topmost[0], 0] < 150:
            cv2.line(frame, (topmost[0]-30, topmost[1]), (topmost[0]+30, topmost[1]), (0, 0, 255), thickness=2)
    cv2.imshow('liveStream', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
