import time
from cvzone.HandTrackingModule import HandDetector
from pynput.mouse import Controller
import cv2

mouse = Controller()

cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=1)
while True:
    # Get image frame
    success, img = cap.read()
    # Find the hand and its landmarks
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img)  # with draw
    # hands = detector.findHands(img, draw=False)  # without draw

    if hands:
        # Hand 1
        hand1 = hands[0]
        lmList1 = hand1["lmList"]  # List of 21 Landmark points
        bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h
        centerPoint1 = hand1['center']  # center of the hand cx,cy
        handType1 = hand1["type"]  # Handtype Left or Right

        fingers1 = detector.fingersUp(hand1)

        if fingers1[1] == 1 and fingers1[2] != 1 and fingers1[3] != 1 and fingers1[4] != 1 or fingers1[1] == 0 and fingers1[2] != 0 and fingers1[3] != 0 and fingers1[4] != 0:
            mouse.position = (lmList1[8][0], lmList1[8][1])

            cv2.putText(img, text=str(lmList1[8]), org=(10, 160), 
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(255,255,255), thickness=2)

    # Display
    cv2.imshow("Image", img)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
cap.release()
cv2.destroyAllWindows()