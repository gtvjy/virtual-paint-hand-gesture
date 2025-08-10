
import cv2
import numpy as np
import mediapipe as mp

# Step 1: Take name input
name = input("👋 Welcome! Please enter your name to start drawing: ")
print(f"✅ Hello {name}, launching Virtual Paint...")

# Initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

# Default brush color and mode
draw_color = (255, 0, 255)
mode = "Draw"
brush_thickness = 10
eraser_thickness = 50

# Canvas
canvas = None

# Video
cap = cv2.VideoCapture(0)

# Previous position
x_prev, y_prev = 0, 0

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    if canvas is None:
        canvas = np.zeros_like(img)

    # Detect hands
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lmList = []
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((cx, cy))

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            # Get index & middle finger tips
            index_finger = lmList[8]
            middle_finger = lmList[12]
            x1, y1 = index_finger

            # Drawing Mode if fingers apart
            dist = np.linalg.norm(np.array(index_finger) - np.array(middle_finger))
            if dist > 40:
                if x_prev == 0 and y_prev == 0:
                    x_prev, y_prev = x1, y1

                if mode == "Draw":
                    cv2.line(canvas, (x_prev, y_prev), (x1, y1), draw_color, brush_thickness)
                    cv2.circle(img, (x1, y1), brush_thickness // 2, draw_color, 2)
                elif mode == "Erase":
                    cv2.line(canvas, (x_prev, y_prev), (x1, y1), (0, 0, 0), eraser_thickness)
                    cv2.circle(img, (x1, y1), eraser_thickness // 2, (0, 0, 0), 2)

                x_prev, y_prev = x1, y1
            else:
                x_prev, y_prev = 0, 0

    # Combine canvas and webcam
    img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, img_inv)
    img = cv2.bitwise_or(img, canvas)

    # Show user info and mode
    cv2.putText(img, f'User: {name}', (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(img, f'Mode: {mode}', (10, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(img, f'Brush Size: {brush_thickness}', (10, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Draw selected color indicator
    cv2.circle(img, (580, 30), 20, draw_color if mode == "Draw" else (0, 0, 0), -1)
    cv2.putText(img, 'Color', (540, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # === Tooltips ===
    tooltip_color = (200, 255, 200)
    cv2.putText(img, '[Q] Quit', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, tooltip_color, 1)
    cv2.putText(img, '[S] Save Drawing', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, tooltip_color, 1)
    cv2.putText(img, '[R] Red  [G] Green  [B] Blue  [K] Black', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, tooltip_color, 1)
    cv2.putText(img, '[E] Eraser  [+/-] Brush Size', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, tooltip_color, 1)

    # Show final output
    cv2.imshow("Virtual Paint", img)

    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('+') or key == ord('='):
        brush_thickness = min(brush_thickness + 2, 50)
        eraser_thickness = min(eraser_thickness + 5, 100)
    elif key == ord('-') or key == ord('_'):
        brush_thickness = max(2, brush_thickness - 2)
        eraser_thickness = max(10, eraser_thickness - 5)
    elif key == ord('s'):
        filename = f"{name}_drawing.png"
        cv2.imwrite(filename, canvas)
        print(f"🖼 Drawing saved as {filename}")
        saved_img = cv2.imread(filename)
        if saved_img is not None:
            cv2.imshow("🖼 Your Saved Drawing", saved_img)
        else:
            print("❌ Failed to load saved image.")
    elif key == ord('r'):
        draw_color = (0, 0, 255)
        mode = "Draw"
    elif key == ord('g'):
        draw_color = (0, 255, 0)
        mode = "Draw"
    elif key == ord('b'):
        draw_color = (255, 0, 0)
        mode = "Draw"
    elif key == ord('k'):
        draw_color = (0, 0, 0)
        mode = "Draw"
    elif key == ord('e'):
        mode = "Erase"

# Cleanup
cap.release()
cv2.destroyAllWindows()