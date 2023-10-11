import cv2
import numpy as np
import time

# Create a VideoCapture object to access the webcam
cap = cv2.VideoCapture(0)  # 0 is the default camera

last_print_time = time.time()

while True:
    ret, frame = cap.read()  # Read a frame from the webcam

    if not ret:
        break

    # Convert the frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for red and orange in HSV
    lower_red = np.array([0, 100, 100])     # Lower bound for red
    upper_red = np.array([10, 255, 255])    # Upper bound for red
    lower_orange = np.array([11, 100, 100])  # Lower bound for orange
    upper_orange = np.array([30, 255, 255])  # Upper bound for orange

    # Create masks for both red and orange
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)

    # Combine the masks to detect both red and orange
    mask = cv2.bitwise_or(mask_red, mask_orange)

    # Count the number of non-zero pixels in the mask
    non_zero_pixels = cv2.countNonZero(mask)

    # Determine the detected color
    if non_zero_pixels > 1000:  # Adjust this threshold as needed
        color = "Red or Orange"
        hsv_values = hsv[np.nonzero(mask)]
        hsv_values = np.mean(hsv_values, axis=0)
        current_time = time.time()
        if current_time - last_print_time >= 3:
            print(f"Detected Color: {color}")
            print(f"HSV Values: {hsv_values}")
            last_print_time = current_time
    else:
        color = "No Color Detected"

    # Display the original frame
    cv2.imshow('Original', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
