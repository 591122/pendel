import cv2
import numpy as np

# Define the lower and upper bounds for the green color in HSV
lower_green = np.array([35, 100, 100])
upper_green = np.array([85, 255, 255])

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Set the position of the static dot
static_dot_x = 620  # Middle of the screen in a 2560x1600 resolution
static_dot_y = 50    # Adjust the padding from the top as needed

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    
    # Convert the frame to HSV (Hue, Saturation, Value) color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create a mask to extract green regions
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Apply Gaussian blur to the mask to reduce noise
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
    # Apply morphological operations to further reduce noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Initialize variables to track the largest contour
    largest_contour = None
    largest_contour_area = 0
    
    for contour in contours:
        # Filter small contours to avoid noise
        if cv2.contourArea(contour) > 100:
            # If the current contour is larger than the previous largest
            if cv2.contourArea(contour) > largest_contour_area:
                largest_contour = contour
                largest_contour_area = cv2.contourArea(contour)
    
    # If the largest contour is found, process it
    if largest_contour is not None:
        # Find the center of the largest green object
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Print the pixel coordinates of the center
            print(f"Center Coordinates (x, y): ({cx}, {cy})")
        
            vinkel = (np.arccos((cy - static_dot_y) / np.sqrt((cx - static_dot_x)**2 + (cy - static_dot_y)**2))/np.pi)*180

            print(f"Vinkelen = " + str(vinkel))
            
            # Draw a red dot at the center of the largest object
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)  # Red color
        
        # Draw a bounding box around the largest detected green object
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Draw the static dot at the top middle of the screen
    cv2.circle(frame, (static_dot_x, static_dot_y), 5, (0, 0, 255), -1)  # Red color
    
    # Display the original frame with the center dot, bounding box, and static dot
    cv2.imshow('Green Object Detection', frame)
    
    # Break the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
