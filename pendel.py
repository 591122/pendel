import cv2
import numpy as np
import time
import csv
import datetime
import os

# Define the lower and upper bounds for the red color in HSV
lower_red = np.array([0, 100, 100])
upper_red = np.array([5, 255, 255])

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Set the position of the static dot
static_dot_x = 960  # Middle of the screen in a 2560x1600 resolution
static_dot_y = 80    # Adjust the padding from the top as needed

# Initialize variables to track the recording state, data, and start time
recording = False
data = []
start_time = 0
entries = -1

# Get the frames per second (fps) of the camera
fps = int(cap.get(cv2.CAP_PROP_FPS))
print(f"Camera is capturing at {fps} fps")

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    
    # Convert the frame to HSV (Hue, Saturation, Value) color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create a mask to extract green regions
    mask = cv2.inRange(hsv, lower_red, upper_red)
    
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
            if(cx>=620):
                vinkel = (np.arccos((cy - static_dot_y) / np.sqrt((cx - static_dot_x)**2 + (cy - static_dot_y)**2))/np.pi)*180
            else:
                vinkel = -(np.arccos((cy - static_dot_y) / np.sqrt((cx - static_dot_x)**2 + (cy - static_dot_y)**2))/np.pi)*180

            print(f"Vinkelen = " + str(vinkel))
            
            # Record data if recording is active
            if recording:
                # Calculate elapsed time in seconds
                current_time = time.time() - start_time

                if(current_time is 0):
                    vinkel_hastighet = 0
                else:
                    if entries >= 0:
                        vinkel_hastighet = (vinkel - data[entries][3]) / (current_time - data[entries][2])
                    else:
                        vinkel_hastighet = 0

                data.append((cx, cy, current_time, vinkel, vinkel_hastighet))
                entries = entries + 1
        
        # Draw a bounding box around the largest detected green object
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw the red dot at the center of the box
        center_x = x + w // 2
        center_y = y + h // 2
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
    
    # Draw the static dot at the top middle of the screen
    cv2.circle(frame, (static_dot_x, static_dot_y), 5, (0, 0, 255), -1)  # Red color

    # Display the frames per second (fps) in one of the corners
    cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the original frame with the center dot, bounding box, and static dot
    cv2.imshow('Red Object Detection', frame)
    
    # Check for key presses
    key = cv2.waitKey(1) & 0xFF
    
    # Start/stop recording when 'p' key is pressed
    if key == ord('p'):
        if recording:
            recording = False
        else:
            recording = True
            start_time = time.time()  # Record the start time
    
    # Break the loop when the 'q' key is pressed
    if key == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Print the recorded data with time in seconds
print("Recorded Data:")
x = str(datetime.datetime.now())
dato = x[:10]
timer_min = x[11:19]
for entry in data:
    print(f"Position: x:{entry[0]}, y:{entry[1]}, Time (s): {entry[2]:.3f}, vikelen er {entry[3]:.3f} og vinkelhastighet: {entry[4]:.3f}"  )
if(data[0] is not None):
    # Specify the folder name
    folder_name = "recorded_data"

    # Create the folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Specify the file path with the folder
    file_path = os.path.join(folder_name, dato + "_" + timer_min + '.csv')
    with open(file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Position_X', 'Position_Y', 'Time (s)', 'Vinkel', 'Vinkel_hastighet'])
        for entry in data:
            csv_writer.writerow([entry[0], entry[1], entry[2], entry[3], entry[4]])


x = str(datetime.datetime.now())
dato = x[:10]
timer_min = x[11:19]

#print(dato+"_"+timer_min)

print('Data saved as ' + dato + '_' + timer_min + '.csv')
