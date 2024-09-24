import cv2
import numpy as np
import mysql.connector
import os

# Constants
REAL_WORLD_DISTANCE = 8.0  # Distance in meters between the lines in the real world
LINE_1_Y = 220  # Y-coordinate for the first line (upper side)
LINE_2_Y = 330  # Y-coordinate for the second line (upper side)
RESIZE_FACTOR = 0.6  # Resize factor for the frames
LINE_THICKNESS = 1  # Thickness of the line for visualization
CONTOUR_AREA_THRESHOLD = 1350  # Minimum contour area to be considered a vehicle
SAVE_IMAGE_PATH = "captured_images"  # Directory to save captured images

# Create the directory if it doesn't exist
if not os.path.exists(SAVE_IMAGE_PATH):
    os.makedirs(SAVE_IMAGE_PATH)

# Establish a connection to the MySQL database
mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="atcc"
)
# Create a cursor object to execute SQL queries
mycursor = mydb.cursor()

# Initialize video capture
cap = cv2.VideoCapture("test.mp4")

# Get video frame rate for accurate timing
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30  # Assume a default fps if it cannot be determined

# Initialize the background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

# Initialize variables for tracking speed
vehicle_id_count = 0
timing_dict = {}
speed_dict = {}
vehicle_positions = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame
    frame = cv2.resize(frame, None, fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)

    # Apply background subtraction
    fg_mask = bg_subtractor.apply(frame)

    # Threshold the mask to get binary image
    _, thresh = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    current_vehicles = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > CONTOUR_AREA_THRESHOLD:
            x, y, w, h = cv2.boundingRect(contour)
            center_y = y + h // 2
            center_x = x + w // 2

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Associate a unique ID for the vehicle
            vehicle_id = None
            for vid, pos in vehicle_positions.items():
                if abs(pos - center_y) < 50:  # Check proximity to assign the same vehicle ID
                    vehicle_id = vid
                    break

            if vehicle_id is None:
                vehicle_id_count += 1
                vehicle_id = vehicle_id_count
                timing_dict[vehicle_id] = None
                speed_dict[vehicle_id] = None

            vehicle_positions[vehicle_id] = center_y
            current_vehicles.append(vehicle_id)

            # Check if the vehicle crosses the lines
            if LINE_1_Y - 10 < center_y < LINE_1_Y + 10:
                timing_dict[vehicle_id] = cv2.getTickCount()  # Start timing for this vehicle
                print(f"Vehicle {vehicle_id} crossed Line 1")

            if LINE_2_Y - 10 < center_y < LINE_2_Y + 10:
                if timing_dict.get(vehicle_id) is not None:
                    time_taken = (cv2.getTickCount() - timing_dict[vehicle_id]) / cv2.getTickFrequency()
                    if time_taken > 0:
                        # Calculate speed based on real-world distance and time
                        speed = (REAL_WORLD_DISTANCE / time_taken) * 3.6  # Speed in km/h
                        speed_dict[vehicle_id] = speed  # Store speed
                        print(f"Vehicle {vehicle_id} crossed Line 2. Speed: {speed:.2f} km/h")

                        # If speed seems too high or too low, set a reasonable limit for highway speeds
                        if 10 <= speed <= 300:
                            speed_kmph_rounded = round(speed, 2)
                            # Insert speed into the database
                            try:
                                mycursor.execute("INSERT INTO speed_data (speed) VALUES (%s)", (speed_kmph_rounded,))
                                mydb.commit()
                                print(f"Speed {speed:.2f} km/h for Vehicle {vehicle_id} stored in database")
                            except mysql.connector.Error as err:
                                print(f"Error: {err}")

                        if speed > 70:
                            # Annotate speed on the full image
                            cv2.putText(frame, f"Speed: {speed_kmph_rounded:.2f} km/h",
                                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                            # Capture full frame with speed annotation
                            image_filename = os.path.join(SAVE_IMAGE_PATH, f"vehicle_{vehicle_id}_speed_{speed_kmph_rounded}.png")
                            cv2.imwrite(image_filename, frame)
                            print(f"Captured image of Vehicle {vehicle_id} with speed {speed:.2f} km/h")

            # Display the speed above the vehicle if it exists
            if speed_dict.get(vehicle_id) is not None:
                speed_text = f"{speed_dict[vehicle_id]:.2f} km/h"
                cv2.putText(frame, speed_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Remove old vehicle positions
    vehicle_positions = {vid: pos for vid, pos in vehicle_positions.items() if vid in current_vehicles}

    # Draw the lines
    cv2.line(frame, (0, int(LINE_1_Y * RESIZE_FACTOR)), (frame.shape[1], int(LINE_1_Y * RESIZE_FACTOR)), (255, 0, 0),
             LINE_THICKNESS)
    cv2.line(frame, (0, int(LINE_2_Y * RESIZE_FACTOR)), (frame.shape[1], int(LINE_2_Y * RESIZE_FACTOR)), (255, 0, 0),
             LINE_THICKNESS)

    # Show the frame
    cv2.imshow("Vehicle Speed Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
