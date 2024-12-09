import cv2
import numpy as np
import os
import json

# Global variables for HSV bounds
hsv_lower_bound = np.array([0, 0, 70])
hsv_upper_bound = np.array([180, 40, 140])

hsv_settings_file = "D:/Programming/Python/cardGameProject/hsv_settings.json"

# Function to update the HSV bounds from trackbars

# Function to load HSV bounds from a file
def load_hsv_settings(file_path):
    global hsv_lower_bound, hsv_upper_bound
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            hsv_data = json.load(file)
            hsv_lower_bound = np.array(hsv_data["lower_bound"])
            hsv_upper_bound = np.array(hsv_data["upper_bound"])
            print("Loaded HSV settings:", hsv_lower_bound, hsv_upper_bound)
    else:
        print("HSV settings file not found. Using default values.")

load_hsv_settings(hsv_settings_file)


def update_hsv_bounds(x):
    global hsv_lower_bound, hsv_upper_bound
    hsv_lower_bound = np.array([
        cv2.getTrackbarPos("Lower H", "Controls"),
        cv2.getTrackbarPos("Lower S", "Controls"),
        cv2.getTrackbarPos("Lower V", "Controls")
    ])
    hsv_upper_bound = np.array([
        cv2.getTrackbarPos("Upper H", "Controls"),
        cv2.getTrackbarPos("Upper S", "Controls"),
        cv2.getTrackbarPos("Upper V", "Controls")
    ])

# Load HSV settings from file before creating trackbars
load_hsv_settings(hsv_settings_file)

# Function to draw a circle at the specified (x, y) location
def drawCircle(image, x, y):
    center_coordinates = (x, y)
    radius = 4
    color = (0, 0, 255)  # Red color for the corners
    thickness = 2
    image = cv2.circle(image, center_coordinates, radius, color, thickness)
    return image

# Function to draw circles at the corners of detected contours
def draw_corners(frame, contours):
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Check if the polygon has four points, indicating a potential card shape
        if len(approx) == 4:
            for point in approx:
                x, y = point[0]
                frame = drawCircle(frame, x, y)
    return frame

# Function to draw a circle at the clicked point and update HSV bounds
def pick_color(event, x, y, flags, param):
    global hsv_lower_bound, hsv_upper_bound, frame
    if event == cv2.EVENT_LBUTTONDOWN:
        bgr_color = frame[y, x]
        hsv_lower_bound, hsv_upper_bound = get_hsv_bounds_from_bgr(bgr_color)
        update_trackbars()  # Update the trackbars to reflect the new bounds

def get_hsv_bounds_from_bgr(bgr_color, hue_tolerance=15, saturation_tolerance=50, value_tolerance=50):
    bgr_color = np.uint8([[bgr_color]])
    hsv_color = cv2.cvtColor(bgr_color, cv2.COLOR_BGR2HSV)[0][0]
    
    lower_bound = np.array([
        max(0, hsv_color[0] - hue_tolerance),
        max(0, hsv_color[1] - saturation_tolerance),
        max(0, hsv_color[2] - value_tolerance)
    ])
    upper_bound = np.array([
        min(180, hsv_color[0] + hue_tolerance),
        min(255, hsv_color[1] + saturation_tolerance),
        min(255, hsv_color[2] + value_tolerance)
    ])
    return lower_bound, upper_bound

# Function to update the trackbars when HSV bounds are changed by clicking
def update_trackbars():
    cv2.setTrackbarPos("Lower H", "Controls", hsv_lower_bound[0])
    cv2.setTrackbarPos("Lower S", "Controls", hsv_lower_bound[1])
    cv2.setTrackbarPos("Lower V", "Controls", hsv_lower_bound[2])
    cv2.setTrackbarPos("Upper H", "Controls", hsv_upper_bound[0])
    cv2.setTrackbarPos("Upper S", "Controls", hsv_upper_bound[1])
    cv2.setTrackbarPos("Upper V", "Controls", hsv_upper_bound[2])

# Function to zoom the frame based on a zoom scale
def zoom_frame(frame, scale=1.0):
    if scale == 1.0:
        return frame  # No zoom applied
    height, width = frame.shape[:2]
    new_height, new_width = int(height * scale), int(width * scale)
    resized_frame = cv2.resize(frame, (new_width, new_height))
    
    # Crop or pad the frame to the original size to center the zoom
    if scale > 1.0:  # Zoom in
        start_y = (new_height - height) // 2
        start_x = (new_width - width) // 2
        return resized_frame[start_y:start_y + height, start_x:start_x + width]
    else:  # Zoom out
        padded_frame = np.zeros_like(frame)
        offset_y = (height - new_height) // 2
        offset_x = (width - new_width) // 2
        padded_frame[offset_y:offset_y + new_height, offset_x:offset_x + new_width] = resized_frame
        return padded_frame

# Initialize the camera
cam = cv2.VideoCapture(4)
if not cam.isOpened():
    print("Error: Couldn't open camera.")
    exit(1)

# Create an OpenCV window for control sliders
cv2.namedWindow("Controls")
cv2.createTrackbar("Lower H", "Controls", hsv_lower_bound[0], 180, update_hsv_bounds)
cv2.createTrackbar("Lower S", "Controls", hsv_lower_bound[1], 255, update_hsv_bounds)
cv2.createTrackbar("Lower V", "Controls", hsv_lower_bound[2], 255, update_hsv_bounds)
cv2.createTrackbar("Upper H", "Controls", hsv_upper_bound[0], 180, update_hsv_bounds)
cv2.createTrackbar("Upper S", "Controls", hsv_upper_bound[1], 255, update_hsv_bounds)
cv2.createTrackbar("Upper V", "Controls", hsv_upper_bound[2], 255, update_hsv_bounds)

# Create a trackbar for zoom control
cv2.createTrackbar("Zoom", "Controls", 10, 30, lambda x: None)  # Range from 0.1x to 3.0x

# Set mouse callback for the OpenCV window
cv2.namedWindow("Combined View")
cv2.setMouseCallback("Combined View", pick_color)

while True:
    ret, frame = cam.read()
    if not ret:
        print("Error: Couldn't capture frame.")
        break

    # Get zoom level from the trackbar
    zoom_level = cv2.getTrackbarPos("Zoom", "Controls") / 10.0  # Convert trackbar position to a scale

    # Apply zoom to the frame
    frame = zoom_frame(frame, zoom_level)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get updated HSV bounds from trackbars
    update_hsv_bounds(0)

    # Create mask with the dynamically set bounds
    mask = cv2.inRange(hsv, hsv_lower_bound, hsv_upper_bound)

    # Apply Gaussian Blur to reduce noise
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Refine the mask using erosion and dilation to clean up noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Invert the mask to isolate only non-background objects (e.g., cards)
    mask = cv2.bitwise_not(mask)
    foreground = cv2.bitwise_and(frame, frame, mask=mask)

    # Find contours of the masked areas
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame = draw_corners(frame, contours)

    # Resize each frame to fit within the display window
    display_scale = 0.6  # Adjust this value to fit the combined display window
    frame_resized = cv2.resize(frame, (0, 0), fx=display_scale, fy=display_scale)
    mask_resized = cv2.resize(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), (0, 0), fx=display_scale, fy=display_scale)
    foreground_resized = cv2.resize(foreground, (0, 0), fx=display_scale, fy=display_scale)

    # Combine all frames into one display window
    combined_frame = np.hstack((frame_resized, mask_resized, foreground_resized))

    # Display the combined window
    cv2.imshow("Combined View", combined_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cam.release()
cv2.destroyAllWindows()
