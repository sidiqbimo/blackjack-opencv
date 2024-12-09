import cv2
import numpy as np
import os
from datetime import datetime
import time
import json

# Global variables for HSV bounds and saving state
hsv_lower_bound = np.array([0, 0, 70])
hsv_upper_bound = np.array([180, 40, 140])
saving_enabled = False
input_name = ""

hsv_settings_file = "D:/Programming/Python/cardGameProject/hsv_settings.json"

# Function to save HSV bounds to a file
def save_hsv_settings():
    hsv_data = {
        "lower_bound": hsv_lower_bound.tolist(),
        "upper_bound": hsv_upper_bound.tolist()
    }
    with open(hsv_settings_file, 'w') as file:
        json.dump(hsv_data, file)

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

# Function to update the HSV bounds from trackbars
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
    save_hsv_settings()  # Save settings after updating

# Function to draw a circle at the specified (x, y) location
def drawCircle(image, x, y):
    center_coordinates = (x, y)
    radius = 4
    color = (0, 0, 255)  # Red color for the corners
    thickness = 2
    image = cv2.circle(image, center_coordinates, radius, color, thickness)
    return image

# Function to zoom the frame based on a zoom scale
def zoom_frame(frame, scale=1.0):
    if scale == 1.0:
        return frame  # No zoom applied
    height, width = frame.shape[:2]
    new_height, new_width = int(height * scale), int(width * scale)
    resized_frame = cv2.resize(frame, (new_width, new_height))
    
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

# Function to crop and adjust the orientation of the card
def crop_and_orient_card(frame, contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    width, height = int(rect[1][0]), int(rect[1][1])
    
    src_pts = box.astype("float32")
    dst_pts = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(frame, M, (width, height))

    if height > width:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

    aspect_ratio = 3.5 / 2.5  # Standard playing card aspect ratio (height/width)
    h, w = warped.shape[:2]
    new_width = int(h / aspect_ratio)
    resized_warped = cv2.resize(warped, (new_width, h))

    return resized_warped, box

# Mouse callback function to trigger background color detection
def pick_color(event, x, y, flags, param):
    global hsv_lower_bound, hsv_upper_bound, frame
    if event == cv2.EVENT_LBUTTONDOWN:
        bgr_color = frame[y, x]
        hsv_lower_bound, hsv_upper_bound = get_hsv_bounds_from_bgr(bgr_color)
        update_trackbars()

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

# Initialize the camera
cam = cv2.VideoCapture(0)
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
cv2.createTrackbar("Zoom", "Controls", 10, 30, lambda x: None)

# Set mouse callback for the camera feed
cv2.namedWindow("Combined View")
cv2.setMouseCallback("Combined View", pick_color)

capture_count = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("Error: Couldn't capture frame.")
        break

    zoom_level = cv2.getTrackbarPos("Zoom", "Controls") / 10.0
    frame = zoom_frame(frame, zoom_level)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    update_hsv_bounds(0)
    mask = cv2.inRange(hsv, hsv_lower_bound, hsv_upper_bound)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.bitwise_not(mask)
    foreground = cv2.bitwise_and(frame, frame, mask=mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    card_contour = None
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            card_contour = contour
            break

    if card_contour is not None:
        _, corners = crop_and_orient_card(frame, card_contour)
        for corner in corners:
            drawCircle(frame, corner[0], corner[1])

    cv2.putText(frame, "Press 's' to Save Card", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    if saving_enabled and card_contour is not None and capture_count < 20:
        cropped_card, _ = crop_and_orient_card(frame, card_contour)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        save_path = f"D:\\Programming\\Python\\cardGameProject\\dataset\\{input_name}"
        os.makedirs(save_path, exist_ok=True)
        filename = os.path.join(save_path, f"{timestamp}.jpg")
        time.sleep(0.5)  # Delay between captures
        cv2.imwrite(filename, cropped_card)
        print(f"Saved: {filename}")
        capture_count += 1

        if capture_count >= 20:
            print("Captured 20 images.")
            saving_enabled = False
            capture_count = 0

    mask_resized = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    display_scale = 0.6
    frame_resized = cv2.resize(frame, (0, 0), fx=display_scale, fy=display_scale)
    mask_resized = cv2.resize(mask_resized, (0, 0), fx=display_scale, fy=display_scale)
    foreground_resized = cv2.resize(foreground, (0, 0), fx=display_scale, fy=display_scale)
    combined_frame = np.hstack((frame_resized, mask_resized, foreground_resized))

    cv2.imshow("Combined View", combined_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        input_name = input("Enter card name (e.g., 'K_Keriting'): ").strip()
        if input_name:
            saving_enabled = True

cam.release()
cv2.destroyAllWindows()
