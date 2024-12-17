import cv2
import numpy as np
from tensorflow.keras.models import load_model
import json
import os
import pygame
import sys
import subprocess

# model_path = "D:\\Programming\\Python\\cardGameProject\\kaggle_set\\train\\\card_classifier_model.keras"
# model_path = "D:\\Programming\\Python\\cardGameProject\\akk-card_classifier_model.keras"
model_path = "D:\\Programming\\Python\\cardGameProject\\akk-card_classifier_model.keras"
model = load_model(model_path)


class_labels = [
    "As | Keriting | [1 or 11]", "As | Wajik | [1 or 11]", "As | Hati | [1 or 11]", "As | Sekop | [1 or 11]",
    "8 | Keriting | [8]", "8 | Wajik | [8]", "8 | Hati | [8]", "8 | Sekop | [8]",
    "5 | Keriting | [5]", "5 | Wajik | [5]", "5 | Hati | [5]", "5 | Sekop | [5]",
    "4 | Keriting | [4]", "4 | Wajik | [4]", "4 | Hati | [4]", "4 | Sekop | [4]",
    "Jack | Keriting | [10]", "Jack | Wajik | [10]", "Jack | Hati | [10]", "Jack | Sekop | [10]",
    "Joker | [0]","King | Keriting | [10]", "King | Wajik | [10]", "King | Hati | [10]", "King | Sekop | [10]",
    "9 | Keriting | [9]", "9 | Wajik | [9]", "9 | Hati | [9]", "9 | Sekop | [9]",
    "Queen | Keriting | [10]", "Queen | Wajik | [10]", "Queen | Hati | [10]", "Queen | Sekop | [10]",
    "7 | Keriting | [7]", "7 | Wajik | [7]", "7 | Hati | [7]", "7 | Sekop | [7]",
    "6 | Keriting | [6]", "6 | Wajik | [6]", "6 | Hati | [6]", "6 | Sekop | [6]",
    "10 | Keriting | [10]", "10 | Wajik | [10]", "10 | Hati | [10]", "10 | Sekop | [10]",
    "3 | Keriting | [3]", "3 | Wajik | [3]", "3 | Hati | [3]", "3 | Sekop | [3]",
    "2 | Keriting | [2]", "2 | Wajik | [2]", "2 | Hati | [2]", "2 | Sekop | [2]"
]



def load_hsv_settings(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            settings = json.load(file)
        return settings
    else:
        return {"lower_bound": [0, 0, 70], "upper_bound": [180, 40, 140]}

# Load HSV settings
json_file_path = "D:/Programming/Python/cardGameProject/hsv_settings.json"
hsv_settings = load_hsv_settings(json_file_path)
hsv_lower_bound, hsv_upper_bound = np.array(hsv_settings["lower_bound"]), np.array(hsv_settings["upper_bound"])


def save_hsv_settings():
    hsv_data = {
        "lower_bound": hsv_lower_bound.tolist(),
        "upper_bound": hsv_upper_bound.tolist()
    }
    with open(json_file_path, 'w') as file:
        json.dump(hsv_data, file)


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

# Function to preprocess an image for model prediction
def preprocess_image(image):
    image = cv2.resize(image, (190, 338))  # Resize to the same dimensions used during training
    image = image.astype(np.float32) / 255.0  # Normalize the image to range [0, 1]
    return np.expand_dims(image, axis=0)  # Add batch dimension

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

def backtomenu():
    sys.exit()
    subprocess.run(["python", "D:/Programming/Python/cardGameProject/blackjack_mainmenu.py"])

cam = cv2.VideoCapture(3)
if not cam.isOpened():
    print("Error: Couldn't open camera.")
    exit(1)


cv2.namedWindow("Controls")
cv2.createTrackbar("Lower H", "Controls", hsv_lower_bound[0], 180, update_hsv_bounds)
cv2.createTrackbar("Lower S", "Controls", hsv_lower_bound[1], 255, update_hsv_bounds)
cv2.createTrackbar("Lower V", "Controls", hsv_lower_bound[2], 255, update_hsv_bounds)
cv2.createTrackbar("Upper H", "Controls", hsv_upper_bound[0], 180, update_hsv_bounds)
cv2.createTrackbar("Upper S", "Controls", hsv_upper_bound[1], 255, update_hsv_bounds)
cv2.createTrackbar("Upper V", "Controls", hsv_upper_bound[2], 255, update_hsv_bounds)
cv2.createTrackbar("Zoom", "Controls", 10, 30, lambda x: None)

# Function to draw a circle at the clicked point and update HSV bounds
def pick_color(event, x, y, flags, param):
    global hsv_lower_bound, hsv_upper_bound, frame
    if event == cv2.EVENT_LBUTTONDOWN:
        bgr_color = frame[y, x]
        hsv_lower_bound, hsv_upper_bound = get_hsv_bounds_from_bgr(bgr_color)
        update_trackbars()  # Update the trackbars to reflect the new bounds
        
# update trackbar when clicked
def update_trackbars():
    cv2.setTrackbarPos("Lower H", "Controls", hsv_lower_bound[0])
    cv2.setTrackbarPos("Lower S", "Controls", hsv_lower_bound[1])
    cv2.setTrackbarPos("Lower V", "Controls", hsv_lower_bound[2])
    cv2.setTrackbarPos("Upper H", "Controls", hsv_upper_bound[0])
    cv2.setTrackbarPos("Upper S", "Controls", hsv_upper_bound[1])
    cv2.setTrackbarPos("Upper V", "Controls", hsv_upper_bound[2])

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

cv2.namedWindow("Combined View")
cv2.setMouseCallback("Combined View", pick_color)

while True:
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('m'):  # M to go back to the main menu
       backtomenu()
    elif key == ord('q'): 
        break

    ret, frame = cam.read()
    if not ret:
        print("Error: Couldn't capture frame.")
        break

    zoom_level = cv2.getTrackbarPos("Zoom", "Controls") / 10.0
    frame = zoom_frame(frame, zoom_level)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_lower_bound, hsv_upper_bound)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    mask = cv2.bitwise_not(mask)
    foreground = cv2.bitwise_and(frame, frame, mask=mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            card_image = foreground[y:y + h, x:x + w]  
            if card_image.size > 0:
                preprocessed_card = preprocess_image(card_image)
                prediction = model.predict(preprocessed_card)
                print(f"Prediction: {prediction}")  
                class_id = np.argmax(prediction)
                confidence = prediction[0][class_id]
                print(f"Class ID: {class_id}, Confidence: {confidence}")  
                if class_id < len(class_labels) and confidence > 0.5:  
                    label = f"{class_labels[class_id]} ({confidence:.2f})"
                    cv2.drawContours(frame, [approx], 0, (0, 255, 0), 2)
                    for point in approx:
                        cv2.circle(frame, tuple(point[0]), 5, (0, 0, 255), -1)  
                    text_x, text_y = x + w // 2, y + h // 2
                    cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    print(f"Unknown class ID: {class_id} with confidence {confidence}")

    mask_resized = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    display_scale = 0.6
    frame_resized = cv2.resize(frame, (0, 0), fx=display_scale, fy=display_scale)
    mask_resized = cv2.resize(mask_resized, (0, 0), fx=display_scale, fy=display_scale)
    foreground_resized = cv2.resize(foreground, (0, 0), fx=display_scale, fy=display_scale)
    combined_frame = np.hstack((frame_resized, mask_resized, foreground_resized))

    cv2.imshow("Combined View", combined_frame)



cam.release()
cv2.destroyAllWindows()