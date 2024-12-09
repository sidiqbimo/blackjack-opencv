import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import os

# Load the trained model
model_path = "D:\\Programming\\Python\\cardGameProject\\card_classifier_model.keras"
model = load_model(model_path)

# Load the label dictionary
label_dict = {0: 'K_Diamond', 1: 'K_Sekop'}  # Replace with your actual label mapping

# Load HSV settings from JSON file
def load_hsv_settings(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            settings = json.load(file)
        return settings
    else:
        return {
            "lower_bound": [0, 0, 70],
            "upper_bound": [180, 40, 140]
        }

# Path to the JSON file
json_file_path = "D:/Programming/Python/cardGameProject/hsv_settings.json"
hsv_settings = load_hsv_settings(json_file_path)

# Global variables for HSV bounds
hsv_lower_bound = np.array(hsv_settings["lower_bound"])
hsv_upper_bound = np.array(hsv_settings["upper_bound"])

# Function to update the HSV bounds from trackbars
def update_hsv_bounds(val):
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

# Function to draw a label on the frame
def draw_label(frame, text, x, y):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, (x, y), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

# Function to preprocess an image for model prediction
def preprocess_image(image):
    image = cv2.resize(image, (128, 128))
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to zoom the frame based on a zoom scale
def zoom_frame(frame, scale=1.0):
    if scale == 1.0:
        return frame
    height, width = frame.shape[:2]
    new_height, new_width = int(height * scale), int(width * scale)
    resized_frame = cv2.resize(frame, (new_width, new_height))
    
    if scale > 1.0:
        start_y = (new_height - height) // 2
        start_x = (new_width - width) // 2
        return resized_frame[start_y:start_y + height, start_x:start_x + width]
    else:
        padded_frame = np.zeros_like(frame)
        offset_y = (height - new_height) // 2
        offset_x = (width - new_width) // 2
        padded_frame[offset_y:offset_y + new_height, offset_x:offset_x + new_width] = resized_frame
        return padded_frame

# Function to ensure the card is vertical
def ensure_vertical_orientation(image, contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    width, height = int(rect[1][0]), int(rect[1][1])

    if width < height:
        M = cv2.getRotationMatrix2D((width // 2, height // 2), 90, 1)
        image = cv2.warpAffine(image, M, (width, height))

    return image, box

# Initialize the camera
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Error: Couldn't open camera.")
    exit(1)

# Create an OpenCV window for control sliders
cv2.namedWindow("Controls")
cv2.createTrackbar("Lower H", "Controls", hsv_settings["lower_bound"][0], 180, update_hsv_bounds)
cv2.createTrackbar("Lower S", "Controls", hsv_settings["lower_bound"][1], 255, update_hsv_bounds)
cv2.createTrackbar("Lower V", "Controls", hsv_settings["lower_bound"][2], 255, update_hsv_bounds)
cv2.createTrackbar("Upper H", "Controls", hsv_settings["upper_bound"][0], 180, update_hsv_bounds)
cv2.createTrackbar("Upper S", "Controls", hsv_settings["upper_bound"][1], 255, update_hsv_bounds)
cv2.createTrackbar("Upper V", "Controls", hsv_settings["upper_bound"][2], 255, update_hsv_bounds)

# Create a trackbar for zoom control
cv2.createTrackbar("Zoom", "Controls", 10, 30, lambda x: None)

while True:
    ret, frame = cam.read()
    if not ret:
        print("Error: Couldn't capture frame.")
        break

    # Get zoom level from the trackbar
    zoom_level = cv2.getTrackbarPos("Zoom", "Controls") / 10.0
    frame = zoom_frame(frame, zoom_level)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create mask with the dynamically set bounds
    mask = cv2.inRange(hsv, hsv_lower_bound, hsv_upper_bound)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    mask = cv2.bitwise_not(mask)
    foreground = cv2.bitwise_and(frame, frame, mask=mask)

    # Detect contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            card_image = frame[y:y + h, x:x + w]

            if card_image.size > 0:
                card_image, _ = ensure_vertical_orientation(card_image, approx)
                preprocessed_card = preprocess_image(card_image)
                prediction = model.predict(preprocessed_card)
                class_id = np.argmax(prediction)
                confidence = prediction[0][class_id]

                if confidence > 0.7:
                    label = f"{label_dict[class_id]} ({confidence:.2f})"
                    cv2.drawContours(frame, [approx], 0, (0, 255, 0), 2)
                    draw_label(frame, label, x, y - 10)

    # Display the processed frame and original frame
    cv2.imshow("Card Classification", frame)
    cv2.imshow("Masked Foreground", foreground)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cam.release()
cv2.destroyAllWindows()
