import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pygame 
import os
import json
import time
import subprocess
import sys

pygame.init()
pygame.font.init()
window_size = (600,800)

model_path = "D:\\Programming\\Python\\cardGameProject\\kaggle_set\\train\\card_classifier_model.keras"
model = load_model(model_path)


class_labels = [
    "K(A)", "K(W)", "K(H)", "K(S)",  
    "8(K)", "8(W)", "8(H)", "8(S)",  
    "5(K)", "5(W)", "5(H)", "5(S)", 
    "4(K)", "4(W)", "4(H)", "4(S)",  
    "J(K)", "J(W)", "J(H)", "J(S)",  
    "Joker", 
    "K(K)", "K(W)", "K(H)", "K(S)",  
    "9(K)", "9(W)", "9(H)", "9(S)", 
    "Q(K)", "Q(W)", "Q(H)", "Q(S)",  
    "7(K)", "7(W)", "7(H)", "7(S)", 
    "6(K)", "6(W)", "6(H)", "6(S)",  
    "10(K)", "10(W)", "10(H)", "10(S)",  
    "3(K)", "3(W)", "3(H)", "3(S)",  
    "2(K)", "2(W)", "2(H)", "2(S)"  
]
blackjack_values = [
    11, 11, 11, 11, 8, 8, 8, 8, 5, 5, 5, 5,
    4, 4, 4, 4, 10, 10, 10, 10, 0, 10, 10, 10,
    10, 9, 9, 9, 9, 10, 10, 10, 10, 7, 7, 7,
    7, 6, 6, 6, 6, 10, 10, 10, 10, 3, 3, 3,
    3, 2, 2, 2, 2
]

# Load HSV settings from JSON
def load_hsv_settings(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            settings = json.load(file)
        return settings
    else:
        return {"lower_bound": [0, 0, 70], "upper_bound": [180, 40, 140]}

json_file_path = "D:/Programming/Python/cardGameProject/hsv_settings.json"
hsv_settings = load_hsv_settings(json_file_path)
hsv_lower_bound, hsv_upper_bound = np.array(hsv_settings["lower_bound"]), np.array(hsv_settings["upper_bound"])

# dict
detected_cards = {}

# Preprocess card image for prediction
def preprocess_image(image):
    image = cv2.resize(image, (190, 338))
    image = image.astype(np.float32) / 255.0
    return np.expand_dims(image, axis=0)

boost_image = cv2.imread("D:\\Programming\\Python\\cardGameProject\\illust\\boost.png", cv2.IMREAD_UNCHANGED)
ensure_image = cv2.imread("D:\\Programming\\Python\\cardGameProject\\illust\\ensure.png", cv2.IMREAD_UNCHANGED)
blackjack_image = cv2.imread("D:\\Programming\\Python\\cardGameProject\\illust\\blackjack.png", cv2.IMREAD_UNCHANGED)
busted_image = cv2.imread("D:\\Programming\\Python\\cardGameProject\\illust\\busted.png", cv2.IMREAD_UNCHANGED)

bell_sound_path = "D:\\Programming\\Python\\cardGameProject\\illust\\bell.wav"
bg_music_path = "D:\\Programming\\Python\\cardGameProject\\illust\\bg.mp3"

boost_image_resized = cv2.resize(boost_image, (400, 400))
ensure_image_resized = cv2.resize(ensure_image, (400, 400))
blackjack_image_resized = cv2.resize(blackjack_image, (400, 400))
busted_image_resized = cv2.resize(busted_image, (400, 400))

bell_sound = pygame.mixer.Sound(bell_sound_path)
bgmusic_sound = pygame.mixer.Sound(bg_music_path)

bgmusic_sound.set_volume(0.5)
bgmusic_sound.play(loops=-1)


def overlay_transparent(background, overlay, x, y):
    """Overlays a transparent image onto a background."""
    overlay_h, overlay_w = overlay.shape[:2]
    background_h, background_w = background.shape[:2]

    if overlay_h + y > background_h:
        overlay_h = background_h - y
    if overlay_w + x > background_w:
        overlay_w = background_w - x

    if overlay_h <= 0 or overlay_w <= 0:
        return background  
    
    overlay_resized = cv2.resize(overlay, (overlay_w, overlay_h), interpolation=cv2.INTER_AREA)
    alpha_channel = overlay_resized[:, :, 3] / 255.0

    for c in range(3):  
        background[y:y+overlay_h, x:x+overlay_w, c] = (
            alpha_channel * overlay_resized[:, :, c] +
            (1 - alpha_channel) * background[y:y+overlay_h, x:x+overlay_w, c]
        )
    return background

def overlay_image_center(frame, overlay):
    overlay_height, overlay_width = overlay.shape[:2]
    x_offset = (frame.shape[1] - overlay_width) // 2
    y_offset = (frame.shape[0] - overlay_height) // 2

    if overlay.shape[2] == 4:  
        alpha_overlay = overlay[:, :, 3] / 255.0
        alpha_frame = 1.0 - alpha_overlay

        for c in range(0, 3):
            frame[y_offset:y_offset+overlay_height, x_offset:x_offset+overlay_width, c] = (
                alpha_overlay * overlay[:, :, c] +
                alpha_frame * frame[y_offset:y_offset+overlay_height, x_offset:x_offset+overlay_width, c]
            )
    else:
        frame[y_offset:y_offset+overlay_height, x_offset:x_offset+overlay_width] = overlay

    return frame

def reset_game():
    global blackjack_reached, busted_reached, total_blackjack_value, start_time
    blackjack_reached = False
    busted_reached = False
    total_blackjack_value = 0
    start_time = time.time()
    detected_cards.clear()

def count_card_value(class_id):
    # Assuming class_id corresponds to card values (e.g., 0-9 for 2-10, 10 for J, 11 for Q, 12 for K, 13 for A)
    if class_id >= 0 and class_id <= 8:
        return class_id + 2
    elif class_id >= 9 and class_id <= 11:
        return 10
    elif class_id == 12:
        return 11
    return 0

def backtomenu():
    pygame.quit()  
    subprocess.run(["python", "D:/Programming/Python/cardGameProject/blackjack_mainmenu.py"])
    sys.exit() 

cam = cv2.VideoCapture(4)
if not cam.isOpened():
    print("Error: Couldn't open camera.")
    exit(1)

start_time = time.time()
blackjack_reached = False
busted_reached = False
total_blackjack_value = 0
overlay_displayed = None

fps_start_time = time.time()
fps_frame_count = 0
fps = 0


while True:
    ret, frame = cam.read()
    if not ret:
        print("Error: Couldn't capture frame.")
        break

    current_time = time.time()
    fps_frame_count += 1

    if current_time - fps_start_time >= 1.0:
        fps = fps_frame_count / (current_time - fps_start_time)
        fps_start_time = current_time
        fps_frame_count = 0

    frame_height, frame_width = frame.shape[:2]

    # Calculate text size and position
    text = f"FPS: {fps:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = frame_width - text_size[0] - 10  # 10 pixels from the right edge
    text_y = text_size[1] + 10  # 10 pixels from the top edge

    # Display text on every frame
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 255, 255), thickness)

    # Display the boost image and play the sound for the first 3 seconds
    if current_time - start_time < 3:
        overlay_displayed = boost_image_resized
        if current_time - start_time < 0.1:
            bell_sound.play()
    elif blackjack_reached:
        overlay_displayed = blackjack_image_resized
    elif busted_reached:
        overlay_displayed = busted_image_resized
        
    # elif total_blackjack_value == 0:
    #     overlay_displayed = ensure_image_resized
    else:
        overlay_displayed = None

    # Overlay the image if there is one to display
    if overlay_displayed is not None:
        frame = overlay_image_center(frame, overlay_displayed)
    
    # Create a mask to exclude the overlay region
    mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
    if overlay_displayed is not None:
        overlay_height, overlay_width = overlay_displayed.shape[:2]
        x_offset = (frame.shape[1] - overlay_width) // 2
        y_offset = (frame.shape[0] - overlay_height) // 2
        mask[y_offset:y_offset+overlay_height, x_offset:x_offset+overlay_width] = 0

    # Apply the mask to the frame
    frame_masked = cv2.bitwise_and(frame, frame, mask=mask)

    # Preprocess frame for card detection
    hsv = cv2.cvtColor(frame_masked, cv2.COLOR_BGR2HSV)
    hsv_mask = cv2.inRange(hsv, hsv_lower_bound, hsv_upper_bound)
    hsv_mask = cv2.GaussianBlur(hsv_mask, (5, 5), 0)
    hsv_mask = cv2.bitwise_not(hsv_mask)
    foreground = cv2.bitwise_and(frame_masked, frame_masked, mask=hsv_mask)
    
    contours, _ = cv2.findContours(hsv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
                
                
                if confidence > 0.5:
                    if class_id not in detected_cards:
                        if overlay_displayed is None:
                                detected_cards[class_id] = True
                                total_blackjack_value += blackjack_values[class_id]

                    # Visualize detection
                    label = f"{class_labels[class_id]} ({confidence:.2f})"
                    cv2.drawContours(frame, [approx], 0, (0, 255, 0), 2)
                    for point in approx:
                        cv2.circle(frame, tuple(point[0]), 5, (0, 0, 255), -1)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                else:
                    print(f"Low confidence for class {class_id}: {confidence}")

                    

    # Check for blackjack or busted
    if total_blackjack_value == 21:
        blackjack_reached = True
    elif total_blackjack_value > 21:
        busted_reached = True

    cv2.putText(frame, f"Value: {total_blackjack_value}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    cv2.imshow("Blackjack Cam", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):  # Space to reset
        if busted_reached or blackjack_reached:
            reset_game()
    elif key == ord('m'):  # M to go back to the main menu
       backtomenu()
    elif key == ord('q'): 
        break


cam.release()
cv2.destroyAllWindows()
pygame.quit()