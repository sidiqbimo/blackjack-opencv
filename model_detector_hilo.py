import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pygame
import os
import time
import random
import json
import subprocess
import sys

pygame.init()
pygame.font.init()
font = pygame.font.Font(None, 36)

window_size = (600, 800)

model_path = "D:\\Programming\\Python\\cardGameProject\\akk-card_classifier_model.keras"
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

card_directory = "D:/Programming/Python/cardGameProject/illust/cardpile/"
background_drawtoready = "D:\\Programming\\Python\\cardGameProject\\illust\\hilo_draw.png"
background_playstate = "D:\\Programming\\Python\\cardGameProject\\illust\\hilo_play.png"
background_won = "D:\\Programming\\Python\\cardGameProject\\illust\\hilo_won.png"
background_lost = "D:\\Programming\\Python\\cardGameProject\\illust\\hilo_lost.png"

background_image = pygame.image.load(background_drawtoready)
background_image = pygame.transform.scale(background_image, window_size)

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

# Game phases: 'ready', 'guessing', 'result'
phase = "ready"

computer_card = None
player_card = None
overlay_displayed = None 

card_value_mapping = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10,
    "jack": 10, "queen": 10, "king": 10, "ace": 11
}

# Function to pick a card
def pick_card(card_directory):
    card_files = [
        "2_of_clubs", "2_of_diamonds", "2_of_hearts", "2_of_spades",
        "3_of_clubs", "3_of_diamonds", "3_of_hearts", "3_of_spades",
        "4_of_clubs", "4_of_diamonds", "4_of_hearts", "4_of_spades",
        "5_of_clubs", "5_of_diamonds", "5_of_hearts", "5_of_spades",
        "6_of_clubs", "6_of_diamonds", "6_of_hearts", "6_of_spades",
        "7_of_clubs", "7_of_diamonds", "7_of_hearts", "7_of_spades",
        "8_of_clubs", "8_of_diamonds", "8_of_hearts", "8_of_spades",
        "9_of_clubs", "9_of_diamonds", "9_of_hearts", "9_of_spades",
        "10_of_clubs", "10_of_diamonds", "10_of_hearts", "10_of_spades",
        "jack_of_clubs", "jack_of_diamonds", "jack_of_hearts", "jack_of_spades",
        "queen_of_clubs", "queen_of_diamonds", "queen_of_hearts", "queen_of_spades",
        "king_of_clubs", "king_of_diamonds", "king_of_hearts", "king_of_spades",
        "ace_of_clubs", "ace_of_diamonds", "ace_of_hearts", "ace_of_spades"
    ]
    card = random.choice(card_files)
    return card

# Function to reset the game
def reset_game():
    global computer_card, player_card, phase, background_image
    computer_card = pick_card(card_directory)  # Draw computer's card
    player_card = None
    phase = "ready"
    background_image = pygame.image.load(background_drawtoready)
    background_image = pygame.transform.scale(background_image, window_size)

# Preprocess card image for prediction
def preprocess_image(image):
    image = cv2.resize(image, (190, 338))
    image = image.astype(np.float32) / 255.0
    return np.expand_dims(image, axis=0)

def backtomenu():
    pygame.quit()  
    subprocess.run(["python", "D:/Programming/Python/cardGameProject/blackjack_mainmenu.py"])
    sys.exit() 

# Function to detect player's card
def detect_player_card():
    global player_card, phase, background_image, detected_class_id
    ret, frame = cam.read()
    if not ret:
        print("Error: Couldn't capture frame.")
        return

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
                class_id = np.argmax(prediction)
                confidence = prediction[0][class_id]
                
                if confidence > 0.6:
                    card_label = class_labels[class_id].split("(")[0]  # Get the card label
                    player_card = card_value_mapping.get(card_label, 0)  # Get the card value
                    background_image = pygame.image.load(background_playstate)
                    phase = "guessing"
                    detected_class_id = class_id  # Store the detected class_id

                    # Print card information
                    print(f"Card Prediction: {class_labels[class_id]}, Confidence: {confidence:.2f}")
                    print(f"Player Card Value: {player_card}")
                    print(f"Computer Card: {computer_card}")

                    # Visualize detection
                    label = f"{class_labels[class_id]} ({confidence:.2f})"
                    cv2.drawContours(frame, [approx], 0, (0, 255, 0), 2)
                    for point in approx:
                        cv2.circle(frame, tuple(point[0]), 5, (0, 0, 255), -1)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                else:
                    print(f"Low confidence for class {class_id}: {confidence}")

# Function to get card file name from class label
def get_card_file_name(class_label):
    card_files = [
        "2_of_clubs", "2_of_diamonds", "2_of_hearts", "2_of_spades",
        "3_of_clubs", "3_of_diamonds", "3_of_hearts", "3_of_spades",
        "4_of_clubs", "4_of_diamonds", "4_of_hearts", "4_of_spades",
        "5_of_clubs", "5_of_diamonds", "5_of_hearts", "5_of_spades",
        "6_of_clubs", "6_of_diamonds", "6_of_hearts", "6_of_spades",
        "7_of_clubs", "7_of_diamonds", "7_of_hearts", "7_of_spades",
        "8_of_clubs", "8_of_diamonds", "8_of_hearts", "8_of_spades",
        "9_of_clubs", "9_of_diamonds", "9_of_hearts", "9_of_spades",
        "10_of_clubs", "10_of_diamonds", "10_of_hearts", "10_of_spades",
        "jack_of_clubs", "jack_of_diamonds", "jack_of_hearts", "jack_of_spades",
        "queen_of_clubs", "queen_of_diamonds", "queen_of_hearts", "queen_of_spades",
        "king_of_clubs", "king_of_diamonds", "king_of_hearts", "king_of_spades",
        "ace_of_clubs", "ace_of_diamonds", "ace_of_hearts", "ace_of_spades"
    ]
    for card_file in card_files:
        if card_file.startswith(class_label.lower()):
            return card_file
    return None

def draw_text(surface, text, position, font, color=(255, 255, 255)):
    words = [word.split(' ') for word in text.splitlines()]
    space = font.size(' ')[0]
    max_width, max_height = surface.get_size()
    x, y = position
    for line in words:
        for word in line:
            word_surface = font.render(word, True, color)
            word_width, word_height = word_surface.get_size()
            if x + word_width >= max_width:
                x = position[0]  # Reset the x.
                y += word_height  # Start on new row.
            surface.blit(word_surface, (x, y))
            x += word_width + space
        x = position[0]  # Reset the x.
        y += word_height  # Start on new row.

def draw_text_with_background(surface, text, position, font, text_color=(255, 255, 255), bg_color=(0, 0, 0)):
    words = [word.split(' ') for word in text.splitlines()]
    space = font.size(' ')[0]
    max_width, max_height = surface.get_size()
    x, y = position
    for line in words:
        for word in line:
            word_surface = font.render(word, True, text_color)
            word_width, word_height = word_surface.get_size()
            if x + word_width >= max_width:
                x = position[0]  # Reset the x.
                y += word_height  # Start on new row.
            # Create background surface for the text
            bg_surface = pygame.Surface((word_width, word_height))
            bg_surface.fill(bg_color)
            surface.blit(bg_surface, (x, y))
            surface.blit(word_surface, (x, y))
            x += word_width + space
        x = position[0]  # Reset the x.
        y += word_height  # Start on new row.

# Function to handle player's guess
def handle_guess(is_higher, class_id):
    global phase, background_image, result_text
    player_card_value = player_card
    computer_card_label = computer_card.split("_")[0]  # Get the card label
    computer_card_value = card_value_mapping.get(computer_card_label, 0)  # Get the card value
    
    if (is_higher and player_card_value > computer_card_value) or (not is_higher and player_card_value < computer_card_value):
        print("Player won!")
        background_image = pygame.image.load(background_won)
        result_text = f"Computer drew {computer_card.replace('_', ' ')} valued {computer_card_value}, you drew {class_labels[class_id]} valued {player_card_value}. Your prediction was correct, you won!"
    else:
        print("Player lost!")
        background_image = pygame.image.load(background_lost)
        result_text = f"Computer drew {computer_card.replace('_', ' ')} valued {computer_card_value}, you drew {class_labels[class_id]} valued {player_card_value}. Your prediction was incorrect, you lost!"
    
    # Get the file names for the card images
    player_card_file = get_card_file_name(class_labels[class_id].split("(")[0])
    computer_card_file = computer_card

    # Load and resize card images
    player_card_image = pygame.image.load(f"D:/Programming/Python/cardGameProject/illust/cardpile/{player_card_file}.png")
    computer_card_image = pygame.image.load(f"D:/Programming/Python/cardGameProject/illust/cardpile/{computer_card_file}.png")
    player_card_image = pygame.transform.scale(player_card_image, (100, 150))  # Resize to 100x150
    computer_card_image = pygame.transform.scale(computer_card_image, (100, 150))
    screen.blit(player_card_image, (280, 150))
    screen.blit(computer_card_image, (480, 150))
    
    phase = "result"

# Initialize camera
cam = cv2.VideoCapture(3)
if not cam.isOpened():
    print("Error: Couldn't open camera.")
    exit(1)

screen = pygame.display.set_mode(window_size)
clock = pygame.time.Clock()

reset_game()

# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if phase == "guessing":
                if event.key == pygame.K_UP or event.key == pygame.K_DOWN:
                    guess = "UP" if event.key == pygame.K_UP else "DOWN"
                    handle_guess(is_higher=(guess == "UP"), class_id=detected_class_id)
            elif event.key == pygame.K_r:
                phase = "ready"
                computer_card = pick_card("path_to_card_images")
                player_card = None
                background_image = pygame.image.load(background_drawtoready)
            elif event.key == pygame.K_m:
                backtomenu()
            elif event.key == pygame.K_q:
                pygame.quit()
                sys.exit()

    screen.blit(background_image, (0, 0))

    if phase == "ready":
        detect_player_card()
    elif phase == "result":
        # Display card images
        player_card_file = get_card_file_name(class_labels[detected_class_id].split("(")[0])
        computer_card_file = computer_card
        player_card_image = pygame.image.load(f"D:/Programming/Python/cardGameProject/illust/cardpile/{player_card_file}.png")
        computer_card_image = pygame.image.load(f"D:/Programming/Python/cardGameProject/illust/cardpile/{computer_card_file}.png")

        player_card_image = pygame.transform.scale(player_card_image, (100, 150))  # Resize to 100x150
        computer_card_image = pygame.transform.scale(computer_card_image, (100, 150))
        screen.blit(player_card_image, (280, 150))
        screen.blit(computer_card_image, (480, 150))
        
        # Render result text with background
        draw_text_with_background(screen, result_text, (50, 50), font, text_color=(255, 255, 255), bg_color=(0, 0, 0))

    # Display the camera feed
    ret, frame = cam.read()
    if not ret:
        print("Error: Couldn't capture frame.")
        break

    frame_height, frame_width = frame.shape[:2]
    camera_width, camera_height = 560, 320
    frame_resized = cv2.resize(frame, (camera_width, camera_height))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_flipped = cv2.flip(frame_rgb, 1)
    frame_surface = pygame.surfarray.make_surface(np.rot90(frame_flipped))
    screen.blit(frame_surface, (20, 450))

    pygame.display.flip()
    clock.tick(30)

cam.release()
cv2.destroyAllWindows()
pygame.quit()
