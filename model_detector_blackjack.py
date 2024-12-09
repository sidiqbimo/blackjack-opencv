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
import random

pygame.init()
pygame.font.init()
font = pygame.font.Font(None, 36)

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

card_values = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10,
    "jack": 10, "queen": 10, "king": 10, "ace": 11
}

dealer_cards = []
dealer_card_values = []

# Function to randomly pick a card and calculate its value
def pick_card(card_directory):
    # Load all card image paths
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
    
    # Exclude jokers
    card_files = [card for card in card_files if "joker" not in card.lower()]
    
    card = random.choice(card_files)
    card_path = f"{card_directory}/{card}.png"

    # Calculate card value
    value = 10 if card.startswith(("jack", "queen", "king")) else 11 if card.startswith("ace") else int(card.split("_")[0])
    return card_path, value

def calculate_best_value(card_values):
    total = sum(card_values)
    aces = card_values.count(11)
    while total > 21 and aces > 0:
        total -= 10
        aces -= 1
    return total

card_directory = "D:/Programming/Python/cardGameProject/illust/cardpile/"
card_folder = card_directory
dealer_cards.append(pick_card(card_directory))
dealer_cards.append(pick_card(card_directory))  # Second card (face down)
dealer_card_values = [dealer_cards[0][1], dealer_cards[1][1]]




# Add "Stand" button functionality
def display_stand_button(screen, font):
    pygame.draw.rect(screen, (200, 200, 200), (20, 450, 100, 50))  # Grey rectangle
    if isinstance(font, pygame.font.Font):  # Ensure font is a Font object
        text = font.render("Stand", True, (0, 0, 0))
    else:
        font = pygame.font.Font(None, 36)
        text = font.render("Stand", True, (0, 0, 0))

    screen.blit(text, (30, 460))

'''Start Features'''
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

# Card pile directory
card_pile_dir = "D:\\Programming\\Python\\cardGameProject\\illust\\cardpile\\"
card_files = [file for file in os.listdir(card_pile_dir) if file.endswith(".png")]
face_down_card = "D:\\Programming\\Python\\cardGameProject\\illust\\blank.png"

# Function to load and display dealer cards
def display_dealer_cards(screen, card_pile_dir, card_files, face_down_card):
    # Randomly choose one face-up card
    face_up_card = os.path.join(card_pile_dir, random.choice(card_files))

    # Load card images
    face_up_image = pygame.image.load(face_up_card)
    face_down_image = pygame.image.load(face_down_card)

    # Resize images
    face_up_image = pygame.transform.scale(face_up_image, (100, 150))
    face_down_image = pygame.transform.scale(face_down_image, (100, 150))

    # Blit cards onto the screen
    defaultX, defaultY = 2000,5000

    screen.blit(face_down_image, (defaultX, defaultY))  # Adjust positions as needed
    screen.blit(face_up_image, (defaultX+120, defaultY))   # Adjust positions as needed
    # selisih X 120

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
win_image = cv2.imread("D:/Programming/Python/cardGameProject/illust/win.png", cv2.IMREAD_UNCHANGED)
lost_image = cv2.imread("D:/Programming/Python/cardGameProject/illust/lost.png", cv2.IMREAD_UNCHANGED)
push_image = cv2.imread("D:/Programming/Python/cardGameProject/illust/push.png", cv2.IMREAD_UNCHANGED)

background_image_bid_path = "D:\\Programming\\Python\\cardGameProject\\illust\\bg_bid.png"
background_image_gamestart_path = "D:\\Programming\\Python\\cardGameProject\\illust\\bg_steady.png"
background_image_win_path = "D:\\Programming\\Python\\cardGameProject\\illust\\bg_win.png"
background_image_lost_path = "D:\\Programming\\Python\\cardGameProject\\illust\\bg_lost.png"
background_image_bankrupt_path = "D:\\Programming\\Python\\cardGameProject\\illust\\bg_bankrupt.png"

background_image = pygame.image.load(background_image_bid_path)
background_image = pygame.transform.scale(background_image, (window_size)) 


def update_background():
    global background_image
    if player_bank <= 0:
        background_image = pygame.image.load(background_image_bankrupt_path)
    elif win_reached:
        background_image = pygame.image.load(background_image_win_path)
    elif lost_reached:
        background_image = pygame.image.load(background_image_lost_path)
    elif busted_reached:
        background_image = pygame.image.load(background_image_lost_path)
    elif blackjack_reached:
        background_image = pygame.image.load(background_image_win_path)
    elif game_started:
        background_image = pygame.image.load(background_image_gamestart_path)
    else:
        background_image = pygame.image.load(background_image_bid_path)

    # Ensure scaling
    background_image = pygame.transform.scale(background_image, window_size)

bell_sound_path = "D:\\Programming\\Python\\cardGameProject\\illust\\bell.wav"
bg_music_path = "D:\\Programming\\Python\\cardGameProject\\illust\\bg.mp3"

boost_image_resized = cv2.resize(boost_image, (400, 400))
ensure_image_resized = cv2.resize(ensure_image, (400, 400))
blackjack_image_resized = cv2.resize(blackjack_image, (400, 400))
busted_image_resized = cv2.resize(busted_image, (400, 400))
win_image_resized = cv2.resize(win_image, (400, 400))
lost_image_resized = cv2.resize(lost_image, (400, 400))
push_image_resized = cv2.resize(push_image, (400, 400))


bell_sound = pygame.mixer.Sound(bell_sound_path)
bgmusic_sound = pygame.mixer.Sound(bg_music_path)

bgmusic_sound.set_volume(0.5)
bgmusic_sound.play(loops=-1)


def display_result_image(frame, result_image):
    return overlay_image_center(frame, result_image)

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
    frame_height, frame_width = frame.shape[:2]
    overlay_height, overlay_width = overlay.shape[:2]

    # Resize overlay if it's larger than the frame
    if overlay_height > frame_height or overlay_width > frame_width:
        scaling_factor = min(frame_height / overlay_height, frame_width / overlay_width)
        overlay = cv2.resize(overlay, (int(overlay_width * scaling_factor), int(overlay_height * scaling_factor)))

    overlay_height, overlay_width = overlay.shape[:2]
    x_offset = (frame_width - overlay_width) // 2
    y_offset = (frame_height - overlay_height) // 2

    if overlay.shape[2] == 4:  # If overlay has an alpha channel
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
    global blackjack_reached, busted_reached, total_blackjack_value, dealer_cards, dealer_card_values, start_time, player_bid, win_reached, lost_reached, show_second_card, detection_started, game_started, push_reached

    blackjack_reached = False
    busted_reached = False
    total_blackjack_value = 0
    dealer_cards.clear()
    dealer_card_values.clear()
    initialize_dealer_cards()  
    start_time = time.time()
    detected_cards.clear()
    show_second_card = False

    win_reached = False
    lost_reached = False
    push_reached = False
    detection_started = False
    game_started = False

    update_background()

    if player_bid >= player_bank:
        player_bid = player_bank

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

cam = cv2.VideoCapture(3)
if not cam.isOpened():
    print("Error: Couldn't open camera.")
    exit(1)

start_time = time.time()
blackjack_reached = False
busted_reached = False
win_reached = False
push_reached = False
lost_reached = False

total_blackjack_value = 0
overlay_displayed = None
screen = pygame.display.set_mode((600, 800))
font = pygame.font.SysFont('Arial', 36)

fps_start_time = time.time()
fps_frame_count = 0
fps = 0

clock = pygame.time.Clock()


# GAME START

# Bank and Bid initialization
player_bank = 2000
player_bid = 100

# Function to handle bidding display and logic
def display_bidding_area(screen, font, player_bank, player_bid):

    bank_text = None
    bid_text = None
    
    # Render Bank and Bid text
    if not game_started and not win_reached and not lost_reached and not busted_reached and not blackjack_reached and not push_reached:
        if isinstance(font, pygame.font.Font):  # Ensure font is a Font object
            bank_text = font.render(f"Your Bank: ${player_bank}", True, (255, 255, 255))
            bid_text = font.render(f"Place Bid: ", True, (255, 255, 255))
            bid_amount_text = font.render(f"${player_bid}", True, (255, 255, 0))  # Yellow color for bid amount
        else:
            font = pygame.font.Font(None, 36)
            bank_text = font.render(f"Your Bank: ${player_bank}", True, (255, 255, 255))
            bid_text = font.render(f"Place Bid: ", True, (255, 255, 255))
            bid_amount_text = font.render(f"${player_bid}", True, (255, 255, 0))  # Yellow color for bid amount

        screen.blit(bank_text, (60, 170))  # Bank position
        screen.blit(bid_text, (60, 210))  # Bid position
        screen.blit(bid_amount_text, (60 + bid_text.get_width(), 210))  # Position bid amount next to "Place Bid: $"
    
    # Check if texts are not None before rendering
    if bank_text and bid_text:
        screen.blit(bank_text, (60, 170))  # Bank position
        screen.blit(bid_text, (60, 210))  # Bid position

def initialize_dealer_cards():
    """Randomly select two cards for the dealer."""
    global dealer_cards, dealer_card_values
    dealer_cards = random.sample(card_files[:-1], 2)  # Exclude "blank.png"
    dealer_card_values = [calculate_card_value(dealer_cards[0]), calculate_card_value(dealer_cards[1])]

    # Initially, only count the value of the first card
    global dealer_visible_total
    dealer_visible_total = dealer_card_values[0]  # Visible card value only

def calculate_card_value(card_name):
    """Calculate the blackjack value of a given card based on its filename."""
    for key, value in card_values.items():
        if key in card_name:
            return value
    return 0  # Default for unrecognized cards

def dealer_draw_card():
    """Dealer draws additional cards until their total is 17 or more."""
    global dealer_cards, dealer_card_values, dealer_total
    dealer_total = calculate_best_value(dealer_card_values)  # Recalculate total before drawing
    while dealer_total < 17:
        new_card = pick_card(card_directory)
        dealer_cards.append(new_card[0])
        dealer_card_values.append(new_card[1])
        dealer_total = calculate_best_value(dealer_card_values)

def display_dealer_area(screen, font):
    """Display the dealer's cards on the top area."""
    y_offset = 125

    for idx, card in enumerate(dealer_cards):
        if idx == 1 and not show_second_card:  # If it's the second card and it's hidden
            card_image = pygame.image.load(face_down_card)
        else:
            card_image = pygame.image.load(os.path.join(card_folder, card))

        card_image_resized = pygame.transform.scale(card_image, (100, 150))  # Resize card image
        x_pos = 325 + (idx * 20)  # Space between cards
        screen.blit(card_image_resized, (x_pos, y_offset))

        


    font = pygame.font.Font(None, 36)
    if show_second_card:
        total = dealer_total
    else:
        total = dealer_card_values[0]  # Show only the first card value
    total_text = font.render(f"Dealer Total: {total}", True, (255, 255, 255))
    bank_total_text = font.render(f"Bank: ${player_bank}", True, (255, 255, 255))

    screen.blit(total_text, (290, 10))  # Adjust position as needed
    screen.blit(bank_total_text, (120, 10))  # Adjust position as needed


initialize_dealer_cards()
game_started = False
show_second_card = False
player_standing = False
dealer_turn = False

detection_started = False

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                if win_reached or lost_reached or busted_reached or blackjack_reached or push_reached:
                    reset_game()  # Prepare for next round
            elif event.key == pygame.K_r and player_bank <= 0:
                player_bank = 2000
                update_background()
                reset_game()
            elif event.key == pygame.K_m:
                backtomenu()
            elif event.key == pygame.K_q:
                pygame.quit()
                sys.exit()
            elif event.key == pygame.K_UP:
                if not game_started:
                    if player_bid >= player_bank:
                        player_bid = player_bank
                    if player_bid + 1000 <= player_bank:  
                        player_bid += 1000
            elif event.key == pygame.K_DOWN:
                if not game_started:
                    if player_bid >= player_bank:
                        player_bid = player_bank
                    if player_bid - 10 > 0: 
                        player_bid -= 10
            elif event.key == pygame.K_RETURN: 
                # screen.fill((0, 0, 0))  color black
                print(f"Game started with a bid of ${player_bid}")
                game_started = True 
                detection_started = True
                break
            elif event.key == pygame.K_s and game_started and not dealer_turn and not blackjack_reached and not busted_reached:
                player_standing = True
                show_second_card = True
                dealer_turn = True
                dealer_visible_total = sum(dealer_card_values)
                dealer_draw_card()

    update_background()
    screen.blit(background_image, (0, 0))  # Display backgrond image

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
        if current_time - start_time < 0.1:
            bell_sound.play()
    elif blackjack_reached:
        overlay_displayed = blackjack_image_resized
    elif busted_reached:
        overlay_displayed = busted_image_resized
    elif win_reached:
        overlay_displayed = win_image_resized
    elif push_reached:
        overlay_displayed = push_image_resized
    elif lost_reached:
        overlay_displayed = lost_image_resized
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
                # print(f"Prediction: {prediction}")  
                class_id = np.argmax(prediction)
                confidence = prediction[0][class_id]
                # print(f"Class ID: {class_id}, Confidence: {confidence}")  
                
                if confidence > 0.6:
                    if class_id not in detected_cards:
                        if overlay_displayed is None and detection_started is True:
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
        win_reached = True
        dealer_turn = False
    elif total_blackjack_value > 21:
        busted_reached = True
        lost_reached = True
        dealer_turn = False
        

    cv2.putText(frame, f"Value: {total_blackjack_value}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # BIDDING
    display_bidding_area(screen, font, player_bank, player_bid)
    if player_bank <= 0:
        # Display the current bank money in bankrupt mode
        font = pygame.font.Font(None, 36)
        bank_text = font.render(f"Your Bank: ${player_bank}", True, (255, 255, 255))
        screen.blit(bank_text, (60, 170))  # Bank position
        dealer_turn = False
        game_started = False

    dealer_total = calculate_best_value(dealer_card_values)

    if not game_started:
        frame = overlay_image_center(frame, boost_image_resized)
        
        
    # Game logic
    if game_started:
        # screen.fill((0, 0, 0)) 
        display_dealer_area(screen, font)
        display_stand_button(screen, font)

        # Dealer turn logic
        if dealer_turn:
            dealer_total = calculate_best_value(dealer_card_values)
            while dealer_total < 17:
                new_card = pick_card(card_directory)
                dealer_cards.append(new_card[0]) 
                dealer_card_values.append(new_card[1])
                dealer_total = calculate_best_value(dealer_card_values)

            if dealer_total > 21:  # Dealer busted
                win_reached = True
                player_bank += player_bid
                update_background()
            else:
                # Compare final values if both under 21
                if total_blackjack_value > dealer_total:
                    win_reached = True
                    player_bank += player_bid
                    update_background()
                elif total_blackjack_value < dealer_total:
                    lost_reached = True
                    player_bank -= player_bid
                    update_background()
                else:
                    push_reached = True
                    update_background()
            # Display result


            dealer_turn = False  # End dealer turn logic

            # Display dealer and player areas
            # screen.fill((0, 0, 0)) COLOR BLACK
            display_bidding_area(screen, font, player_bank, player_bid)
            display_dealer_area(screen, font)

            # Overlay frame with OpenCV display
            frame_height, frame_width = frame.shape[:2]
            camera_width, camera_height = 560, 320
            frame_resized = cv2.resize(frame, (camera_width, camera_height))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frame_flipped = cv2.flip(frame_rgb, 1)
            frame_surface = pygame.surfarray.make_surface(np.rot90(frame_flipped))
            screen.blit(frame_surface, (20, 450))

            pygame.display.flip()
            clock.tick(30)

    # BIDDING logic
    if not game_started and not detection_started and not win_reached and not lost_reached and not busted_reached and not blackjack_reached:
        display_bidding_area(screen, font, player_bank, player_bid)
        
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