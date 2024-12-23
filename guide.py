import pygame
import sys
import subprocess

pygame.init()

# Screen setup
window_size = (600, 400)
screen = pygame.display.set_mode(window_size)
pygame.display.set_caption("How to Play Blackjack")

# Load background image
bg_image = pygame.image.load("D://Programming//Python//cardGameProject//illust//guide_bg.png")
bg_image = pygame.transform.scale(bg_image, window_size)

# Font setup
font = pygame.font.Font(None, 28)

# Instruction text
menu_items = [
    "How to Play Blackjack:",
    "",
    "1. Place your bet using the UP and DOWN arrow keys.",
    "   Press ENTER to confirm your bet.",
    "",
    "2. The game will start and cards will be dealt.",
    "   Your goal is to get as close to 21 as possible.",
    "",
    "3. Press SPACE to hit (get another card).",
    "   Press S to stand (stop drawing cards).",
    "",
    "4. The dealer will reveal their cards and draw until",
    "   they reach at least 17. Whoever is closer to 21 wins.",
    "",
    "5. If your total exceeds 21, you lose (busted).",
    "",
    "6. Press R to restart the game if you lose all your money.",
    "",
    "Press Q to quit the game or M to return to the menu.",
]
scroll_offset = 0  # Tracks how far the text has scrolled

# Back to menu function
def backtomenu():
    pygame.quit()  
    subprocess.run(["python", "D:/Programming/Python/cardGameProject/blackjack_mainmenu.py"])
    sys.exit()

# Scrollable menu rendering
def render_menu():
    screen.blit(bg_image, (0, 0))  # Draw background image

    y_offset = 20 - scroll_offset  # Adjust based on scroll offset
    for item in menu_items:
        color = (255, 255, 255)  # White text
        text = font.render(item, True, color)
        x = (window_size[0] - text.get_width()) // 2
        screen.blit(text, (x, y_offset))
        y_offset += 30  # Space between lines

    pygame.display.flip()

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:  # Quit
                pygame.quit()
                sys.exit()
            elif event.key == pygame.K_m:  # Back to menu
                backtomenu()
            elif event.key == pygame.K_DOWN:  # Scroll down
                scroll_offset += 20
            elif event.key == pygame.K_UP:  # Scroll up
                scroll_offset = max(0, scroll_offset - 20)

    render_menu()

pygame.quit()
sys.exit()
