import pygame
import pygame_menu
from pygame_menu.baseimage import BaseImage, IMAGE_MODE_FILL
import subprocess
import sys

pygame.init()
surface = pygame.display.set_mode((600, 500))

# Background music
bg_music_path = "D:\\Programming\\Python\\cardGameProject\\illust\\bg.mp3"
bg_sound = pygame.mixer.Sound(bg_music_path)
bg_sound.play(loops=-1)

bg_image_path = "D:\\Programming\\Python\\cardGameProject\\illust\\the.png"
bg_image = BaseImage(
    image_path=bg_image_path,
    drawing_mode=IMAGE_MODE_FILL 
)

casino_theme = pygame_menu.themes.Theme(
    background_color=bg_image,
    title_background_color=(0, 100, 0),  # Dark green
    title_font=pygame_menu.font.FONT_HELVETICA,
    title_bar_style=pygame_menu.widgets.MENUBAR_STYLE_ADAPTIVE,
    widget_font=pygame_menu.font.FONT_HELVETICA,
    widget_font_color=(255, 255, 255),  # White
    widget_background_color=(0, 128, 0),  # Green
    widget_margin=(10, 10),
    widget_padding=10
)

menu = pygame_menu.Menu(
    title='Let\'s play!',
    width=600,
    height=500,
    theme=casino_theme
)

def gamestart():
    surface.fill((0, 0, 0)) 
    font = pygame.font.Font(pygame_menu.font.FONT_HELVETICA, 36)  
    text = font.render("Preparing the deck...", True, (255, 255, 255))  
    text_rect = text.get_rect(center=(surface.get_width() // 2, surface.get_height() // 2))  
    surface.blit(text, text_rect)  
    pygame.display.update()
    
    pygame.time.wait(3000)  
    
    pygame.quit()  

    subprocess.run(["python", "D:/Programming/Python/cardGameProject/model_detector_blackjack.py"])

    sys.exit() 

def gamecrazy():
    surface.fill((0, 0, 0)) 
    font = pygame.font.Font(pygame_menu.font.FONT_HELVETICA, 36)  
    text = font.render("Please wait...", True, (255, 255, 255))  
    text_rect = text.get_rect(center=(surface.get_width() // 2, surface.get_height() // 2))  
    surface.blit(text, text_rect)  
    pygame.display.update()
    
    pygame.time.wait(3000)  
    
    pygame.quit()  

    subprocess.run(["python", "D:/Programming/Python/cardGameProject/testingheart.py"])

    sys.exit() 

def guidemenu():
    surface.fill((0, 0, 0)) 
    font = pygame.font.Font(pygame_menu.font.FONT_HELVETICA, 36)  
    text = font.render("Please wait...", True, (255, 255, 255))  
    text_rect = text.get_rect(center=(surface.get_width() // 2, surface.get_height() // 2))  
    surface.blit(text, text_rect)  
    pygame.display.update()
    
    pygame.quit()  

    subprocess.run(["python", "D:/Programming/Python/cardGameProject/guide.py"])

    sys.exit() 

def gamecalibrate():
    surface.fill((0, 0, 0)) 
    font = pygame.font.Font(pygame_menu.font.FONT_HELVETICA, 36)  
    text = font.render("Please wait...", True, (255, 255, 255))  
    text_rect = text.get_rect(center=(surface.get_width() // 2, surface.get_height() // 2))  
    surface.blit(text, text_rect)  
    pygame.display.update()
    
    pygame.time.wait(3000)  
    
    pygame.quit()  

    subprocess.run(["python", "D:/Programming/Python/cardGameProject/model_detector_base.py"])

    sys.exit() 

menu.add.button('Play Blackjack', gamestart)
menu.add.button('Play Traditional Heart', gamecrazy)
menu.add.button('Calibrate', gamecalibrate)
menu.add.button('Guide', guidemenu)
menu.add.button('Quit', pygame_menu.events.EXIT)

# main loop
menu.mainloop(surface)
