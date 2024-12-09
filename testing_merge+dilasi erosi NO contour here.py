import cv2
import numpy as np
import math

def drawCircle(image, x, y):
    # Draws a small circle at the specified (x, y) location
    center_coordinates = (x, y)
    radius = 4
    color = (0, 0, 255)  # Red color for the corners
    thickness = 2
    image = cv2.circle(image, center_coordinates, radius, color, thickness)
    return image

def get_hsv_bounds_from_bgr(bgr_color, hue_tolerance=10, saturation_tolerance=40, value_tolerance=40):
    # Convert BGR to HSV
    bgr_color = np.uint8([[bgr_color]])
    hsv_color = cv2.cvtColor(bgr_color, cv2.COLOR_BGR2HSV)[0][0]

    # Define lower and upper bounds with tolerance
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

# Callback function to capture color on mouse click
def pick_color(event, x, y, flags, param):
    global hsv_lower_bound, hsv_upper_bound, frame

    if event == cv2.EVENT_LBUTTONDOWN:
        # Capture the color at the clicked position
        bgr_color = frame[y, x]
        hsv_lower_bound, hsv_upper_bound = get_hsv_bounds_from_bgr(bgr_color)
        print("Lower Bound:", hsv_lower_bound)
        print("Upper Bound:", hsv_upper_bound)

# Initialize the camera
cam = cv2.VideoCapture(3)
if not cam.isOpened():
    print("Error: Couldn't open camera.")
    exit(1)

# Set initial HSV bounds
hsv_lower_bound = np.array([0, 0, 70])
hsv_upper_bound = np.array([180, 40, 140])

# Set up the window and mouse callback for color picking
cv2.namedWindow("Camera Feed")
cv2.setMouseCallback("Camera Feed", pick_color)

while True:
    ret, frame = cam.read()
    if not ret:
        print("Error: Couldn't capture frame.")
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create mask with the dynamically set bounds
    mask = cv2.inRange(hsv, hsv_lower_bound, hsv_upper_bound)


    # Refine the mask using erosion and dilation to clean up noise
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)


    # Invert the mask to isolate only non-background objects (e.g., cards)
    mask = cv2.bitwise_not(mask)
    foreground = cv2.bitwise_and(frame, frame, mask=mask)
    
    num_labels, labels_im = cv2.connectedComponents(mask)
    for i in range(1, num_labels):
        b, k = np.where(labels_im == i)
        
        # Find the four corners based on extreme points
        top_left = (k.min(), b.min())          # Top-left corner (min x, min y)
        top_right = (k.max(), b.min())         # Top-right corner (max x, min y)
        bottom_left = (k.min(), b.max())       # Bottom-left corner (min x, max y)
        bottom_right = (k.max(), b.max())      # Bottom-right corner (max x, max y)

        # Draw circles at each detected corner
        frame = drawCircle(frame, *top_left)
        frame = drawCircle(frame, *top_right)
        frame = drawCircle(frame, *bottom_left)
        frame = drawCircle(frame, *bottom_right)



    # Display the frames
    cv2.imshow("Camera Feed", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Foreground", foreground)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cam.release()
cv2.destroyAllWindows()
