import cv2
import numpy as np

def get_hsv_bounds_from_bgr(bgr_color, hue_tolerance=10, saturation_tolerance=40, value_tolerance=40):
    # Convert the BGR color to HSV
    bgr_color = np.uint8([[bgr_color]])
    hsv_color = cv2.cvtColor(bgr_color, cv2.COLOR_BGR2HSV)[0][0]

    # Set lower and upper bounds with tolerance
    lower_bound = np.array([
        max(0, hsv_color[0] - hue_tolerance),          # Hue
        max(0, hsv_color[1] - saturation_tolerance),    # Saturation
        max(0, hsv_color[2] - value_tolerance)          # Value
    ])
    
    upper_bound = np.array([
        min(180, hsv_color[0] + hue_tolerance),         # Hue
        min(255, hsv_color[1] + saturation_tolerance),  # Saturation
        min(255, hsv_color[2] + value_tolerance)        # Value
    ])

    return lower_bound, upper_bound

# Callback function to capture color on mouse click
def pick_color(event, x, y, flags, param):
    global hsv_lower_bound, hsv_upper_bound, frame
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Get the color of the pixel at (x, y) in BGR format
        bgr_color = frame[y, x]
        # Get HSV bounds based on the clicked color
        hsv_lower_bound, hsv_upper_bound = get_hsv_bounds_from_bgr(bgr_color)
        print("Lower Bound:", hsv_lower_bound)
        print("Upper Bound:", hsv_upper_bound)

# Initialize camera
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Error: Couldn't open camera.")
    exit(1)

# Set default HSV bounds
hsv_lower_bound = np.array([0, 0, 0])
hsv_upper_bound = np.array([180, 255, 255])

# Set up window and callback function for color picking
cv2.namedWindow("Camera Feed")
cv2.setMouseCallback("Camera Feed", pick_color)

while True:
    ret, frame = cam.read()
    if not ret:
        print("Error: Couldn't capture frame.")
        break

    # Convert frame to HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Apply the mask using dynamically updated bounds
    mask = cv2.inRange(hsv_frame, hsv_lower_bound, hsv_upper_bound)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Display frames
    cv2.imshow("Camera Feed", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Filtered Color", result)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cam.release()
cv2.destroyAllWindows()
