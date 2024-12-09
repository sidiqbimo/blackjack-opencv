import cv2
import numpy as np

def drawCircle(image, x, y):
    # Draws a small circle at the specified (x, y) location
    center_coordinates = (x, y)
    radius = 4
    color = (0, 0, 255)  # Red color for the corners
    thickness = 2
    image = cv2.circle(image, center_coordinates, radius, color, thickness)
    return image

# Initialize the camera
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Error: Couldn't open camera.")
    exit(1)

while True:
    ret, frame = cam.read()
    if not ret:
        print("Error: Couldn't capture frame.")
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Set HSV range to detect the color #83827B
    lower_bound = np.array([0, 0, 70])   # Adjusted lower threshold
    upper_bound = np.array([180, 40, 140])  # Adjusted upper threshold
    mask = cv2.inRange(hsv, lower_bound, upper_bound)


    # Invert the mask to keep only the non-green objects (e.g., cards)
    mask = cv2.bitwise_not(mask)

    # Define a kernel and apply morphological transformations to remove noise
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Apply the mask to keep only the detected cards
    foreground = cv2.bitwise_and(frame, frame, mask=mask)

    # Detect connected components to find card-like objects in the mask
    num_labels, labels_im = cv2.connectedComponents(mask)
    for i in range(1, num_labels):
        # Get coordinates of pixels that belong to the current label
        y_coords, x_coords = np.where(labels_im == i)

        # Find the bounding box around the connected component
        y_min, y_max = y_coords.min(), y_coords.max()
        x_min, x_max = x_coords.min(), x_coords.max()

        # Calculate the corner coordinates
        top_left = (x_min, y_min)
        top_right = (x_max, y_min)
        bottom_left = (x_min, y_max)
        bottom_right = (x_max, y_max)

        # Draw circles at each corner of the detected card
        frame = drawCircle(frame, *top_left)
        frame = drawCircle(frame, *top_right)
        frame = drawCircle(frame, *bottom_left)
        frame = drawCircle(frame, *bottom_right)

    # Show the result with only detected cards, the mask, and the isolated foreground
    cv2.imshow("Detected Cards with Lime Green Background Removed", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Foreground", foreground)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cam.release()
cv2.destroyAllWindows()
