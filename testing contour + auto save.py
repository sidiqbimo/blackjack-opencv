import cv2
import numpy as np
import json
import os


def drawCircle(image, x, y):
    center_coordinates = (x, y)
    radius = 4
    color = (0, 0, 255) 
    thickness = 2
    image = cv2.circle(image, center_coordinates, radius, color, thickness)
    return image

def get_hsv_bounds_from_bgr(bgr_color, hue_tolerance=10, saturation_tolerance=40, value_tolerance=40):
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

def pick_color(event, x, y, flags, param): 
    global hsv_lower_bound, hsv_upper_bound, frame

    if event == cv2.EVENT_LBUTTONDOWN:
        bgr_color = frame[y, x]
        hsv_lower_bound, hsv_upper_bound = get_hsv_bounds_from_bgr(bgr_color)
        print("Lower Bound:", hsv_lower_bound)
        print("Upper Bound:", hsv_upper_bound)

        save_hsv_bounds(hsv_lower_bound, hsv_upper_bound)


def save_hsv_bounds(lower_bound, upper_bound):
    bounds = {
        "lower_bound": lower_bound.tolist(),
        "upper_bound": upper_bound.tolist()
    }
    with open("hsv_bounds.json", "w") as file:
        json.dump(bounds, file)

def load_hsv_bounds():
    if os.path.exists("hsv_bounds.json"):
        with open("hsv_bounds.json", "r") as file:
            bounds = json.load(file)
            return (np.array(bounds["lower_bound"]), np.array(bounds["upper_bound"]))
    return None, None

cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Error: Couldn't open camera.")
    exit(1)

hsv_lower_bound, hsv_upper_bound = load_hsv_bounds()
if hsv_lower_bound is None or hsv_upper_bound is None:
    hsv_lower_bound = np.array([0, 0, 70])
    hsv_upper_bound = np.array([180, 40, 140])

cv2.namedWindow("Citra Mentah")
cv2.setMouseCallback("Citra Mentah", pick_color)

while True:
    ret, frame = cam.read()
    if not ret:
        print("Error: Couldn't capture frame.")
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, hsv_lower_bound, hsv_upper_bound)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=3)
    mask = cv2.dilate(mask, kernel, iterations=3)

    mask = cv2.bitwise_not(mask)
    foreground = cv2.bitwise_and(frame, frame, mask=mask) # INVERT

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:

        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) == 4:
            for point in approx:
                x, y = point[0]
                frame = drawCircle(frame, x, y)


    cv2.imshow("Citra Mentah", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Citra Kartu", foreground)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
