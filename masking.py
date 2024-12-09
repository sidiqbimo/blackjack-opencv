import cv2
import numpy as np


def drawCircle(image, b,k):
    center_coordinates = (k,b)
    radius = 4
    color = (0, 0, 255)
    thickness = 2
    image = cv2.circle(frame, center_coordinates, radius, color, thickness)
    return image

# Initialize the camera
cam = cv2.VideoCapture(3)

if not cam.isOpened():
    print("Error: Couldn't open camera.")
    exit(1)
while True:
    ret, frame = cam.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Set HSV range to detect the color #83827B
    lower_bound = np.array([0, 0, 70])   # Adjusted lower threshold
    upper_bound = np.array([180, 40, 140])  # Adjusted upper threshold
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    mask = cv2.bitwise_not(mask)

    kernel = np.array ([[1,1,1],
                        [1,1,1],
                        [1,1,1]], dtype=np.uint8)
    
    # For before erode
    m = mask.copy()
    
    # For erode
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)
    foreground = cv2.bitwise_and(frame, frame, mask=mask)

    num_labels, labels_im = cv2.connectedComponents(mask)
    for i in range (1,num_labels):
        b,k = np.where(labels_im == i)
        bmin = b.min()
        bmax = b.max()
        kmin = k.min()
        kmax = k.max()

        thexbmin = int((bmin+bmax)/2)
        theykmin = int((kmin+kmax)/2)

        xbmin = k[np.where(b == bmin)[0][0]]
        xbmax = k[np.where(b == bmax)[0][0]]
        ykmin = b[np.where(k == kmin)[0][0]]
        ykmax = b[np.where(k == kmax)[0][0]]

        print(i)
        print("baris",bmin,bmax)
        print("kolom",kmin,kmax)

        frame = drawCircle(frame, bmin, xbmin)
        frame = drawCircle(frame, bmax, xbmax)
        frame = drawCircle(frame, kmin, ykmin)
        frame = drawCircle(frame, kmax, ykmax)

    if not ret :
        print("Error: Couldn't capture frame.")
        break
    
    cv2.imshow("Original", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Foreground", foreground)
    cv2.imshow("Before Erode", m)
    # cv2.imshow("Erode", mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break