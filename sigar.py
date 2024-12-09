import cv2 
import numpy as np

img = cv2.imread('cardGameProject/abstract.png', cv2.IMREAD_GRAYSCALE)
_, binary_image = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

num_labels, labels_im = cv2.connectedComponents(binary_image)
for i in range (num_labels):
    print(i)
    print("baris",b)
    print("kolom",k)

cv2.waitKey(0)
cv2.destroyAllWindows()