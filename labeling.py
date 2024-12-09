import cv2
import numpy as np

img = cv2.imread('cardGameProject/abstract.png', cv2.IMREAD_GRAYSCALE)
_, binary_image = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

num_labels, labels_im = cv2.connectedComponents(binary_image)
np.savetxt('cardGameProject/labeling.txt', labels_im, fmt='%d')
output_image = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
colors = np.random.randint(0, 255, size=(num_labels, 3))

for label in range(1, num_labels):
    output_image[labels_im == label] = colors[label]

cv2.imshow('Original Image', img)
cv2.imshow('Labeled Image', output_image)
cv2.waitKey(0)  
cv2.destroyAllWindows()  