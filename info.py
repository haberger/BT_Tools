import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("/home/david/UNI/BachelorThesis/scenes/003/depth/000015.png", cv2.IMREAD_UNCHANGED)
print(img)


img = cv2.normalize(img, np.zeros_like(img), 0, 255, cv2.NORM_MINMAX)
img = cv2.resize(img, (1104,621))
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

plt.hist(img.ravel(), 256, [0,256])
plt.show()

print(img.dtype)