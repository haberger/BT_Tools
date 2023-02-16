import matplotlib.pyplot as plt
import numpy as np
import cv2


img = cv2.imread("/home/dalina/David/Uni/BachelorThesis/D435 dataset old/scenes/002/gt/00001.png", cv2.IMREAD_UNCHANGED)
print(img)
cv2.imshow("img", img*50)
cv2.waitKey(0)

