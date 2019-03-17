import cv2
import numpy as np

lower_red1 = np.genfromtxt('lower_red1.csv', delimiter=',')
lower_red2 = np.genfromtxt('lower_red2.csv', delimiter=',')
upper_red1 = np.genfromtxt('upper_red1.csv', delimiter=',')
upper_red2 = np.genfromtxt('upper_red2.csv', delimiter=',')

print("lower_red1 =", lower_red1)
print("lower_red2 =", lower_red2)
print("upper_red1 =", upper_red1)
print("upper_red2 =", upper_red2)