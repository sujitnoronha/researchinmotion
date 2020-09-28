import cv2
import numpy as np
import ast

file = open('./test.txt','r')

s= file.read()

x = ast.literal_eval(s)

print(type(x),x)