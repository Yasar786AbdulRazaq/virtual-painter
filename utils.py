import cv2
import numpy as np

def draw_color_buttons(img, colors, selected_index):
    for i, color in enumerate(colors):
        x = 20 + i * 60
        y = 10
        thickness = 3 if i == selected_index else 1
        cv2.rectangle(img, (x, y), (x + 40, y + 40), color, thickness)
    return img

def distance(p1, p2):
    return ((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)**0.5
