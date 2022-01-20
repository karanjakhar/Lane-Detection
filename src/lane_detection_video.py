import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny_edges(gray_img, low_threshold, high_threshold, kernel_size=5):
    blur_gray = cv2.GaussianBlur(gray_img, (kernel_size, kernel_size), 0)
    return cv2.Canny(blur_gray, low_threshold, high_threshold)


def hough_transform(img, masked_edges, rho=1,
                    theta=np.pi/180,
                    threshold=15,
                    min_line_length=40,
                    max_line_gap=20):
    
    line_image = np.copy(img)*0
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold,
                        np.array([]), min_line_length, max_line_gap)


    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2,y2), (0,0,255), 10)
    
    return line_image


def draw_lines(img, line_image):
    annotated_img = cv2.addWeighted(img, 0.8, line_image, 1, 0)
    return annotated_img

def region_of_interest(edges):
    imshape = edges.shape
    vertices = np.array([[(0, imshape[0]), (400, 300),
                    (imshape[1], imshape[0])]], dtype=np.int32)
    mask = np.zeros_like(edges)
    ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_edges = cv2.bitwise_and(edges, mask)
    return masked_edges



filename = '/home/karan/kj_workspace/kj_learnings/self-driving/code/lane detection/solidYellowLeft.mp4'

cap = cv2.VideoCapture(filename)

output_filename = filename.split('/')[-1].split('.')[0] + '_output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter(output_filename,fourcc, fps, (width,height))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = canny_edges(gray_img,120,240)
    masked_edges = region_of_interest(edges)
    line_img = hough_transform(frame, masked_edges)

    annotated_img = draw_lines(frame, line_img)

    writer.write(annotated_img)



