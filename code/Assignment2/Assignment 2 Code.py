# Assignment 2
# Jian Zhang, student ID: 219012058
# Date: 19.10.15

# import necessary library
import numpy as np
import math
import queue
import cv2 # use opencv to read image
import matplotlib.pyplot as plt # use matplotlib to show image

# Exercise 1: Straight Line Detection

# (a) Canny edge detector

# read gray image
img_house = cv2.imread('House.png', 0)

img_blur = cv2.boxFilter(img_house, -1, (7, 7), normalize=True)
img_blur = cv2.GaussianBlur(img_blur, (17, 17), 0)
img_canny = cv2.Canny(img_blur, 23, 27)

cv2.imwrite('canny.jpg', img_canny)

# (b) implement the Hough transformation to detect the straight line 

def HoughLines(edge):
    
    height, width = edge.shape
    tmp = int(math.sqrt(width**2 + height**2))
    M = np.zeros((tmp, 181), dtype = np.int)
    lines = []
    
    # calculte Hesse normal form for every pixel
    for x in range(height):
        for y in range(width):
            if edge[x][y] == 255:
                for theta in range(181):
                    rho = int(x*math.sin(theta*math.pi/180) + y*math.cos(theta*math.pi/180))
                    M[rho, theta] += 1

    
    # accumulator
    for rho in range(-tmp, tmp):
        for theta in range(181):
            if M[rho, theta] > 120:
                lines += [[rho, theta*math.pi/180]]

    return lines

lines = HoughLines(img_canny)

# draw lines on the origin House.png
img_lines = img_house.copy()
for (rho,theta) in lines:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(img_lines,(x1,y1),(x2,y2),(0,0,255),2)

cv2.imwrite('lines.jpg', img_lines)

# Exercise 2

# (a) implement the boundary following algorithm

# read boundary test image
img_bound = cv2.imread('./Boundary_Following.png', 0)

def ClockWise(target, prev):
    # clockwise offset
    offset = {
                  (0,-1) : (-1,-1),
                  (-1,-1): (-1,0),
                  (-1,0) : (-1,1),
                  (-1,1) : (0,1),
                  (0,1)  : (1,1),
                  (1,1)  : (1,0),
                  (1,0)  : (1,-1),
                  (1,-1) : (0,-1)
             }
    
    tmp = (prev[0] - target[0], prev[1] - target[1])
    return (offset[tmp][0] + target[0], offset[tmp][1] + target[1])

def BoundaryFollow(input_img):
    
    img = input_img.copy()
    # store the points of boundary
    points = []

    # h denotes height, w denotes width
    h, w = img.shape
    # Find uppermost and leftmost point as initial point b0:(h0, w0)
    for i in range(h - 1, -1, -1):
        for j in range(w - 1, -1, -1):
            if img[i, j]:
                b0 = (i, j)
                break
    
    points.append(b0)
    c0 = (b0[0], b0[1] - 1)
    prev = c0
    boundary = b0
    
    curr = ClockWise(boundary, prev)
    
    while (curr != b0):
        
        if img[curr[0], curr[1]]:
            points.append(curr)
            prev = boundary
            boundary = curr
            curr = ClockWise(boundary, prev)
        else:
            prev = curr
            curr = ClockWise(boundary, prev)
    
    return points


points = BoundaryFollow(img_bound)

img_shape = np.zeros(img_bound.shape)

for point in points:
    img_shape[point[0], point[1]] = 255

cv2.imwrite('points.jpg', img_shape)

plt.imshow(img_shape,cmap="gray")
plt.axis('off')
plt.show()


img_paint = img_bound.copy()
def FloodFilling(node):
    
    x = node[0]
    y = node[1]
    
    if img_paint[x, y] == 255:
        return
    
    img_paint[x, y] = 255
    q = queue.Queue()
    q.put(node)
    
    while not q.empty():
        
        n = q.get()
        x = n[0]
        y = n[1]
        
        if img_paint[x - 1, y] == 0:
            img_paint[x - 1, y] = 255
            q.put((x - 1, y))
        if img_paint[x + 1, y] == 0:
            img_paint[x + 1, y] = 255
            q.put((x + 1, y))
        if img_paint[x, y - 1] == 0:
            img_paint[x, y - 1] = 255
            q.put((x, y - 1))
        if img_paint[x, y + 1] == 0:
            img_paint[x, y + 1] = 255
            q.put((x, y + 1))
    
    return

mid_x = int(0.5 * img_paint.shape[0])
mid_y = int(0.5 * img_paint.shape[1])
FloodFilling((mid_x, mid_y))


cv2.imwrite('paint.jpg', img_paint)


# Exercise 3: Morphological Image Processing


# parameter: gray image, kerner size(same dimension), e.g 5*5, 11*11
def Erosion(img, kernel):
    
    k_size = len(kernel)
    h, w = img.shape
    offset = k_size // 2
    img_pad = np.pad(img, ((offset,offset), (offset,offset)), 'constant')

    new_img = np.zeros(img.shape)
    
    for i in range(offset, h - offset):
        for j in range(offset, w - offset):
            # assume current pixel is white, check around pixels
            # if any pixel around does not equal to kernel, than set current pixel black
            new_img[i - offset, j - offset] = 255
            flag = True
            for k in range(-offset, offset):
                if not flag:
                    break
                for l in range(-offset, offset):
                    if img_pad[i + k][j + l] != 255 and img_pad[i + k][j + l] != (kernel[k+offset][l+offset] * 255):
                        new_img[i - offset, j - offset] = 0
                        flag = False
                        break
    
    return new_img

def Dilation(img, kernel):
    
    k_size = len(kernel)
    h, w = img.shape
    offset = k_size // 2
    img_pad = np.pad(img, ((offset,offset), (offset,offset)), 'constant')

    new_img = np.zeros(img.shape)
    
    for i in range(offset, h - offset):
        for j in range(offset, w - offset):
            # it's like inverse operation of erosion
            # assume current pixel is black first, check around pixels
            # if any pixel around equals to kernel, than set current pixel white
            new_img[i - offset, j - offset] = 0
            flag = True
            for k in range(-offset, offset):
                if not flag:
                    break
                for l in range(-offset, offset):
                    if img_pad[i + k][j + l] and img_pad[i + k][j + l] == (kernel[k+offset][l+offset] * 255):
                        new_img[i - offset, j - offset] = 255
                        flag = False
                        break
    
    return new_img


# (a) implement Opening operation to eliminate the most thinnest lines between different objects
#     on the image of Wirebond.png, but other lines should be kept
# (b) implement Closing operation make the text on the image of Text.png look better
img_wirebond = cv2.imread('./Wirebond.png', 0)
img_text = cv2.imread('./Text.png', 0)


# choose 3*3 all elements are 1 kernel
kernel_3 = [ [1]*3 for i in range(3)]
# Opening operation, apply erosion first then apply dilation by same kernel
# After that, the most thinnest lines are eliminated
wirebond_erosion = Erosion(img_wirebond, kernel_3)
wirebond_dilation = Dilation(wirebond_erosion, kernel_3)

cv2.imwrite('ErosionWirebond.jpg', wirebond_erosion)

cv2.imwrite('OpeningOperation.jpg', wirebond_dilation)


kernel_text = [ [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0]]
text_dilation = Dilation(img_text, kernel_text)
text_erosion = Erosion(text_dilation, kernel_text)

cv2.imwrite('DilationText.jpg', text_dilation)

cv2.imwrite('ClosingOperation.jpg', text_erosion)
