import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn import datasets
from sklearn.model_selection import train_test_split

img = []
for i in range(1, 7):
    img.append(cv2.imread('img/Expo_' + str(pow(10, i)) + '/Gain_12.bmp'))

# caculate the total mean value of each RGB channel
def RGB_mean(img):
    b, g, r = cv2.split(img) # split the image into three channels
    r_mean = np.mean(r)
    g_mean = np.mean(g)
    b_mean = np.mean(b)
    return r_mean, g_mean, b_mean

# caculate the total mean value of gray channel
def GRAY_mean(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_mean = np.mean(gray)
    return gray_mean

# caculate the total mean value of each HSV channel
def HSV_mean(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    h_mean = np.mean(h)
    s_mean = np.mean(s)
    v_mean = np.mean(v)
    return h_mean, s_mean, v_mean

# caculate the total mean value of each YCrCb channel
def YCrCb_mean(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_mean = np.mean(y)
    cr_mean = np.mean(cr)
    cb_mean = np.mean(cb)
    return y_mean, cr_mean, cb_mean

r_mean = []
g_mean = []
b_mean = []
for i in range(0, 6):
    r, g, b = RGB_mean(img[i])
    r_mean.append(r)
    g_mean.append(g)
    b_mean.append(b)

# convert list to array
r_mean = np.array(r_mean)
g_mean = np.array(g_mean)
b_mean = np.array(b_mean)

# use sklearn to fit the data
def Value_LinearRegression(x, y, order):
    X = []
    for i in range(0, order+1):
        X.append(x**i)
    X = np.array(X)
    X = X.T
    model = LinearRegression(fit_intercept=False)
    model.fit(X,y)
    W = model.coef_
    y_hat = model.predict(X)
    return W, y_hat

# red channel
y = r_mean.T
x = np.array([1, 2, 3, 4, 5, 6]) # log exposure time
order = 3
W, y_hat = Value_LinearRegression(x, y, order)
print(W)

# draw the curve
plt.figure()
plt.suptitle("RGB channel")

plt.subplot(3, 1, 1)
plt.title("Red channel")
plt.scatter(x, y, s=10, c="orange")
plt.scatter(x, y_hat, s=10, c="k")

x2 = []
y_hat2 = []
b = 1
for i in range(50):
    x2.append(b)
    b += 5/49
    y_hat2.append(W[0] + W[1]*b + W[2]*b*b + W[3]*b*b*b)
plt.plot(x2, y_hat2, c="r")

# green channel
y = g_mean.T
x = np.array([1, 2, 3, 4, 5, 6])
order = 3
W, y_hat = Value_LinearRegression(x, y, order)
print(W)

# draw the curve
plt.subplot(3, 1, 2)
plt.title("Green channel")
plt.scatter(x, y, s=10, c="orange")
plt.scatter(x, y_hat, s=10, c="k")

x2 = []
y_hat2 = []
b = 1
for i in range(50):
    x2.append(b)
    b += 5/49
    y_hat2.append(W[0] + W[1]*b + W[2]*b*b + W[3]*b*b*b)
plt.plot(x2, y_hat2, c="g")

# blue channel
y = b_mean.T
x = np.array([1, 2, 3, 4, 5, 6])
order = 3
W, y_hat = Value_LinearRegression(x, y, order)
print(W)

# draw the curve
plt.subplot(3, 1, 3)
plt.title("Blue channel")
plt.scatter(x, y, s=10, c="orange")
plt.scatter(x, y_hat, s=10, c="k")

x2 = []
y_hat2 = []
b = 1
for i in range(50):
    x2.append(b)
    b += 5/49
    y_hat2.append(W[0] + W[1]*b + W[2]*b*b + W[3]*b*b*b)
plt.plot(x2, y_hat2, c="b")

plt.show()

# gray channel
gray_mean = []
for i in range(0, 6):
    gray_mean.append(GRAY_mean(img[i]))

gray_mean = np.array(gray_mean)
print(gray_mean)

# use sklearn to fit the data
y = gray_mean.T
x = np.array([1, 2, 3, 4, 5, 6])
order = 3
W, y_hat = Value_LinearRegression(x, y, order)

# draw the curve
plt.figure()
plt.title("Gray channel")
plt.scatter(x, y, s=10, c="orange")
plt.scatter(x, y_hat, s=10, c="k")

x2 = []
y_hat2 = []
b = 1
for i in range(50):
    x2.append(b)
    b += 5/49
    y_hat2.append(W[0] + W[1]*b + W[2]*b*b + W[3]*b*b*b)
plt.plot(x2, y_hat2, c="gray")
plt.show()

# HSV channel
h_mean = []
s_mean = []
v_mean = []

for i in range(0, 6):
    h, s, v = HSV_mean(img[i])
    h_mean.append(h)
    s_mean.append(s)
    v_mean.append(v)

h_mean = np.array(h_mean)
s_mean = np.array(s_mean)
v_mean = np.array(v_mean)

print(h_mean)
print(s_mean)
print(v_mean)

# H channel
x = np.array([1, 2, 3, 4, 5, 6])
y = h_mean.T
order = 4
W, y_hat = Value_LinearRegression(x, y, order)

# draw the curve
plt.figure()
plt.suptitle("HSV channel")

plt.subplot(3, 1, 1)
plt.title("H channel")
plt.scatter(x, y, s=10, c="orange")
plt.scatter(x, y_hat, s=10, c="k")

x2 = []
y_hat2 = []
b = 1
for i in range(50):
    x2.append(b)
    b += 5/49
    y_hat2.append(W[0] + W[1]*b + W[2]*b*b + W[3]*b*b*b + W[4]*b*b*b*b)
plt.plot(x2, y_hat2, c="r")

# S channel
y = s_mean.T
x = np.array([1, 2, 3, 4, 5, 6])
order = 4
W, y_hat = Value_LinearRegression(x, y, order)

# draw the curve
plt.subplot(3, 1, 2)
plt.title("S channel")
plt.scatter(x, y, s=10, c="orange")
plt.scatter(x, y_hat, s=10, c="k")

x2 = []
y_hat2 = []
b = 1
for i in range(50):
    x2.append(b)
    b += 5/49
    y_hat2.append(W[0] + W[1]*b + W[2]*b*b + W[3]*b*b*b + W[4]*b*b*b*b)

plt.plot(x2, y_hat2, c="g")

# V channel
y = v_mean.T
x = np.array([1, 2, 3, 4, 5, 6])
order = 4
W, y_hat = Value_LinearRegression(x, y, order)

# draw the curve
plt.subplot(3, 1, 3)
plt.title("V channel")
plt.scatter(x, y, s=10, c="orange")
plt.scatter(x, y_hat, s=10, c="k")


x2 = []
y_hat2 = []
b = 1
for i in range(50):
    x2.append(b)
    b += 5/49
    y_hat2.append(W[0] + W[1]*b + W[2]*b*b + W[3]*b*b*b + W[4]*b*b*b*b)
plt.plot(x2, y_hat2, c="b")

plt.show()

# YCrCb channel
y_mean = []
cr_mean = []
cb_mean = []

for i in range(0, 6):
    y, cr, cb = YCrCb_mean(img[i])
    y_mean.append(y)
    cr_mean.append(cr)
    cb_mean.append(cb)

y_mean = np.array(y_mean)
cr_mean = np.array(cr_mean)
cb_mean = np.array(cb_mean)

print(y_mean)
print(cr_mean)
print(cb_mean)

# Y channel
x = np.array([1, 2, 3, 4, 5, 6])
y = y_mean.T
order = 3
W, y_hat = Value_LinearRegression(x, y, order)

# draw the curve
plt.figure()
plt.suptitle("YCrCb channel")

plt.subplot(3, 1, 1)
plt.title("Y channel")
plt.scatter(x, y, s=10, c="orange")
plt.scatter(x, y_hat, s=10, c="k")

x2 = []
y_hat2 = []
b = 1

for i in range(50):
    x2.append(b)
    b += 5/49
    y_hat2.append(W[0] + W[1]*b + W[2]*b*b + W[3]*b*b*b)

plt.plot(x2, y_hat2, c="r")

# Cr channel
y = cr_mean.T
x = np.array([1, 2, 3, 4, 5, 6])
order = 0
W, y_hat = Value_LinearRegression(x, y, order)

# draw the curve
plt.subplot(3, 1, 2)
plt.title("Cr channel")
plt.scatter(x, y, s=10, c="orange")
plt.scatter(x, y_hat, s=10, c="k")

x2 = []
y_hat2 = []
b = 1

for i in range(50):
    x2.append(b)
    b += 5/49
    y_hat2.append(W[0])

plt.plot(x2, y_hat2, c="g")

# Cb channel
y = cb_mean.T
x = np.array([1, 2, 3, 4, 5, 6])
order = 0
W, y_hat = Value_LinearRegression(x, y, order)

# draw the curve
plt.subplot(3, 1, 3)
plt.title("Cb channel")
plt.scatter(x, y, s=10, c="orange")
plt.scatter(x, y_hat, s=10, c="k")

x2 = []
y_hat2 = []
b = 1

for i in range(50):
    x2.append(b)
    b += 5/49
    y_hat2.append(W[0])

plt.plot(x2, y_hat2, c="b")

plt.show()

#————————————————————————————————————————————————————————————————————————————————————————————————————————#
# analysis on subimage
height, width, channels = img[0].shape
subheight = height // 3
subwidth = width // 3

subimg = []
for i in range(0, 6):
    small = []
    for j in range(0, 3):
        for k in range(0, 3):
            small.append(img[i][j*subheight:(j+1)*subheight, k*subwidth:(k+1)*subwidth])
    subimg.append(small)
# segment way:
# 1 2 3
# 4 5 6
# 7 8 9

# caculate the mean value of each subimage
def RGB_subimg_mean(subimg):
    r_mean = []
    g_mean = []
    b_mean = []
    for i in range(0, 9):
        r, g, b = RGB_mean(subimg[i])
        r_mean.append(r)
        g_mean.append(g)
        b_mean.append(b)
    return r_mean, g_mean, b_mean

def GRAY_subimg_mean(subimg):
    gray_mean = []
    for i in range(0, 9):
        gray_mean.append(GRAY_mean(subimg[i]))
    return gray_mean

def HSV_subimg_mean(subimg):
    h_mean = []
    s_mean = []
    v_mean = []
    for i in range(0, 9):
        h, s, v = HSV_mean(subimg[i])
        h_mean.append(h)
        s_mean.append(s)
        v_mean.append(v)
    return h_mean, s_mean, v_mean

def YCrCb_subimg_mean(subimg):
    y_mean = []
    cr_mean = []
    cb_mean = []
    for i in range(0, 9):
        y, cr, cb = YCrCb_mean(subimg[i])
        y_mean.append(y)
        cr_mean.append(cr)
        cb_mean.append(cb)
    return y_mean, cr_mean, cb_mean

# Red channel
r_mean = []
g_mean = []
b_mean = []
for i in range(0, 6):
    r, g, b = RGB_subimg_mean(subimg[i])
    r_mean.append(r)
    g_mean.append(g)
    b_mean.append(b)

r_mean = np.array(r_mean)
g_mean = np.array(g_mean)
b_mean = np.array(b_mean)

x = np.array([1, 2, 3, 4, 5, 6])
y = []
order = 3
for i in range(9):
    y.append(r_mean[:, i])

# use sklearn to fit the data
W = []
y_hat = []
for i in range(9):
    W1, y_hat1 = Value_LinearRegression(x, y[i], order)
    W.append(W1)
    y_hat.append(y_hat1)

# draw the curve
plt.figure()
plt.suptitle("Red")

for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.title("subimage "+str(i+1))
    plt.scatter(x, y[i], s=10, c="orange")
    plt.scatter(x, y_hat[i], s=10, c="k")
    x2 = []
    y_hat2 = []
    b = 1
    for j in range(50):
        x2.append(b)
        b += 5 / 49
        y_hat2.append(W[i][0] + W[i][1]*b + W[i][2]*b*b + W[i][3]*b*b*b)

    plt.plot(x2, y_hat2, c="r")
plt.show()

# Gray channel
gray_mean = []
for i in range(0, 6):
    gray_mean.append(GRAY_subimg_mean(subimg[i]))

gray_mean = np.array(gray_mean)

x = np.array([1, 2, 3, 4, 5, 6])
y = []
order = 3

for i in range(9):
    y.append(gray_mean[:, i])

# use sklearn to fit the data
W = []
y_hat = []

for i in range(9):
    W1, y_hat1 = Value_LinearRegression(x, y[i], order)
    W.append(W1)
    y_hat.append(y_hat1)

# draw the curve
plt.figure()
plt.suptitle("Gray")

for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.title("subimage "+str(i+1))
    plt.scatter(x, y[i], s=10, c="orange")
    plt.scatter(x, y_hat[i], s=10, c="k")
    x2 = []
    y_hat2 = []
    b = 1
    for j in range(50):
        x2.append(b)
        b += 5 / 49
        y_hat2.append(W[i][0] + W[i][1]*b + W[i][2]*b*b + W[i][3]*b*b*b)

    plt.plot(x2, y_hat2, c="gray")
plt.show()

# S channel
h_mean = []
s_mean = []
v_mean = []

for i in range(0, 6):
    h, s, v = HSV_subimg_mean(subimg[i])
    h_mean.append(h)
    s_mean.append(s)
    v_mean.append(v)

h_mean = np.array(h_mean)
s_mean = np.array(s_mean)
v_mean = np.array(v_mean)

x = np.array([1, 2, 3, 4, 5, 6])
y = []
order = 3

for i in range(9):
    y.append(s_mean[:, i])

# use sklearn to fit the data
W = []
y_hat = []

for i in range(9):
    W1, y_hat1 = Value_LinearRegression(x, y[i], order)
    W.append(W1)
    y_hat.append(y_hat1)

# draw the curve
plt.figure()
plt.suptitle("S")

for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.title("subimage "+str(i+1))
    plt.scatter(x, y[i], s=10, c="orange")
    plt.scatter(x, y_hat[i], s=10, c="k")
    x2 = []
    y_hat2 = []
    b = 1
    for j in range(50):
        x2.append(b)
        b += 5 / 49
        y_hat2.append(W[i][0] + W[i][1]*b + W[i][2]*b*b + W[i][3]*b*b*b)

    plt.plot(x2, y_hat2, c="g")
plt.show()

# V channel
x = np.array([1, 2, 3, 4, 5, 6])
y = []
order = 3

for i in range(9):
    y.append(v_mean[:, i])

# use sklearn to fit the data
W = []
y_hat = []

for i in range(9):
    W1, y_hat1 = Value_LinearRegression(x, y[i], order)
    W.append(W1)
    y_hat.append(y_hat1)

# draw the curve
plt.figure()
plt.suptitle("V")

for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.title("subimage "+str(i+1))
    plt.scatter(x, y[i], s=10, c="orange")
    plt.scatter(x, y_hat[i], s=10, c="k")
    x2 = []
    y_hat2 = []
    b = 1
    for j in range(50):
        x2.append(b)
        b += 5 / 49
        y_hat2.append(W[i][0] + W[i][1]*b + W[i][2]*b*b + W[i][3]*b*b*b)

    plt.plot(x2, y_hat2, c="b")
plt.show()

# Y channel
y_mean = []
cr_mean = []
cb_mean = []

for i in range(0, 6):
    y, cr, cb = YCrCb_subimg_mean(subimg[i])
    y_mean.append(y)
    cr_mean.append(cr)
    cb_mean.append(cb)

y_mean = np.array(y_mean)
cr_mean = np.array(cr_mean)
cb_mean = np.array(cb_mean)

x = np.array([1, 2, 3, 4, 5, 6])
y = []
order = 3

for i in range(9):
    y.append(y_mean[:, i])

# use sklearn to fit the data
W = []
y_hat = []

for i in range(9):
    W1, y_hat1 = Value_LinearRegression(x, y[i], order)
    W.append(W1)
    y_hat.append(y_hat1)

# draw the curve
plt.figure()
plt.suptitle("Y")

for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.title("subimage "+str(i+1))
    plt.scatter(x, y[i], s=10, c="orange")
    plt.scatter(x, y_hat[i], s=10, c="k")
    x2 = []
    y_hat2 = []
    b = 1
    for j in range(50):
        x2.append(b)
        b += 5 / 49
        y_hat2.append(W[i][0] + W[i][1]*b + W[i][2]*b*b + W[i][3]*b*b*b)

    plt.plot(x2, y_hat2, c="r")
plt.show()