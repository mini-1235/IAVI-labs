import cv2
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

img = []
for i in range(0, 12):
    img.append(cv2.imread('Archive/origin/basler_' + str(i+1) + '1.jpg', cv2.IMREAD_GRAYSCALE))
    cv2.imwrite('Archive/gray/gray_' + str(i+1) + '.jpg', img[i])

gains = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
num_images = len(img)
height, width = img[0].shape
black_img = np.zeros((height, width), dtype=np.uint8)
cv2.imwrite('Archive/black_img.jpg', black_img)

mean_value = []
for i in range(0, 12):
    mean_value.append(np.mean(img[i]))
mean_value = np.array(mean_value)

standard_deviation = []
for i in range(0, 12):
    standard_deviation.append(np.std(img[i]))
standard_deviation = np.array(standard_deviation)

# noise model
def gaussian_noise(image, mean, std):
    noise = np.random.normal(mean, std, image.shape)
    out = image + noise
    out = np.clip(out, 0, 255)
    return out

def poisson_noise(image, mean):
    noise = np.random.poisson(mean, image.shape)
    out = image + noise
    out = np.clip(out, 0, 255)
    return out

def uniform_noise(image, mean, std):
    low = mean - pow(3, 0.5)*std
    high = mean + pow(3, 0.5)*std
    noise = np.random.uniform(low, high, image.shape)
    out = image + noise
    out = np.clip(out, 0, 255)
    return out

def gamma_noise(image, mean, std):
    def gamma_params_estimator(mean, std):
        def objective_func(params):
            k, theta = params
            return (mean - k * theta) ** 2 + (std - np.sqrt(k * theta ** 2)) ** 2
        initial_guess = [1, 1]
        result = minimize(objective_func, initial_guess, method='Nelder-Mead')
        k, theta = result.x
        return k, theta
    k, theta = gamma_params_estimator(mean, std)
    noise = np.random.gamma(k, theta, image.shape)
    out = image + noise
    out = np.clip(out, 0, 255)
    return out

# add noise on black images
gaussian_img = []
for i in range(0, 12):
    gaussian_img.append(gaussian_noise(black_img, mean_value[i], standard_deviation[i]))
    cv2.imwrite('Archive/gaussian/gaussian_' + str(i+1) + '.jpg', gaussian_img[i])

poisson_img = []
for i in range(0, 12):
    poisson_img.append(poisson_noise(black_img, mean_value[i]))
    cv2.imwrite('Archive/poisson/poisson_' + str(i+1) + '.jpg', poisson_img[i])

uniform_img = []
for i in range(0, 12):
    uniform_img.append(uniform_noise(black_img, mean_value[i], standard_deviation[i]))
    cv2.imwrite('Archive/uniform/uniform_' + str(i+1) + '.jpg', uniform_img[i])

gamma_img = []
for i in range(0, 12):
    gamma_img.append(gamma_noise(black_img, mean_value[i], standard_deviation[i]))
    cv2.imwrite('Archive/gamma/gamma_' + str(i+1) + '.jpg', gamma_img[i])

# calculate mse from noise images and original images
def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0]*imageA.shape[1])
    return err

gaussian_mse = []
for i in range(0, 12):
    gaussian_mse.append(mse(img[i], gaussian_img[i]))
gaussian_mse = np.array(gaussian_mse)

poisson_mse = []
for i in range(0, 12):
    poisson_mse.append(mse(img[i], poisson_img[i]))
poisson_mse = np.array(poisson_mse)

uniform_mse = []
for i in range(0, 12):
    uniform_mse.append(mse(img[i], uniform_img[i]))
uniform_mes = np.array(uniform_mse)

gamma_mse = []
for i in range(0, 12):
    gamma_mse.append(mse(img[i], gamma_img[i]))
gamma_mse = np.array(gamma_mse)

# plot mse
plt.figure(1)
plt.title("Gaussian noise")
plt.xlabel("Gain")
plt.ylabel("MSE")
plt.plot(gains, gaussian_mse, c="b")
plt.show()

plt.figure(2)
plt.title("Poisson noise")
plt.xlabel("Gain")
plt.ylabel("MSE")
plt.plot(gains, poisson_mse, c="b")
plt.show()

plt.figure(3)
plt.title("Uniform noise")
plt.xlabel("Gain")
plt.ylabel("MSE")
plt.plot(gains, uniform_mse, c="b")
plt.show()

plt.figure(4)
plt.title("Gamma noise")
plt.xlabel("Gain")
plt.ylabel("MSE")
plt.plot(gains, gamma_mse, c="b")
plt.show()

print('Gaussian mse', np.mean(gaussian_mse))
print('Poisson mse', np.mean(poisson_mse))
print('Uniform mse', np.mean(uniform_mse))
print('Gamma mse', np.mean(gamma_mse))

poisson_mean = []
for i in range(0, 12):
    poisson_mean.append(np.mean(poisson_img[i]))

poisson_mean = np.array(poisson_mean)

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

x = np.array(gains)
y = poisson_mean.T
order = 2
W, y_hat = Value_LinearRegression(x, y, order)
print(W)

# draw the curve
plt.figure()
plt.suptitle("Noise value")
plt.scatter(x, y, s=10, c="orange")
plt.scatter(x, y_hat, s=10, c="k")

x2 = []
y_hat2 = []
b = 2

for i in range(50):
    x2.append(b)
    b += 22/49
    y_hat2.append(W[0] + W[1]*b + W[2]*b*b)


plt.plot(x2, y_hat2, c="g")
plt.show()