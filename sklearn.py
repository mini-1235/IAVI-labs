import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Given data
data = '''
result+Expo_1000000.txt
Gain_0.bmp 249.78714718126048
Gain_4.bmp 253.36180267129842
Gain_8.bmp 254.47718678290488
Gain_12.bmp 254.66334520641755
Gain_20.bmp 254.66666666666666
Gain_24.bmp 254.66666666666666
Gain_16.bmp 254.66666666666666

result+Expo_100000.txt
Gain_0.bmp 184.27959777049568
Gain_4.bmp 201.44926227846787
Gain_8.bmp 215.42331527629597
Gain_12.bmp 230.2273939069671
Gain_16.bmp 241.43236430231673
Gain_20.bmp 248.85570928116903
Gain_24.bmp 252.35334081652525

result+Expo_1000.txt
Gain_0.bmp 18.923448177043642
Gain_4.bmp 24.99734396962692
Gain_8.bmp 31.59241199575776
Gain_12.bmp 41.305327262633575
Gain_16.bmp 51.66912232055157
Gain_20.bmp 65.14739996126099
Gain_24.bmp 80.93947452539416

result+Expo_10000.txt
Gain_0.bmp 67.69676342687852
Gain_4.bmp 86.57251075378076
Gain_8.bmp 106.37267433614456
Gain_12.bmp 131.55570274528358
Gain_16.bmp 156.81161778776948
Gain_20.bmp 179.58848272462276
Gain_24.bmp 196.22343447020694

result+Expo_100.txt
Gain_0.bmp 4.259268851398838
Gain_4.bmp 5.226738995368254
Gain_8.bmp 6.592922099125302
Gain_12.bmp 9.855741762138225
Gain_16.bmp 13.770572625594845
Gain_20.bmp 19.07735035203814
Gain_24.bmp 25.25906556419245

result+Expo_10.txt
Gain_0.bmp 3.3011939766126437
Gain_4.bmp 4.074258706222798
Gain_8.bmp 5.01439555231672
Gain_12.bmp 7.4103375083617
Gain_16.bmp 10.583851044259852
Gain_20.bmp 14.991095980668597
Gain_24.bmp 20.288636741308068
'''

# Split the data into lines
lines = data.strip().split('\n')

# Initialize lists to store exposure, gain, and pixel values
exposure_gain = []
pixel_values = []

# Parse the data
current_exposure_gain = None
for line in lines:
    if line.startswith('result+Expo_'):
        # Extract the exposure time from the line
        exposure = int(line.split('_')[1].split('.')[0])
        current_exposure_gain = (exposure, 0)  # Initialize gain as 0 dB
    elif line.startswith('Gain_'):
        gain = int(line.split('_')[1].split('.')[0])
        current_exposure_gain = (current_exposure_gain[0], gain)
        pixel_value = float(line.split()[-1])
        exposure_gain.append(current_exposure_gain)
        pixel_values.append(pixel_value)

# Convert lists to NumPy arrays
exposure_gain = np.array(exposure_gain)
pixel_values = np.array(pixel_values)

# Create a linear regression model
model = LinearRegression()

# Fit the model
model.fit(exposure_gain, pixel_values)

# Print the coefficients and intercept of the model
print("Coefficient (exposure):", model.coef_[0])
print("Coefficient (gain):", model.coef_[1])
print("Intercept:", model.intercept_)





