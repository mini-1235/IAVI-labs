import numpy as np

# Define the exposure times
exposure_times = [10,100, 1000, 10000, 100000]

# Define the gains
gains = [0, 4, 8, 12, 16, 20, 24]

# Loop over the exposure times
for exposure_time in exposure_times:
    print(f'Exposure time: {exposure_time}')
    print('Gain\tRead noise\tShot noise\tDark noise\tTotal noise')
    # Loop over the gains
    for gain in gains:
        # Read the data from the file
        filename = f'result+Expo_{exposure_time}.txt'
        with open(filename, 'r') as f:
            lines = f.readlines()
        data = [line.split() for line in lines if f'Gain_{gain}.bmp' in line][0][1:]

        # Compute the total noise'
        #read noise = standard deviation of the mean
        read_noise = float(data[1])
        #shot noise = standard deviation of the mean
        shot_noise = float(data[2])
        dark_noise = float(data[3])
        total_noise = np.sqrt(read_noise**2 + shot_noise**2 + dark_noise**2)

        # Print the results
        print(f'{gain}\t{read_noise:.2f}\t\t{shot_noise:.2f}\t\t{dark_noise:.2f}\t\t{total_noise:.2f}')
    print()