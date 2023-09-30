import os
from PIL import Image
import numpy as np
import sys
# folder_name = input("Enter the folder name: ")
folder_name = sys.argv[1]
dir = "./img/" + folder_name + "/";
output_file = "result+" + folder_name + ".txt"
output = open(output_file, "w")
# read all files in the directory
for file in os.listdir(dir):
    # get the file extension
    print(file)
    img = Image.open(dir + "/" + file)
    img.load()
    data = np.asarray(img, dtype="int32")
    #compute the sum of the pixels
    data_sum = data.sum()
    data_mean = data_sum / (data.shape[0] * data.shape[1])
    mse = ((data-data_mean)**2).mean()
    output.write(str(file) + " ")
    output.write(str(data.mean())+ " ")
    output.write(str(data.std())+ " ")
    output.write(str(data.var())+ " ")
    output.write(str(mse)+ " ")
    output.write("\n")
output.close()

    
