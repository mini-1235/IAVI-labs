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
    data.sum()
    #compute the average of the pixels
    data.mean()
    #write filename and average to file
    #open the file
    #write the filename and average to the file
    output.write(str(file) + " ")
    output.write(str(data.mean()))
    output.write("\n")
output.close()

    
