import os 

#loop through every txt file in the directory
output_list = []
output_str = ""
for file in os.listdir("./"):
  if file.endswith(".txt"):
    print(file)
    #open the file
    input_file = open(file, "r")
    #read the file
    input_str = input_file.read()
    #sort the result by the float value in the line
    lines = input_str.split("\n")
    # print(lines)
    sorted_lines = sorted(lines, key=lambda x: float(x.split()[1]) if len(x.split()) >= 2 else 0)
    print(sorted_lines)
    #write the sorted file to a new file
    output_str = file + "\n" + "\n".join(sorted_lines) + "\n"
    output_list.append(output_str)
    print("output_str")
    print(output_str)
#final.txt
final_output = '\n'.join(output_list)
file = open("final.txt", "w")
# file.write(output_str)
file.write(final_output)