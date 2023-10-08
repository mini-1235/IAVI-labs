#!/bin/bash

# List all files in the /dataset/lab3/ directory and grep the output
for file in $(ls dataset/lab3/ | grep -E '.*\.jpg$'); do
  # Run the Python script on the current file
  python3 tmp.py "dataset/lab3/$file" >> $file.txt
done