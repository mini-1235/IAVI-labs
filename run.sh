rm -f *.txt
for file in $(ls img/); do
  # echo $file
  python3 calc.py $file
done
python3 sort.py