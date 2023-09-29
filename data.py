import pandas as pd
import re

# Parse the data from the selection into a pandas DataFrame
data = '''
result+Expo_1000000.txt

Gain_0.bmp 249.78714718126048
Gain_4.bmp 253.36180267129842
Gain_8.bmp 254.47718678290488
Gain_12.bmp 254.66334520641755
Gain_20.bmp 254.66666666666666
Gain_24.bmp 254.66666666666666
Gain_16.bmp 254.66666666666666
'''

df = pd.DataFrame([re.split('\s+', i) for i in data.split('\n') if i], columns=[ 'data', 'exposure'])
df['exposure'] = df['exposure'].str.extract(r'(\d+)').astype(int)
df = df[['filename', 'exposure', 'data']]