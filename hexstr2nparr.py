import sys
import csv
import numpy as np
# from collections import defaultdict

hex2dec = {str(i): i / 15 for i in range(10)}
hex2dec.update({chr(ord('a') + i): (10 + i) / 15 for i in range(6)})



output_arr = np.zeros((0,15))
try:
    with open('./data/raw/{0}'.format(sys.argv[1]), 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        i = 0
        for line in csv_reader:
            if i > 0:
                row = np.asarray([hex2dec[c] for c in line[0]])
                output_arr = np.vstack((output_arr, row))
            i += 1
except IOError:
    print("IOError")
    print('Ensure that raw hex value csv data is placed in ./data/raw/ and provide file name only, not full URI')
    print('Example: Ensure sample.csv is copied over to ./data/raw/sample.csv and run\n\t \'python hexstr2nparr.py sample.csv\'')

np.savetxt('./data/converted/{0}'.format(sys.argv[1]), output_arr, delimiter=',')
print('Converted file saved at ./data/converted/{0}'.format(sys.argv[1]))