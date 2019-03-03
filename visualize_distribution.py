import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys


# fig = plt.figure()


csv_data = genfromtxt('./data/annotated/{0}'.format(sys.argv[1]), delimiter=',')
plt.title('{0}'.format(sys.argv[1]))

# data = csv_data[1:, 2:17]
data = csv_data

# data is a n x 15 array
rows, cols = data.shape

numNeg = 0
numPos = 0 
for x in data : 
    curr = x[15]
    if(curr == 0): 
        numNeg = numNeg + 1
    else: 
        numPos = numPos + 1

print(numNeg)
print(numPos)

labels = 'Positive', 'Negative'
sizes = [numPos, numNeg]
colors = ['lightcoral', 'lightskyblue']


plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
 

plt.axis('equal')
plt.show()

