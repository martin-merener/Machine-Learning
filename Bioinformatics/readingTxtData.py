# Reading a txt file

fname = 'temperatureData.txt'

def file_len(fname): 
    '''
    This function counts the number of lines in fname
    '''
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

nHeader = 6 # Number of lines that should be ignored

# The following script just reads the int values in 3 columns after the header lines
idx = []
col1 = []
col2 = []
with open(fname) as f:
    for i, l in enumerate(f):
        if i+1>nHeader:
            line = l.split()
            idx.append(int(line[0]))
            col1.append(int(line[1]))
            col2.append(int(line[2]))

import matplotlib.pyplot as plt
plt.scatter(col1,col2)
plt.show()
