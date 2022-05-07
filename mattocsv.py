import scipy.io
import numpy as np

data = scipy.io.loadmat("./livedataset/databaserelease2/dmos.mat")

for i in data:
    if '__' not in i and 'readme' not in i:
        print(data[i])
        np.savetxt(("mat_dmos.csv"), data[i], delimiter=',')
