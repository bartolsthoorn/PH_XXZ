import pickle
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import persim
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Calculate distance matrix.')
parser.add_argument('--d', type=int, default=1, help='dimension of homology group')
opt = parser.parse_args()
print(opt)

filename = 'snapshots_6_100000_32_global_barcodes_32.p'
barcodes = pickle.load(open(filename, 'rb' ))
#barcodes = [[np.array(b_dim) for b_dim in b] for b in barcodes]
Hdim = opt.d
SW_M = 40 # 30
m = len(barcodes)
distance_matrix = np.zeros((m,m))
i_j = []
for i in np.arange(m):
    for j in np.arange(i, m):
        if i == j:
            continue
        i_j.append((i,j))
print(len(i_j), 'i_j')

filtered_barcodes = []
for i, barcode in tqdm(enumerate(barcodes), total=len(barcodes)):
    barcode_selection = []
    for barcode_H in barcode:
        barcode_H = np.array(barcode_H)
        lifetimes = barcode_H[:,1] - barcode_H[:,0]
        barcode_selection.append(barcode_H)
        #barcode_selection.append(barcode_H[lifetimes > 1e-3])
    filtered_barcodes.append(barcode_selection)

def calculate_d(x):
    i, j = x
    #return persim.sliced_wasserstein(np.array(barcodes[i][1]), np.array(barcodes[j][1]), M=10)
    #return persim.sliced_wasserstein(barcodes[i][Hdim], barcodes[j][Hdim], M=20)
    return persim.sliced_wasserstein(filtered_barcodes[i][Hdim], filtered_barcodes[j][Hdim], M=SW_M)

ncpu = os.cpu_count()
print(ncpu, '# of CPUs')
with Pool(ncpu) as p:
    results = p.imap(calculate_d, i_j)

    for k, d in tqdm(enumerate(results), total=len(i_j)):
        i, j = i_j[k]
        distance_matrix[i,j] = d
        distance_matrix[j,i] = d

np.save(filename.split('.')[0] + ('_distance_matrix_%d_%d' % (Hdim, SW_M)), distance_matrix)
