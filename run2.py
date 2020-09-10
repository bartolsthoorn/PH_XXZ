import h5py
import numpy as np
from ripser import ripser
import gudhi as gd
from persim import plot_diagrams
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from tqdm import tqdm
from multiprocessing import Pool
import gudhi as gd
import pickle
import persim

#filename = 'snapshots_4_100000_32.hdf5'
filename = 'snapshots_6_100000_32.hdf5'
f = h5py.File(filename, 'r')
L = 6
N = L*L*L*16

spins_type = 'global'

coordinates = f['coordinates'][:].transpose()
coordinates = coordinates.reshape(N,3)
coordinates_D = squareform(pdist(coordinates, metric='euclidean'))
print(np.min(coordinates_D[coordinates_D > 0]))
spins = f['spins_' + spins_type][:].transpose()
print(spins.shape)

J_grid = f['J'][:]
T_grid = f['T'][:]
print(J_grid)

N_T = len(T_grid)
N_J = len(J_grid)

f.close()

params = []
for i, J in enumerate(J_grid):
    for j, T in enumerate(T_grid):
        params.append((J,T))

n_samples = 32
def calculate_barcodes(samples):
    points_list = [coordinates + (spins * 0.3535533905932738 * 0.25) for spins in samples]
    points_list = points_list[0:n_samples]
    dgms = [[] for i in range(3)]
    for points in points_list:
        #rips = ripser(points, thresh=max_death, maxdim=maxdim)
        alpha_complex = gd.AlphaComplex(points=points)
        st_alpha = alpha_complex.create_simplex_tree()
        barcodes = st_alpha.persistence()
        #for i, d in enumerate(rips['dgms']):
        #    dgms[i] += list(d)
        for d in np.arange(3):
            dgms[d] += [bar for dim, bar in barcodes if dim == d]
    return dgms

barcodes = []
for i, J in tqdm(enumerate(J_grid), total=len(J_grid)):
    for j, T in tqdm(enumerate(T_grid), total=len(T_grid)):
        barcodes.append(calculate_barcodes(spins[i, j, :]))

pickle.dump(barcodes, open(filename.split('.')[0] + '_' + spins_type + ('_barcodes_%d.p' % n_samples), 'wb' ))
