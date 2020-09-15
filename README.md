# Finding hidden order in spin models with persistent homology
Code used in the article "Finding hidden order in spin models with persistent homology"

Run the code in order: run1.jl, run2.py and run3.py.

arXiv: [\[2009.05141\] Finding hidden order in spin models with persistent homology](https://arxiv.org/abs/2009.05141)


## Description

The Julia code of `run1.jl` contains the Monte Carlo simulation of the XXZ model on a pyrochlore lattice. Spin configurations are sampled and stored in HDF5 format.

Next, the Python code `run2.py` used GUDHI to calculate barcodes for the spin configurations in the HDF5 file. Barcodes are stored as a pickle .p file.

Finally, the Python code `run3.py` calculates the pairwise sliced Wasserstein distance for the barcodes loaded from the pickle file. The final distance matrix D is stored as a numpy matrix file.
