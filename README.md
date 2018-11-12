# FYS-STK4155_Project_2

NB!!! The file "Ising2DFM_reSample_L40_T=All.pkl" was too big to upload to Github,
but must be put in the main directory to run "energy_models.py". Goto:
https://physics.bu.edu/~pankajm/ML-Review-Datasets/isingMC/
to download it.


There are two main programs in this folder:

1) classification_models.py
2) energy_models.py

The remaining python files contain functions and classes controlled by the two programs above.



* The file "Plotting_code.py" contains many functions that can be called 
to produce certain plots. For each plot, there is a function to generate the required data,
which is then stored in one of the folders "Ising, Ising_cross_entroy, kval". The corresponding
plotting function then leads the file and makes the desired plot.
The whole process (should be) fully automated.

* The file "Methods.py" contains many simple math functions, like "sigmoid",
but also functions to make k-fold cross validation files and plot.
Also some function like "get_partition" which makes a partition of the set {1,...,n}.
Most functions there contain a description to make them readable.

* The file "reg_data.py" contains the two important classes "MLP" amd "reg_data".
The former is to make multilayerperceptrons and the latter to perform linear regression.

When the files "classification_models.py" and "energy_models.py" are run as is, 
they should produce (almost) all plots as shown throughout the report. Some require
some parameters to be changed.

