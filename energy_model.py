import numpy as np
import scipy.sparse as sp
from Methods import *
from regdata import *
import pickle, os
from Plotting_code import *
from timeit import default_timer as timer

np.random.seed(12)

#===================================================================================================
# Generate data
#===================================================================================================

# system size
L=40

# create 10000 random Ising states
states=np.random.choice([-1, 1], size=(10000,L))

# calculate Ising energies
def ising_energies(states,L):
    """
    This function calculates the energies of the states in the nn Ising Hamiltonian
    """
    J=np.zeros((L,L),)
    for i in range(L):
        J[i,(i+1)%L]-=1.0
    # compute energies
    E = np.einsum('...i,ij,...j->...',states,J,states)
    return E

energies = ising_energies(states,L)

#===================================================================================================
# neural network
#===================================================================================================

# shrink data in consideration (optional)
#states = states[0:600]
#energies = energies[0:600]
data_size = len(states)

n = np.size(energies)
min_E = min(energies)
max_E = max(energies)
delta_E = max_E - min_E
scale_energy_to_unit_interval = lambda x: (x-min_E)/delta_E 
energies_scaled = scale_energy_to_unit_interval(energies)

# neural network settings for Ising model
network_size = [L,16*L,16*L,1] # good size = [L,16*L,16*L,1], but slow
t0,t1 = 1,1
k=3

# 3-fold cross validation 
#generate_kval_nn_data("Ising",states,energies_scaled,network_size,
#                      epochs = 101, #(Warning: very slow)
#                      k=k,
#                      t0=t0,t1=t1)

plot_error_kval_nn("Ising model","Ising",data_size,
                   network_size,
                   k=k,
                   t0=t0,t1=t1)

# R2 vs time
train_X , test_X = states[0:6667], states[6667:10000]
train_Y , test_Y = energies_scaled[0:6667], energies_scaled[6667:10000]


#plot_nn_R2_time_batches(network_size,
#                        train_X,train_Y,
#                        test_X,test_Y,
#                        number_batches_list=[1],
#                        time_limit = 601,
#                        t0=t0,t1=t1)

#===================================================================================================
# linear regression
#===================================================================================================

# row in design matrix
def design_row(state):
    """
    This function determines a map from the independent variables {s_i} to a row in the
    design matrix X.
    """
    row = [1] 
    length = np.size(state)
    for j in range(0,length):
        for k in range(0,length):
            row.append(state[j]*state[k])
    return np.array(row)

states = states[0:600]
energies = energies[0:600]
data = regdata(states, energies, design_row)

Nstart = -4
Nstop = 5
k = 3

# 3-fold cross validation
#generate_kval_betas(data, Nstart, Nstop, k, "Ising_J_1_")
plot_error_kval(data, Nstart, Nstop, k, "Ising_J_1_", "Ising model $J=1$ regression performance")

# time and print R2 scores of ols, ridge and lasso for final model
def time_method(data, LAMBDA = None, epsilon = None):
    '''Function to measure time it takes to run either lasso, ridge or lasso.
    Writing:
    time_method(data)
    time_method(data,LAMBDA=1.0)
    time_method(data,LAMBDA=1.0,epsilon = 0.01) #(Warning: slow)

    Results in:
    Avg time (5 iterations):  0.952943677254375 [s]
    Avg time (5 iterations):  0.8987969291276825 [s]
    Avg time (5 iterations):  114.85832889421667 [s]
    '''

    tot_time = 0
    for i in range(5):
        time0 = timer()
        beta = data.get_beta(data.X,data.observations_vector,
                             LAMBDA = LAMBDA, 
                             epsilon = epsilon)

        # time needed to calculate coefficients: beta
        tot_time += timer()-time0

    # average time
    tot_time = tot_time/5

    # times to terminal
    print("Avg time (5 iterations): ", tot_time,"[s]")

# ols, ridge, lasso times
#data = regdata(states[0:400], energies[0:400], design_row)
#time_method(data)
#time_method(data,LAMBDA=1.0)
#time_method(data,LAMBDA=1.0,epsilon = 0.01) #(Warning: slow)


print("success!")
plt.show()



