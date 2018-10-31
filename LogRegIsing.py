import scipy.sparse as sp
from Methods import *
from regdata import *
import pickle, os
import numpy as np
np.random.seed(50)

print("Loading data from ")

# load data
file_name = "Ising2DFM_reSample_L40_T=All.pkl" # this file contains 16*10000 samples taken in T=np.arange(0.25,4.0001,0.25)
data = pickle.load(open(file_name,'rb')) # pickle reads the file and returns the Python object (1D array, compressed bits)
data = np.unpackbits(data).reshape(-1, 1600) # Decompress array and reshape for convenience
data=data.astype('int')
data[np.where(data==0)]=-1 # map 0 state to -1 (Ising variable can take values +/-1)

file_name = "Ising2DFM_reSample_L40_T=All_labels.pkl" # this file contains 16*10000 samples taken in T=np.arange(0.25,4.0001,0.25)
labels = pickle.load(open(file_name,'rb')) # pickle reads the file and returns the Python object (here just a 1D array with the binary labels)

print("Data loaded.")

# divide data into ordered, critical and disordered
X_ordered=data[0:70000]
Y_ordered=labels[0:70000]
X_critical=data[70000:100000]
Y_critical=labels[70000:100000]
X_disordered=data[100000:]
Y_disordered=labels[100000:]
del data,labels

# shuffle data
np.random.shuffle(X_ordered)
np.random.shuffle(Y_ordered)
np.random.shuffle(X_disordered)
np.random.shuffle(Y_disordered)

print("Divide into training data and testing data.")

# divide into training and testing data, and recover original shape
train_X = np.concatenate((X_ordered[0:4000],X_disordered[0:4000]))
train_Y = np.concatenate((Y_ordered[0:4000],Y_disordered[0:4000]))
test_X = np.concatenate((X_ordered[4000:6000],X_disordered[4000:6000]))
test_Y = np.concatenate((Y_ordered[4000:6000],Y_disordered[4000:6000]))

# row in design matrix
def design_row(state):
    row = []
    input = state.reshape((40,40))
    for k in range(0,40):
        for l in range(0,40):
            if k < 39:
                row.append(input[k,l]*input[k+1,l])
            if l < 39:
                row.append(input[k,l]*input[k,l+1])
    return np.array(row)

print("Initiating regdata object from training data.")

# initialize regression data
data = regdata(train_X,train_Y,design_row)
print("Calculate and save coefficients from training data.")


N = 1000
step = 1
beta = data.beta_SGD(M=5, epsilon = 0.01, max_number_iterations=step )
betas = [beta]
for index in range(0,N-1):
    beta = data.beta_SGD(beta_init=beta, M=5, epsilon = 0.01, max_number_iterations=step )
    betas.append(beta)

iterations = np.linspace(0,N,N)
accuracies = np.zeros(N)
data_test = regdata(test_X,test_Y,design_row)
for index, beta in enumerate(betas):
    model_test = data_test.model(beta,method_name='log')
    accuracies[index] = accuracy(test_Y,model_test)
print(accuracies)


plt.figure("Stochastic gradient descent")
plt.plot(iterations,accuracies)
plt.xlabel("Iterations")
plt.ylabel("Accuracy score")



#save_array_to_file(beta,"beta_logistic.txt")

# load beta from file and find model

#beta = open_array_from_file("beta_logistic.txt")


# test accuracy



plt.show()
print("Success!")
