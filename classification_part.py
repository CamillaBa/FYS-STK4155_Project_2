from Methods import *
from regdata import *
import pickle
import numpy as np
np.random.seed(100)
from Plotting_code import *


#==========================================================================================================
# Loading data and dividing data into test/ training data
#==========================================================================================================

print("Loading data used in: https://physics.bu.edu/~pankajm/ML-Review-Datasets/isingMC/")

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





#==========================================================================================================
# Logistic regression
#==========================================================================================================

print("Divide into training data and testing data.")

# divide into training and testing data, and recover original shape
n = 5000
train_X = np.concatenate((X_ordered[0:n],X_disordered[0:n]))
train_Y = np.concatenate((Y_ordered[0:n],Y_disordered[0:n]))

m = n + 5000
test_X = np.concatenate((X_ordered[n:m],X_disordered[n:m]))
test_Y = np.concatenate((Y_ordered[n:m],Y_disordered[n:m]))


print("Data handling successful!")

# row in design matrix
def design_row(state):
    row = [1]
    input = state.reshape((40,40))
    for k in range(0,40):
        for l in range(0,40):
            if k < 39:
                row.append(input[k,l]*input[k+1,l])
            if l < 39:
                row.append(input[k,l]*input[k,l+1])
    return np.array(row)


# make time vs accuracy plot
#plot_acc_time_SGD(train_X,train_Y,test_X,test_Y,design_row,
#                  t0=0.000001,t1=1,
#                  number_batches_list = [1,2,4,8,16,32,64],
#                  time_limit=3)

# generate SGD data and make plot
#generate_logreg_SGD_betas("Ising",train_X,train_Y,design_row,
#                          number_epochs= 25,
#                          t0=0.000001,t1=1,
#                          number_batches_list = [1,2,4,8,16,32,64])

#generate_logreg_SGD_plot("Ising",train_X,train_Y,test_X,test_Y,design_row,
#                         t0=0.000001,t1=1,
#                         number_batches_list = [1,2,4,8,16,32,64])
                         

#==========================================================================================================
# Neural network with cross entropy cost function and sigmoid activation function
#==========================================================================================================


print("Divide into training data and testing data.")

# divide into training and testing data, and recover original shape
n = 500
train_X = np.concatenate((X_ordered[0:n],X_disordered[0:n]))
train_Y = np.concatenate((Y_ordered[0:n],Y_disordered[0:n]))

m = n + 5000
test_X = np.concatenate((X_ordered[n:m],X_disordered[n:m]))
test_Y = np.concatenate((Y_ordered[n:m],Y_disordered[n:m]))

print("Data handling successful!")


# the derivative of cross entropy cost function, to send as argument for the neural network class
def DC_cross(a, target):
    '''Derivative of cross entropy cost function.
    '''
    return (a-target)/(a*(1-a))

# after trail and error, the neural network size that gives us the best results.
network_size = [1600,40,40,1] # (two hidden layers of 40 nodes each)
t0, t1= 0.1, 1

# generate accuracy plotting data for SGD of neural network with cross entropy cost function
#generate_nn_SGD_accuracies_vs_epochs_file("Ising_cross_entropy",
#                                          network_size,
#                                          train_X,train_Y,
#                                          test_X,test_Y,
#                                          DC = DC_cross,
#                                          number_epochs = 25,
#                                          t0=t0,t1=t1,
#                                          number_batches_list = [1,2,4,8,16,32,64])

# generate plot of the above generated data.
#plot_nn_SGD_accuracies_vs_epochs("Ising_cross_entropy",network_size,
#                                 train_X,train_Y,
#                                 test_X,test_Y,
#                                 t0=t0,t1=t1,
#                                 number_batches_list = [1,2,4,8,16,32,64])


#plot_nn_accuracies_time_batches(network_size,
#                                train_X,train_Y,
#                                test_X,test_Y,
#                                DC = DC_cross,
#                                time_limit = 3,
#                                t0=t0,t1=t1,
#                                number_batches_list=[1,2,4,8,16,32,64])

#==========================================================================================================
# Neural network with cross entropy cost function and tangens hyperbolicus activation function
#==========================================================================================================

print("Divide into training data and testing data.")

# divide into training and testing data, and recover original shape
n = 500
train_X = np.concatenate((X_ordered[0:n],X_disordered[0:n]))
train_Y = np.concatenate((Y_ordered[0:n],Y_disordered[0:n]))

m = n + 5000
test_X = np.concatenate((X_ordered[n:m],X_disordered[n:m]))
test_Y = np.concatenate((Y_ordered[n:m],Y_disordered[n:m]))

print("Data handling successful!")

def Dtanh(z):
    '''Derivative with respect to z of tanh(z), expressed in a way
    that is not prone to numerical round off errors.
    Dtanh(z) = 1-tanh^2(z). 
    For z close to 0 we may see numerical loss of precision.'''
    n = len(z)
    output = np.zeros(n)
    for i in range(n):
        if abs(z[i]) < 15:
            output[i]=4*np.exp(2*z[i])/((np.exp(2*z[i])+1)*(np.exp(2*z[i])+1))
        else:
            output[i]=0
    return output

network_size = [1600,40,40,1] # (two hidden layers of 40 nodes each)
t0, t1= 0.1, 1

# generate accuracy plotting data for SGD of neural network with cross entropy cost function
#generate_nn_SGD_accuracies_vs_epochs_file("Ising_cross_entropy/tanh/",
#                                          network_size,
#                                          train_X,train_Y,
#                                          test_X,test_Y,
#                                          DC = DC_cross,
#                                          A_DA = [np.tanh,Dtanh],
#                                          number_epochs = 25,
#                                          t0=t0,t1=t1,
#                                          number_batches_list=[1,2,4,8,16,32,64])

# generate plot of the above generated data.
#plot_nn_SGD_accuracies_vs_epochs("Ising_cross_entropy/tanh/",network_size,
#                                 train_X,train_Y,
#                                 test_X,test_Y,
#                                 t0=t0,t1=t1,
#                                 plot_title = 'Tanh',
#                                 number_batches_list=[1,2,4,8,16,32,64])

#plot_nn_accuracies_time_batches(network_size,
#                                train_X,train_Y,
#                                test_X,test_Y,
#                                DC = DC_cross,
#                                A_DA = [np.tanh,Dtanh],
#                                time_limit = 5,
#                                t0=t0,t1=t1,
#                                number_batches_list=[1,2,4,8,16,32,64])


#==========================================================================================================
# Neural network with cross entropy cost function and clipped ReLu activation function
#==========================================================================================================

print("Divide into training data and testing data.")

# divide into training and testing data, and recover original shape
n = 500
train_X = np.concatenate((X_ordered[0:n],X_disordered[0:n]))
train_Y = np.concatenate((Y_ordered[0:n],Y_disordered[0:n]))

m = n + 5000
test_X = np.concatenate((X_ordered[n:m],X_disordered[n:m]))
test_Y = np.concatenate((Y_ordered[n:m],Y_disordered[n:m]))

print("Data handling successful!")

def clipped_ReLu(z):
    '''Composition clip with rectified linear unit function.'''
    output = np.copy(z)
    return np.clip(output,0.0001,0.9999)

def Dclipped_ReLu(z):
    '''Derivative of clipped rectified linear unit function.'''
    n = len(z)
    output = np.ones(n)
    for i in range(n):
        if z[i]>0.9999 or z[i]<0.0001:
            output[i]=0
    return output

network_size = [1600,40,40,1] # (two hidden layers of 40 nodes each)
t0, t1= 0.01, 1 #good settings 0.01, 1

# generate accuracy plotting data for SGD of neural network with cross entropy cost function
#generate_nn_SGD_accuracies_vs_epochs_file("Ising_cross_entropy/clipped_relu/",
#                                          network_size,
#                                          train_X,train_Y,
#                                          test_X,test_Y,
#                                          DC = DC_cross,
#                                          A_DA = [clipped_ReLu, Dclipped_ReLu],
#                                          number_epochs = 25,
#                                          number_batches_list=[1,2,4,8,16,32,64],
#                                          t0=t0,t1=t1)

# generate plot of the above generated data.
#plot_nn_SGD_accuracies_vs_epochs("Ising_cross_entropy/clipped_relu/",network_size,
#                                 train_X,train_Y,
#                                 test_X,test_Y,
#                                 t0=t0,t1=t1,
#                                 number_batches_list=[1,2,4,8,16,32,64],
#                                 plot_title = 'clipped_relu')

plot_nn_accuracies_time_batches(network_size,
                                train_X,train_Y,
                                test_X,test_Y,
                                DC = DC_cross,
                                A_DA = [clipped_ReLu, Dclipped_ReLu],
                                number_batches_list = [1,2,4,8,16,32,64,128],
                                time_limit = 30,
                                t0=t0,t1=t1)


#==========================================================================================================
# Neural network with cross entropy cost function and activation function = bump
#==========================================================================================================

print("Divide into training data and testing data.")

# divide into training and testing data, and recover original shape
n = 500
train_X = np.concatenate((X_ordered[0:n],X_disordered[0:n]))
train_Y = np.concatenate((Y_ordered[0:n],Y_disordered[0:n]))

m = n + 500
test_X = np.concatenate((X_ordered[n:m],X_disordered[n:m]))
test_Y = np.concatenate((Y_ordered[n:m],Y_disordered[n:m]))

print("Data handling successful!")

def bump(z):
    n = len(z)
    output = np.zeros(n)
    for i in range(n):
        if z[i] < 0.001:
            output[i] = 0.001
        elif z[i] > 0.9999:
            output[i] = 0.9999
        else:
            output[i] = np.sin(np.pi/2*z[i])
    return output

def Dbump(z):
    n = len(z)
    output = np.zeros(n)
    for i in range(n):
        if z[i] < 0.001:
            output[i] = 0.001
        elif z[i] > 0.9999:
            output[i] = 0.9999
        else:
            output[i] = np.cos(np.pi/2*z[i])*np.pi/2
    return output

network_size = [1600,40,40,1] # (two hidden layers of 40 nodes each)
t0, t1= 0.001, 1 #settings for good sin bump

## generate accuracy plotting data for SGD of neural network with cross entropy cost function
#generate_nn_SGD_accuracies_vs_epochs_file("Ising_cross_entropy/bump/",
#                                          network_size,
#                                          train_X,train_Y,
#                                          test_X,test_Y,
#                                          DC = DC_cross,
#                                          A_DA = [bump, Dbump],
#                                          number_batches_list=[1,2,4,8,16,32,64],
#                                          number_epochs = 25,
#                                          t0=t0,t1=t1)

## generate plot of the above generated data.
#plot_nn_SGD_accuracies_vs_epochs("Ising_cross_entropy/bump/",network_size,
#                                 train_X,train_Y,
#                                 test_X,test_Y,
#                                 t0=t0,t1=t1,
#                                 number_batches_list=[1,2,4,8,16,32,64],
#                                 plot_title = 'bump')

#plot_nn_accuracies_time_batches(network_size,
#                                train_X,train_Y,
#                                test_X,test_Y,
#                                name = 'Bump',
#                                DC = DC_cross,
#                                A_DA = [bump, Dbump],
#                                time_limit = 3,
#                                t0=t0,t1=t1)


#==========================================================================================================
# Neural network with cross entropy cost function and activation function = hill
#==========================================================================================================

print("Divide into training data and testing data.")

# divide into training and testing data, and recover original shape
n = 500
train_X = np.concatenate((X_ordered[0:n],X_disordered[0:n]))
train_Y = np.concatenate((Y_ordered[0:n],Y_disordered[0:n]))

m = n + 500
test_X = np.concatenate((X_ordered[n:m],X_disordered[n:m]))
test_Y = np.concatenate((Y_ordered[n:m],Y_disordered[n:m]))

print("Data handling successful!")

def hill(z):
    n = len(z)
    output = np.zeros(n)
    for i in range(n):
        if z[i] < 0.001:
            output[i] = 0.001
        elif z[i] > 0.9999:
            output[i] = 0.9999
        else:
            output[i] = 1+np.sin(3*np.pi/2*z[i])
    return output

def Dhill(z):
    n = len(z)
    output = np.zeros(n)
    for i in range(n):
        if z[i] < 0.001:
            output[i] = 0.001
        elif z[i] > 0.9999:
            output[i] = 0.9999
        else:
            output[i] = np.cos(3*np.pi/2*z[i])*3*np.pi/2
    return output

network_size = [1600,40,40,1] # (two hidden layers of 40 nodes each)
t0, t1= 0.00001, 0.1
## generate accuracy plotting data for SGD of neural network with cross entropy cost function
#generate_nn_SGD_accuracies_vs_epochs_file("Ising_cross_entropy/hill/",
#                                          network_size,
#                                          train_X,train_Y,
#                                          test_X,test_Y,
#                                          DC = DC_cross,
#                                          A_DA = [hill, Dhill],
#                                          number_epochs = 25,
#                                          number_batches_list=[1,2,4,8,16,32,64],
#                                          t0=t0,t1=t1)

## generate plot of the above generated data.
#plot_nn_SGD_accuracies_vs_epochs("Ising_cross_entropy/hill/",network_size,
#                                 train_X,train_Y,
#                                 test_X,test_Y,
#                                 t0=t0,t1=t1,
#                                 plot_title = 'hill',
#                                 number_batches_list=[1,2,4,8,16,32,64])

#plot_nn_accuracies_time_batches(network_size,
#                                train_X,train_Y,
#                                test_X,test_Y,
#                                name = 'Hill',
#                                DC = DC_cross,
#                                A_DA = [hill, Dhill],
#                                time_limit = 3,
#                                t0=t0,t1=t1,
#                                number_batches_list=[1,2,4,8,16,32,64])


plt.show()
print("Success!")
