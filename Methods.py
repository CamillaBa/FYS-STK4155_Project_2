from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
import numpy as np
from math import exp
from scipy import linalg
import matplotlib.pyplot as plt
import time

#from pathlib import Path
import pickle
import os

#================================================================================================================

def get_partition(N,k):
    ''' Creates a random partition of k (almost) equally sized parts of the array
    {1,2,...,N}. This can be used divide indexed objects into training/testing data.
    '''
    indices = np.arange(N)
    indices_shuffle = np.arange(N)
    np.random.shuffle(indices_shuffle)
    partition = []
    for step in range(0,k):
        part = list(indices_shuffle[step:N:k])
        partition.append(part) 
    return partition

#================================================================================================================

def accuracy(t,y):
    # t represents the target values and y the output of the model
    n = len(t)
    counter = 0
    for itemx, itemy in zip(t,y):
        counter += int(itemx==itemy)
    return float(counter)/n

#================================================================================================================

def save_array_to_file(array, filename):
    '''Simple function to write array to file of given filename.'''
    
    #construct string to write
    outstring = ""
    for item in array:
        outstring += str(item)
        outstring += ","
    outstring = outstring[:-1]

    #write to file
    file = open(filename,'w')
    file.write(outstring)
    file.close

def open_array_from_file(filename):
    '''Simple function to read an array from a file.'''

    file = open(filename,"r")
    line = file.readline()
    words = line.split(',')
    for index, word in enumerate(words):
        words[index] = float(word)
    file.close
    return np.array(words)

#================================================================================================================

# Classifyer for logistic regression
def classifyer(x):
    for i in range(0,len(x)):
        if x[i] <= 0.5:
            x[i]=0.0
        else:
            x[i]=1.0
    return x

# Classifyer for logistic regression
def classifyer2(x):
    ''' Serves as a stricter classifyer than the function "classifyer".
    '''
    for i in range(0,len(x)):
        if x[i] <= 0.1:
            x[i]=0.0
        elif x[i] >= 0.9:
            x[i]=1.0
    return x

#================================================================================================================

# Sigmoid function
def sigmoid(x):
    if x >= 0:
        emx= np.exp(-x)
        output = 1 /(1+emx)
    else:
        ex= np.exp(x)
        output = ex/(1+ex)
    return output

def sigmoid_vec(x):
    n = len(x)
    output = np.zeros(n)
    for i in range(0,n):
        output[i]=sigmoid(x[i])
    return output

#================================================================================================================

# Variance
def var(f_model):
    n = np.size(f_model)
    f_model_mean = np.sum(f_model)/n
    #f_model_mean = np.mean(f_model)
    return np.sum((f_model-f_model_mean)**2)/n

#================================================================================================================

# Bias
def bias(f_true,f_model):
    n = np.size(f_model)
    #f_model_mean = np.sum(f_model)/n
    f_model_mean = np.mean(f_model)
    return np.sum((f_true-f_model_mean)**2)/n

#================================================================================================================

# MSE
def MSE(f_true,f_model):
    n = np.size(f_model)
    return np.sum((f_true-f_model)**2)/n

#================================================================================================================

# Extra term
def extra_term(f_true,f_model):
    n = np.size(f_model)
    f_model_mean = np.mean(f_model)
    return 2.0/n*np.sum((f_model_mean-f_true)*(f_model-f_model_mean))

#================================================================================================================

# SVD invert
def SVDinv(A):
    ''' Takes as input a numpy matrix A and returns inv(A) based on singular value decomposition (SVD).
    SVD is numerically more stable (at least in our case) than the inversion algorithms provided by
    numpy and scipy.linalg at the cost of being slower.
    '''
    U, s, VT = linalg.svd(A)
    D = np.zeros((len(U),len(VT)))
    for i in range(0,len(VT)):
        D[i,i]=s[i]
    UT = np.transpose(U); V = np.transpose(VT); invD = np.linalg.inv(D)
    return np.matmul(V,np.matmul(invD,UT))

#================================================================================================================

# R2 score
def R2(x_true,x_predict):
    n = np.size(x_true)
    x_avg = np.sum(x_true)/n
    enumerator = np.sum ((x_true-x_predict)**2)
    denominator = np.sum((x_true-x_avg)**2)
    return 1.0 - enumerator/denominator

#================================================================================================================

def print_list_of_lists(list_of_lists,filename):
    file = open(filename,"w")
    N = len(list_of_lists)
    for index, lst in enumerate(list_of_lists):
        outstring = ""
        for item in lst:
            outstring += str(item)
            outstring += ","
        outstring = outstring[:-1]
        if index < N:
            outstring += "\n"
        file.write(outstring)

def read_list_of_lists(filename):
    list_of_lists = []
    file = open(filename,"r")
    for line in file.readlines():
        lst = line[:-1].split(',')
        for index, item in enumerate(lst):
            lst[index] = int(item)
        list_of_lists.append(lst)
    return list_of_lists

#================================================================================================================

# Two basic functions to make storing objects to file quick and easy, using pickle

def save_object_to_file(obj, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    file = open(filename,"wb")
    pickle.dump(obj, file)
    file.close()

def load_object_from_file(filename):
    file = open(filename,"rb")
    obj = pickle.load(file)
    file.close()
    return obj

#================================================================================================================

def generate_kval_betas(data, Nstart, Nstop, k, dataname, epsilon = 0.01):
    ''' Function that saves models (given by beta) resulting from ols, ridge and lasso.
    The files are of the form "/kval/beta_dataname_kval_ridge_lambda_integer.pickle". Each
    contains a list of k betas for the current bias.
    '''
    N = Nstop-Nstart # number of lambdas
    lambdas = [float(10**index) for index in range(Nstart,Nstop+1)] #interval of lambdas
    X = data.X
    partition = get_partition(data.number_of_observations,k)
    save_object_to_file(partition,"./kval/partition.pickle")
    observations = data.observations_vector

    # loop through partition and calculate betas and append to list
    def print_betas(filename, LAMBDA = None, epsilon = None):
        betas=[]
        for i, test_data in enumerate(partition):
            train_data = [x for j,x in enumerate(partition) if j!=i]
            train_data = sum(train_data, [])
            # find model for training data
            beta = data.get_beta(X[train_data],
                                observations[train_data],
                                LAMBDA=LAMBDA,
                                epsilon=epsilon)
            betas.append(beta)
        save_object_to_file(betas,filename)

    print_betas("./kval/beta_{}_{}val_ols.pickle".format(dataname,k))

    for index, LAMBDA in enumerate(lambdas):
        print_betas("./kval/beta_{}_{}val_ridge_lambda_{}.pickle".format(dataname,k,index),LAMBDA=LAMBDA)
        print_betas("./kval/beta_{}_{}val_lasso_lambda_{}.pickle".format(dataname,k,index),LAMBDA=LAMBDA,epsilon=epsilon)
        print("Completed lambda: ", LAMBDA, " Completion: {:.1%}".format(float(index+1)/N))

#================================================================================================================

def plot_error_kval(data, Nstart, Nstop, k, dataname, plottitle):
    '''Function to plot various error scores based on data gerated by the function "generate_kval_betas".
    The readability of this code has room for improvement, but all in all, it is just book-keeping
    to present the data in a presentable fashion. In other words, there is nothing fancy going on here.
    '''

    partition = load_object_from_file("./kval/partition.pickle")
    observations = data.observations_vector

    N = Nstop-Nstart
    lambdas = [float(10**index) for index in range(Nstart,Nstop+1)]

    def get_errors(index = None,LAMBDA=None,method_name=None):
        out = {}
        if method_name==None:
            betas = load_object_from_file("./kval/beta_{}_{}val_ols.pickle".format(dataname,k))
        elif method_name=="ridge":
            betas = load_object_from_file("./kval/beta_{}_{}val_ridge_lambda_{}.pickle".format(dataname,k,index))
        elif method_name=="lasso":
            betas = load_object_from_file("./kval/beta_{}_{}val_lasso_lambda_{}.pickle".format(dataname,k,index))

        # initiate 0 data in dictionary
        for part in ["train","test"]:
            for score in ["R2","MSE","bias","var"]:
                out[score+"_"+part] = 0

        # add data to dictionary
        for index, beta in enumerate(betas):
            test_data = partition[index]
            train_data = [x for j,x in enumerate(partition) if j!=index]
            train_data = sum(train_data, [])
            model = data.model(beta)
            model.reshape(np.size(model))

            # sum training scores
            out['R2_train']+= R2(observations[train_data],model[train_data])
            out['MSE_train']+= MSE(observations[train_data],model[train_data])
            out['bias_train']+= bias(observations[train_data],model[train_data])
            out['var_train']+= var(model[train_data])

            # sum test scores
            out['R2_test'] += R2(observations[test_data],model[test_data])
            out['MSE_test'] += MSE(observations[test_data],model[test_data])
            out['bias_test'] += bias(observations[test_data],model[test_data])
            out['var_test'] += var(model[test_data])

        # take averages
        for part in ["train","test"]:
            for score in ["R2","MSE","bias","var"]:
                out[score+"_"+part] /= k

        return  out

    plot_data = {}

    for  method_name in ["ols","ridge","lasso"]:
        for part in ["train","test"]:
            for score in ["R2","MSE","bias","var"]:
                plot_data[method_name+"_"+score+"_"+part]=np.ones(N+1) #for example plot_data["ols_R2_train"]

    def m2color(string):
        #method -> color
        if string == "ols":
            output = 'blue'
        if string == "ridge":
            output = 'red'
        if string == "lasso":
            output = 'green'
        return output

    out = get_errors()
    for part in ["train","test"]:
        for score in ["R2","MSE","bias","var"]:
            plot_data['ols_'+score+'_'+part] *= out[score+'_'+part]

    for index, LAMBDA in enumerate(lambdas):
        for method_name in ["ridge","lasso"]:
            out = get_errors(index=index, LAMBDA = LAMBDA, method_name = method_name)
            for part in ["train","test"]:
                for score in ["R2","MSE","bias","var"]:
                    plot_data[method_name+'_'+score+'_'+part][index] = out[score+'_'+part]

    plt.figure("kval_R2")
    for method_name in ["ols","ridge","lasso"]:
            plt.plot(np.log10(lambdas),np.clip(plot_data[method_name+"_R2_test"],0,1),color=m2color(method_name),label = method_name+': test')
            plt.plot(np.log10(lambdas),np.clip(plot_data[method_name+"_R2_train"],0,1),color=m2color(method_name),linestyle='--',label = method_name+': train')
    #plt.axis([Nstart, Nstart+N-1, 0, 1.1])
    plt.xlabel('log $\lambda$')
    plt.ylabel('$R^2$ score')
    plt.title(plottitle+"\n {}-fold cross validation".format(k))
    plt.legend(loc='best')
    plt.grid(True)
    plt.show(block=False)

    for method_name in ["ols","ridge","lasso"]:
        plt.figure(method_name)
        for score in ["MSE","bias","var"]:
            plt.plot(np.log10(lambdas),plot_data[method_name+"_"+score+"_test"],label = score+': test')
           # plt.plot(np.log10(lambdas),plot_data[method_name+"_"+score+"_train"],linestyle='--',label = score+': train')
        plt.xlabel('log $\lambda$')
        plt.ylabel('Error')
        plt.title(plottitle+' ('+method_name+')'+"\n {}-fold cross validation".format(k))
        plt.legend(loc='best')
        plt.grid(True)

    # print R2 scores
    for method_name in ["ols","ridge","lasso"]:
        print(plot_data[method_name+"_R2_test"])
    
   
    plt.show(block=False)

#=====================================================================================================================



#=======================================================================================================================

