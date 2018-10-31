from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
import numpy as np
from math import exp
from scipy import linalg
import matplotlib.pyplot as plt
import time
from pathlib import Path

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

#================================================================================================================

# Sigmoid function
def sigmoid(x):
    if x >= 0:
        emx= exp(-x)
        output = 1.0 /(1.0+emx)
    else:
        ex= exp(x)
        output = ex/(1+ex)
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

#class regdata:
#    def __init__(self, data_points, observations, row_design_matrix):
#        observations_shape = np.shape(observations)
#        number_of_observations = np.prod(observations_shape)
#        number_indep_variables = int(np.size(data_points)/number_of_observations)

#        # reshape observations to vector
#        observations_vector = np.copy(observations).reshape(number_of_observations)
#        data_points_vector =  np.copy(data_points).reshape((number_of_observations,number_indep_variables))
#        number_basis_elts = len(row_design_matrix(data_points_vector[0]))

#        # make design matrix
#        X = np.zeros((number_of_observations,number_basis_elts))
#        for index,  variable_value in enumerate(data_points_vector):
#            X[index,:] = row_design_matrix(variable_value)

#        # self variables (in alphabetic order)
#        self.number_basis_elts = number_basis_elts
#        self.number_of_observations = number_of_observations
#        self.observations_shape = observations_shape
#        self.observations_vector = observations_vector
#        self.data_points_vector = data_points_vector
#        self.X = X
#        self.XT = np.transpose(X)

#    # Regression
#    def get_reg(self, LAMBDA = None, epsilon = None):
#        '''Returns the polynomial fit as a numpy array. If  LAMBDA = None, epsilon = None the fit is based on an ordinary least square.
#        If  LAMBDA = float, epsilon = None, then the fit is found using Ridge for the given bias LAMBDA. If  LAMBDA = float, epsilon = float,
#       then the fit is found using lasso. See the function " __get_beta" for more details.'''

#        X=self.X; observations =self.observations_vector #relabeling self variables
#        beta = self.get_beta(X,observations,LAMBDA=LAMBDA,epsilon=epsilon) #obtaining beta
#        model = self.model(beta) #obtaining model from coefficients beta
#        return model

#    # Get beta (given X and z)
#    def get_beta(self, X, observations, LAMBDA = None, epsilon = None):
#        '''Returns coefficients for a given beta as a numpy array, found using either ordinary least square,
#        Ridge or Lasso regression depending on the arguments. If *args is empty, then beta is found using
#        ordinary least square. If *args contains a number it will be treated as a bias LAMBDA for a Ridge regression.
#        If *args contains two numbers, then the first will count as a LAMBDA and the second as a tolerance epsilon.
#        In this case beta is found using a shooting algorithm that runs until it converges up to the set tolerance.
#        '''

#        XT = np.transpose(X)
#        beta = np.matmul(XT,X)                   
#        if type(LAMBDA)==float: #Ridge parameter LAMBDA 
#            beta[np.diag_indices_from(beta)]+=LAMBDA
#        beta = SVDinv(beta)
#        beta = np.matmul(beta,XT)
#        beta = np.matmul(beta,observations)

#        #Shooting algorithm for Lasso
#        if type(epsilon)==float:
#            D = self.number_basis_elts
#            ints = np.arange(0,D,1)
#            beta_old = 0.0
#            iterations = 0
#            while np.linalg.norm(beta-beta_old)>epsilon and iterations <=25:
#                beta_old = np.copy(beta)
#                for j in range(0,D):
#                    aj = 2*np.sum(X[:,j]**2)
#                    no_j = ints[np.arange(D)!=j]
#                    cj = 2*np.sum(np.multiply(X[:,j],(observations-np.matmul(X[:,no_j],beta[no_j]))))
#                    if cj<-LAMBDA:
#                        beta[j]=(cj+LAMBDA)/aj
#                    elif cj > LAMBDA:
#                        beta[j]=(cj-LAMBDA)/aj
#                    else:
#                        beta[j]=0.0
#                iterations += 1
#        return beta

#    def get_beta_logistic(self, method_name = 'NR'):
#        if method_name == 'NR':
#            beta = self.beta_NR()
#        if method_name == 'SGD':
#            beta = self.beta_SGD()
#        return beta

#    def beta_NR(self, beta_init=None, epsilon = 0.01):
#        # load self variables
#        X= self.X; XT = self.XT
#        observations = self.observations_vector
#        number_basis_elts=self.number_basis_elts
#        N = self.number_of_observations

#        # initiate sequence to converged beta
#        if beta_init.all() == None:
#            beta = np.ones(self.number_basis_elts)*0.001
#        else:
#            beta = beta_init
#        beta_old = 100
#        iterations = 0

#        # start Newtons algorithm
#        while np.linalg.norm(beta-beta_old)>epsilon and iterations <25:
#            beta_old = np.copy(beta)
#            W = np.zeros((N,N))
#            p = np.zeros(N)

#            #construct W and p
#            for i in range(0,N):
#                Xi=X[i,:]
#                p_i = sigmoid(np.dot(beta,Xi))
#                W[i,i]=p_i*(1-p_i)
#                p[i]=p_i
#            XTWX_inv = SVDinv(np.linalg.multi_dot((XT,W,X)))
#            beta += np.linalg.multi_dot((XTWX_inv,XT,(observations-p)))
#            iterations +=1
#            print(iterations)
#        return beta

#    def beta_SGD(self,beta_init=None,epsilon=0.01,step_size=0.01,M=5):
#        # load self variables
#        X= self.X; XT = self.XT
#        observations = self.observations_vector
#        number_basis_elts=self.number_basis_elts
#        N = self.number_of_observations
#        # initiate sequence to converged beta
#        if beta_init == None:
#            beta = np.ones(self.number_basis_elts)*0.001
#        else:
#            beta = beta_init
#        beta_old = 100
#        iterations = 0

#        # load batches
#        batches = self.get_data_partition(M)

#        # start stochastic gradient descent algorithm
#        #iterations = 0
#        while np.linalg.norm(beta-beta_old)>epsilon:# and iterations <10000:
#            beta_old = np.copy(beta)
#            k = np.random.randint(M)
#            batch = batches[k]
#            terms = 0
#            for i in batch:
#                Xi= X[i,:]
#                terms += (observations[i]-sigmoid(np.dot(Xi,beta)))*Xi
#            beta += terms
#            #iterations +=1
#            #print(iterations)

#        return beta

#    # Get model given beta
#    def model(self,beta,method_name=None):
#        '''Returns model in same shape as observations (for easy comparison).
#        '''
#        observation_shape = self.observations_shape
#        X = self.X
#        model = np.matmul(X,beta)

#        # if logistic regression is permoned set output to 0 and 1
#        if method_name=='log':
#            model = classifyer(model)

#        model.reshape(observation_shape)
#        return model

#    def get_data_partition(self,k):
#        ''' Creates a random partition of k (almost) equally sized parts of the array
#        {1,2,...,number_observations}. This can be used to make training/testing data.
#        '''
#        number_observations = self.number_of_observations
#        indices = np.arange(number_observations)
#        indices_shuffle = np.arange(number_observations)
#        np.random.shuffle(indices_shuffle)
#        partition = []
#        for step in range(0,k):
#            part = list(indices_shuffle[step:number_observations:k])
#            partition.append(part) 
#        return partition

#    def bootstrap_step(self, samplesize, LAMBDA = None, epsilon = None):
#        # relabeling self variables
#        number_of_observations =  self.number_of_observations
#        observations = self.observations;
#        X = self.X; 

#        # drawing random sample
#        integers = np.random.randint(low=0, high=mn-1, size=samplesize)
#        observations_new =  observations[integers]
#        X_new = X[integers,:]
#        betanew = self.get_beta(X_new,observations_new,LAMBDA = LAMBDA, epsilon = epsilon)
#        return betanew

#    # Variance/ covariance matrix
#    def var_covar_matrix(self,reg):
#        ''' Returns the variance/covariance matrix for beta based on the given data.
#        This matrix is derived from a statistical viewpoint, where one assumes beta to
#        have a normal distribution.
#        '''
#        p = self.number_basis_elts; invXTX = self.invXTX; N = self.number_of_observations; observations = self.observations # Relabeling self variables
#        sigma2=1.0/(N-p-1)*np.sum((observations-reg)*(observations-reg))
#        return sigma2*invXTX # OBS! Based on matrix inversion. Inaccurate for  N,p>>0.

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

def generate_kval_betas(data, Nstart, Nstop, k, dataname, epsilon = 0.01):
    N = Nstop-Nstart # number of lambdas
    lambdas = [float(10**index) for index in range(Nstart,Nstop+1)] #interval of lambdas
    partition = data.get_data_partition(k)
    print_list_of_lists(partition,"partition.txt") #write partition to file "partition.txt"
    X = data.X
    observations = data.observations_vector
    # loop through partition and calculate betas and append to list
    def print_betas(file, LAMBDA = None, epsilon = None):
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
        # print betas to file
        N = len(betas[0])
        for i in range(0,N):
            outstring = ""
            for beta in betas:
                outstring += str(beta[i])
                outstring += ","
            outstring= outstring[:-1]
            if i < N:
                outstring += "\n"
            file.write(outstring)
    
    file = open("beta_{}_{}val_ols.txt".format(dataname,k), "w")
    print_betas(file)
    file.close()

    for index, LAMBDA in enumerate(lambdas):
        file = open("beta_{}_{}val_ridge_lambda_{}.txt".format(dataname,k,index), "w")
        print_betas(file,LAMBDA=LAMBDA)
        file.close()
        file = open("beta_{}_{}val_lasso_lambda_{}.txt".format(dataname,k,index), "w")
        print_betas(file,LAMBDA=LAMBDA,epsilon=epsilon)
        file.close()
        print("Completed lambda: ", LAMBDA, " Completion: {:.1%}".format(float(index+1)/N))

def plot_error_kval(data, Nstart, Nstop, k, dataname, plottitle):
    partition = read_list_of_lists("partition.txt")
    observations = data.observations_vector

    N = Nstop-Nstart
    lambdas = [float(10**index) for index in range(Nstart,Nstop+1)]

    def get_errors(index = None,LAMBDA=None,method_name=None):
        out = {}
        if method_name==None:
            betas = np.loadtxt("beta_{}_{}val_ols.txt".format(dataname,k), delimiter=',', unpack=True)
        elif method_name=="ridge":
            betas = np.loadtxt("beta_{}_{}val_ridge_lambda_{}.txt".format(dataname,k,index), delimiter=',', unpack=True)
        elif method_name=="lasso":
            betas = np.loadtxt("beta_{}_{}val_lasso_lambda_{}.txt".format(dataname,k,index), delimiter=',', unpack=True)

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

            # average test scores
            out['R2_test'] += R2(observations[test_data],model[test_data])
            out['MSE_test'] += MSE(observations[test_data],model[test_data])
            out['bias_test'] += bias(observations[test_data],model[test_data])
            out['var_test'] += var(model[test_data])

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
    
   
    plt.show(block=False)

#=====================================================================================================================

































#class k_cross_validation:
#    def __init__(self, data, partition, LAMBDA = None, epsilon = None):
#        self.data = data; self.partition = partition; self.LAMBDA = LAMBDA; self.epsilon = epsilon;
#        self.test_R2, self.test_var, self.test_bias, self.test_MSE, self.test_extra_terms = 0, 0, 0, 0, 0
#        self.train_R2 = 0
        
#        #self.train_var, self.train_bias, self.train_MSE, self.train_extra_terms = 0, 0, 0, 0

#    def R2(self):
#        # variables
#        partition = self.partition
#        data = self.data
#        X = data.X; 
#        observations = data.observations_vector; 
#        k = len(partition)
#        test_R2, train_R2 = 0, 0

#        # loop through partition and calculate errors
#        for i, test_data in enumerate(partition):
#            train_data = [x for j,x in enumerate(partition) if j!=i]
#            train_data = sum(train_data, [])

#            # find model for training data
#            beta = data.get_beta(X[train_data],
#                                 observations[train_data],
#                                 LAMBDA=self.LAMBDA,
#                                 epsilon=self.epsilon)
#            model = data.model(beta)
#            model.reshape(np.size(observations))

#            # test errors:
#            observations_test = observations[test_data]
#            model_test = model[test_data]
#            test_R2 += R2(observations_test,model_test )

#            # training errors:
#            observations_train = observations[train_data]
#            model_train = model[train_data]
#            train_R2 += R2(observations_train,model_train)

#        # update self variables
#        self.test_R2 = test_R2/k
#        self.train_R2 = train_R2/k

#    def MSE(self):
#        # variables
#        partition = self.partition
#        LAMBDA = self.LAMBDA
#        epsilon = self.LAMBDA
#        data = self.data
#        X = data.X; 
#        observations = data.observations_vector; 
#        k = len(partition)
#        test_var, test_bias, test_MSE, test_extra_terms = 0, 0, 0, 0
#        #train_var, train_bias, train_MSE, train_extra_terms = 0, 0, 0, 0

#        # loop through partition and calculate errors
#        for i, test_data in enumerate(partition):
#            train_data = [x for j,x in enumerate(partition) if j!=i]
#            train_data = sum(train_data, [])

#            # find model for training data
#            beta = data.get_beta(X[train_data],
#                                 observations[train_data],
#                                 LAMBDA=self.LAMBDA,
#                                 epsilon=self.epsilon)
#            model = data.model(beta)
#            model.reshape(np.size(observations))

#            # test errors:
#            observations_test = observations[test_data] # doesn't work, as with subset??
#            model_test = model[test_data]
#            test_var += var(observations_test) 
#            test_bias += bias(observations_test,model_test)
#            test_MSE += MSE(observations_test,model_test)
#            test_extra_terms += extra_term(observations_test,model_test)

#            ##training errors:
#            #ftrain = get_subset(f,train_data); fregtrain = get_subset(freg,train_data)
#            #train_var += var(fregtrain) 
#            #train_bias += bias(ftrain,fregtrain)
#            #train_MSE += MSE(ftrain,fregtrain)
#            #train_extra_terms += extra_term(ftrain,fregtrain)

#        # self variables
#        self.test_var = test_var/k
#        self.test_bias = test_bias/k
#        self.test_MSE = test_MSE/k
#        self.test_extra_terms = test_extra_terms/k

#        #self.train_var = train_var/k
#        #self.train_bias = train_bias/k
#        #self.train_MSE = train_MSE/k
#        #self.train_extra_terms = train_extra_terms/k

#================================================================================================================

#def plot_R2_scores_k_cross_validation(data,Nstart,Nstop,k, plottitle, epsilon = 0.01):
#    ''' This function makes a plot of the R2 scores vs LAMBDA of the best iteration from a k-fold cross validation on 
#    the data set from the given data. Best in the sense that the fit had the highest R2 score on testing data. The same 
#    partition of the data set is used for each lambda, and each time we select the best training data on which we base the model.
#    See "k_cross_validation" for more details.'''

#    N = Nstop-Nstart # number of lambdas

#    # Comparing R2 scores, regression with fixed degree, variable LAMBDA
#    lambdas = np.zeros(N)
#    partition = data.get_data_partition(k)

    
#    R2_Lasso_test_data = np.zeros(N)
#    R2_Lasso_training_data = np.zeros(N)
#    R2_Ridge_test_data = np.zeros(N)
#    R2_Ridge_training_data = np.zeros(N)

#    # OLS R2 score
#    print("begin ols")
#    kval = k_cross_validation(data,partition)
#    kval.R2()
#    R2score_ols_test, R2score_ols_train = kval.test_R2, kval.train_R2
#    R2_ols_test_data = np.ones(N)*R2score_ols_test
#    R2_ols_training_data = np.ones(N)*R2score_ols_train

#    print("ols completed")

#    for i in range(0,N): 
#        LAMBDA = float(10**(Nstart+i))
#        lambdas[i]=LAMBDA
#        print("begin ridge")
#        kval = k_cross_validation(data,partition,LAMBDA=LAMBDA)
#        kval.R2()
        
#        # Ridge R2 score
#        R2score_ridge_test, R2score_ridge_train = kval.test_R2, kval.train_R2
#        R2_Ridge_test_data[i] = R2score_ridge_test
#        R2_Ridge_training_data[i] = R2score_ridge_train

#        print("ridge completed")
#        print("begin lasso")

#        kval = k_cross_validation(data,partition,LAMBDA=LAMBDA,epsilon=epsilon)
#        kval.R2()
#        print("lasso completed")
#        # Lasso R2 score
#        R2score_lasso_test, R2score_lasso_train = kval.test_R2, kval.train_R2
#        R2_Lasso_test_data[i] = R2score_lasso_test
#        R2_Lasso_training_data[i] = R2score_lasso_train

#        print("Completed lambda: ", LAMBDA, " Completion: {:.1%}".format(float(i)/(N-1)))

#    plt.figure()
#    plt.plot(np.log10(lambdas),np.clip(R2_ols_test_data,0,1),color='blue')
#    plt.plot(np.log10(lambdas),np.clip(R2_ols_training_data,0,1),color='blue',linestyle='--')
#    plt.plot(np.log10(lambdas),np.clip(R2_Ridge_test_data,0,1),color='red')
#    plt.plot(np.log10(lambdas),np.clip(R2_Ridge_training_data,0,1),color='red',linestyle='--')
#    plt.plot(np.log10(lambdas),np.clip(R2_Lasso_test_data,0,1),color='green')
#    plt.plot(np.log10(lambdas),np.clip(R2_Lasso_training_data,0,1),color='green',linestyle='--')
#    plt.axis([Nstart, Nstart+N-1, 0, 1.1])
#    plt.xlabel('log $\lambda$')
#    plt.ylabel('$R^2$ score')
#    plt.legend(('OLS: test data', 'OLS: training data','Ridge: test data', 'Ridge: training data','Lasso: test data', 'Lasso: training data'))
#    plt.title(plottitle)
#    plt.grid(True)
#    plt.show(block=False)

















    ## MSE OLS
    #MSE_ols_test, MSE_ols_train = out['MSE_test'], out['MSE_train']
    #MSE_ols_test = np.ones(N+1)*R2score_ols_test
    #MSE_ols_train = np.ones(N+1)*R2score_ols_train

    #bias_ols_test, bias_ols_train = out['bias_test'], out['bias_train']
    #bias_ols_test = np.ones(N+1)*R2score_ols_test
    #bias_ols_train = np.ones(N+1)*R2score_ols_train

    #var_ols_test, var_ols_train = out['var_test'], out['var_train']
    #var_ols_test = np.ones(N+1)*R2score_ols_test
    #var_ols_train = np.ones(N+1)*R2score_ols_train

    ## R2 scores Ridge and Lasso
    #R2_ridge_test, R2_ridge_train = np.zeros(N+1), np.zeros(N+1)
    #R2_lasso_test, R2_lasso_train = np.zeros(N+1), np.zeros(N+1)

    ## MSE Ridge and Lasso
    #MSE_ridge_test, MSE_ridge_train = np.zeros(N+1), np.zeros(N+1)
    #bias_ridge_test, bias_ridge_train = np.zeros(N+1), np.zeros(N+1)
    #var_ridge_test, var_ridge_train = np.zeros(N+1), np.zeros(N+1)

    #MSE_lasso_test, MSE_lasso_train = np.zeros(N+1), np.zeros(N+1)
    #bias_lasso_test, bias_lasso_train = np.zeros(N+1), np.zeros(N+1)
    #var_lasso_test, var_lasso_train = np.zeros(N+1), np.zeros(N+1)