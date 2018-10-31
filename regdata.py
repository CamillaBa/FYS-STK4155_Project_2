from Methods import*
import numpy as np
from scipy import linalg

class regdata:
    def __init__(self, data_points, observations, row_design_matrix):
        observations_shape = np.shape(observations)
        number_of_observations = np.prod(observations_shape)
        number_indep_variables = int(np.size(data_points)/number_of_observations)

        # reshape observations to vector
        observations_vector = np.copy(observations).reshape(number_of_observations)
        data_points_vector =  np.copy(data_points).reshape((number_of_observations,number_indep_variables))
        number_basis_elts = len(row_design_matrix(data_points_vector[0]))

        # make design matrix
        X = np.zeros((number_of_observations,number_basis_elts))
        for index,  variable_value in enumerate(data_points_vector):
            X[index,:] = row_design_matrix(variable_value)
        print("Design matrix size = {} GB.".format(float(X.nbytes)/1e9))

        # self variables (in alphabetic order)
        self.number_basis_elts = number_basis_elts
        self.number_of_observations = number_of_observations
        self.observations_shape = observations_shape
        self.observations_vector = observations_vector
        self.data_points_vector = data_points_vector
        self.X = X
        self.XT = np.transpose(X)

    # Regression
    def get_reg(self, LAMBDA = None, epsilon = None):
        '''Returns the polynomial fit as a numpy array. If  LAMBDA = None, epsilon = None the fit is based on an ordinary least square.
        If  LAMBDA = float, epsilon = None, then the fit is found using Ridge for the given bias LAMBDA. If  LAMBDA = float, epsilon = float,
       then the fit is found using lasso. See the function " __get_beta" for more details.'''

        X=self.X; observations =self.observations_vector #relabeling self variables
        beta = self.get_beta(X,observations,LAMBDA=LAMBDA,epsilon=epsilon) #obtaining beta
        model = self.model(beta) #obtaining model from coefficients beta
        return model

    # Get beta (given X and z)
    def get_beta(self, X, observations, LAMBDA = None, epsilon = None):
        '''Returns coefficients for a given beta as a numpy array, found using either ordinary least square,
        Ridge or Lasso regression depending on the arguments. If *args is empty, then beta is found using
        ordinary least square. If *args contains a number it will be treated as a bias LAMBDA for a Ridge regression.
        If *args contains two numbers, then the first will count as a LAMBDA and the second as a tolerance epsilon.
        In this case beta is found using a shooting algorithm that runs until it converges up to the set tolerance.
        '''

        XT = np.transpose(X)
        beta = np.matmul(XT,X)                   
        if type(LAMBDA)==float: #Ridge parameter LAMBDA 
            beta[np.diag_indices_from(beta)]+=LAMBDA
        beta = SVDinv(beta)
        beta = np.matmul(beta,XT)
        beta = np.matmul(beta,observations)

        #Shooting algorithm for Lasso
        if type(epsilon)==float:
            D = self.number_basis_elts
            ints = np.arange(0,D,1)
            beta_old = 0.0
            iterations = 0
            while np.linalg.norm(beta-beta_old)>epsilon and iterations <=25:
                beta_old = np.copy(beta)
                for j in range(0,D):
                    aj = 2*np.sum(X[:,j]**2)
                    no_j = ints[np.arange(D)!=j]
                    cj = 2*np.sum(np.multiply(X[:,j],(observations-np.matmul(X[:,no_j],beta[no_j]))))
                    if cj<-LAMBDA:
                        beta[j]=(cj+LAMBDA)/aj
                    elif cj > LAMBDA:
                        beta[j]=(cj-LAMBDA)/aj
                    else:
                        beta[j]=0.0
                iterations += 1
        return beta

    def get_beta_logistic(self, method_name = 'NR'):
        if method_name == 'NR':
            beta = self.beta_NR()
        if method_name == 'SGD':
            beta = self.beta_SGD()
        return beta

    def beta_NR(self, beta_init=None, epsilon = 0.01):
        # load self variables
        X= self.X; XT = self.XT
        observations = self.observations_vector
        number_basis_elts=self.number_basis_elts
        N = self.number_of_observations

        # initiate sequence to converged beta
        if beta_init.all() == None:
            beta = np.ones(self.number_basis_elts)*0.001
        else:
            beta = beta_init
        beta_old = 100
        iterations = 0

        # start Newtons algorithm
        while np.linalg.norm(beta-beta_old)>epsilon and iterations <25:
            beta_old = np.copy(beta)
            W = np.zeros((N,N))
            p = np.zeros(N)

            #construct W and p
            for i in range(0,N):
                Xi=X[i,:]
                p_i = sigmoid(np.dot(beta,Xi))
                W[i,i]=p_i*(1-p_i)
                p[i]=p_i
            XTWX_inv = SVDinv(np.linalg.multi_dot((XT,W,X)))
            beta += np.linalg.multi_dot((XTWX_inv,XT,(observations-p)))
            iterations +=1
            print(iterations)
        return beta

    def beta_SGD(self,beta_init=None,epsilon=0.001,step_size=0.01,M=5,max_number_iterations = 1000):
        # load self variables
        X= self.X; XT = self.XT
        observations = self.observations_vector
        number_basis_elts=self.number_basis_elts
        N = self.number_of_observations
        if type(beta_init) == np.ndarray:
            beta = beta_init
        else:
            beta = np.ones(self.number_basis_elts)*0.001
        beta_old = 100
        iterations = 0

        # load batches
        batches = self.get_data_partition(M)
        terms = 1000
        # start stochastic gradient descent algorithm
        while np.linalg.norm(terms)>epsilon and iterations<max_number_iterations:
            beta_old = np.copy(beta)
            k = np.random.randint(M)
            batch = batches[k]
            directional_derivative_beta = 0
            for i in batch:
                Xi= X[i,:]
                directional_derivative_beta += (observations[i]-sigmoid(np.dot(Xi,beta)))*Xi
            beta += directional_derivative_beta*step_size
            iterations +=1

        return beta

    # Get model given beta
    def model(self,beta,method_name=None):
        '''Returns model in same shape as observations (for easy comparison).
        '''
        observation_shape = self.observations_shape
        X = self.X
        model = np.matmul(X,beta)

        # if logistic regression is permoned set output to 0 and 1
        if method_name=='log':
            model = classifyer(model)

        model.reshape(observation_shape)
        return model

    def get_data_partition(self,k):
        ''' Creates a random partition of k (almost) equally sized parts of the array
        {1,2,...,number_observations}. This can be used to make training/testing data.
        '''
        number_observations = self.number_of_observations
        indices = np.arange(number_observations)
        indices_shuffle = np.arange(number_observations)
        np.random.shuffle(indices_shuffle)
        partition = []
        for step in range(0,k):
            part = list(indices_shuffle[step:number_observations:k])
            partition.append(part) 
        return partition

    def bootstrap_step(self, samplesize, LAMBDA = None, epsilon = None):
        # relabeling self variables
        number_of_observations =  self.number_of_observations
        observations = self.observations;
        X = self.X; 

        # drawing random sample
        integers = np.random.randint(low=0, high=mn-1, size=samplesize)
        observations_new =  observations[integers]
        X_new = X[integers,:]
        betanew = self.get_beta(X_new,observations_new,LAMBDA = LAMBDA, epsilon = epsilon)
        return betanew

    # Variance/ covariance matrix
    def var_covar_matrix(self,reg):
        ''' Returns the variance/covariance matrix for beta based on the given data.
        This matrix is derived from a statistical viewpoint, where one assumes beta to
        have a normal distribution.
        '''
        p = self.number_basis_elts; invXTX = self.invXTX; N = self.number_of_observations; observations = self.observations # Relabeling self variables
        sigma2=1.0/(N-p-1)*np.sum((observations-reg)*(observations-reg))
        return sigma2*invXTX # OBS! Based on matrix inversion. Inaccurate for  N,p>>0.