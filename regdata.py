import numpy as np
from scipy import linalg
from Methods import*




def default_DC(a,target):
    '''Default derivative of cost function in the neural network class.
    '''
    return a-target

def Dsigmoid(z):
    '''Derivative of sigmoid function. This is the default activation function
    in the neural network class.'''
    if type(z) == float:
        output = sigmoid(z)*(1-sigmoid(z))
    else:
        output = sigmoid_vec(z)*(1-sigmoid_vec(z))
    return output

class layer:
    def __init__(self,m,n):
        self.weight = np.random.rand(m,n)-0.5
        self.bias = np.random.random(m)-0.5
        self.a = np.zeros(m)
        self.z = np.zeros(m)

class learning_rate:
    def __init__(self,t0,t1):
        self.t0 = t0
        self.t1 = t1
    def gamma(self,e,M,i):
        # M is the number of batches
        # e is the current epoch
        # i = 0,1,...,M-1
        return self.t0/(e*M+i+self.t1)


#=========================================================================================================================
# Class for neural network regression
#=========================================================================================================================

class MLP:
    def __init__(self, network_size, DC = 'default', A_DA = 'default'):

        # dimensions of (succesive) weight matrices
        L = len(network_size)

        # initiate hidden layers and output layer
        layers = [0]
        for index in range(1,L):
            layers.append(layer(network_size[index],network_size[index-1]))

        # initiate list of delta (error change)
        delta = []
        for index in range(0,L):
            delta.append(np.zeros(network_size[index]))

        # self objects (in alphabetic order)
        self.delta = delta
        self.network_size = network_size
        self.L = L
        self.layers = layers
        self.learn = learning_rate(1,1) # learning rate

        
        if DC == 'default':
            DC = default_DC # derivative of cost function
        if A_DA == 'default':
            A = sigmoid_vec # activation function
            DA = Dsigmoid # derivative of activation function
        else:
            A = A_DA[0]
            DA = A_DA[1]

        # self functions (in alphabetic order)
        self.A = A # activation function
        self.DA = DA # derivative of activation function
        self.DC = DC # derivative of cost function 
        
    def input_to_output(self,input):
        # relabel self variables
        network_size = self.network_size
        layers = self.layers
        L = self.L
        
        # calculate second layer
        layers[1].z = np.dot(layers[1].weight,input) + layers[1].bias
        layers[1].a = self.A(layers[1].z)

        # calculate remaining layers
        for l in range(2,L):
            # feed forward
            layers[l].z=np.dot(layers[l].weight,layers[l-1].a) + layers[l].bias
            layers[l].a= self.A(layers[l].z)
        
        # return ouput layer
        return layers[-1].a

    def model(self,train_X):
        output_layer_size = self.network_size[-1]
        n = len(train_X)
        if output_layer_size > 1:
            model_Y = np.zeros((n,output_layer_size))
            for i, x in enumerate(train_X):
                model_Y[i,:]=self.input_to_output(x)
        else:
            model_Y = np.zeros(n)
            for i, x in enumerate(train_X):
                model_Y[i]=self.input_to_output(x)

        return model_Y

    def backwards_propagation(self, input, target, step_size=0.01):
        # load self variables
        delta = self.delta
        layers = self.layers
        L = self.L
        
        # get output from network
        a = self.input_to_output(input)

        # compute delta_L
        delta[L-1]=self.DA(layers[L-1].z)*self.DC(a,target) #delta[L-1] = a*(1-a)*self.DC(a,target)             Can be substituted to improve speed when working with sigmoid activation function

        # compute remianing delta_l
        for l in range(L-2,1,-1):
            delta[l] = np.dot(np.transpose(layers[l+1].weight),delta[l+1])*self.DA(layers[l].z) #layers[l].a*(1-layers[l].a)      Can be substituted to improve speed when working with sigmoid activation function

        # update weights and biases of network
        for l in range(L-2,1,-1):
            layers[l].bias -= step_size*delta[l]
            layers[l].weight -= step_size*np.outer(delta[l],layers[l-1].a)

    def train(self, training_input, training_output, epoch_start = 0, epochs = 100, number_batches = 1, completion_message = False):
        # divide data into batches
        N = len(training_input)
        batches = get_partition(N,number_batches)

        for epoch in range(epoch_start,epochs+epoch_start):
            for j in range(number_batches):
                k = np.random.randint(number_batches)
                batch = batches[k]
                for x,y in zip(training_input[batch], training_output[batch]):
                    self.backwards_propagation(x,y, step_size = self.learn.gamma(epoch,number_batches,j))

            # print completion to terminal
            if completion_message == True:
                self.__completion_message__(epoch-epoch_start, epochs)

    def __completion_message__(self,current_iteration,iterations):
        # function to print completion percentage to terminal
        completion = int(0.05*iterations)
        if current_iteration % completion == 0:
               print("Completed: {:.1%}".format(current_iteration/iterations))


#=========================================================================================================================
# Class for linear regression
#=========================================================================================================================

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

        # self objects (in alphabetic order)
        self.data_points_vector = data_points_vector
        self.learn = learning_rate(1,1) # learning rate
        self.number_basis_elts = number_basis_elts
        self.number_of_observations = number_of_observations
        self.observations_shape = observations_shape
        self.observations_vector = observations_vector
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
                #print(iterations)
                iterations += 1
        return beta

    def beta_NR(self, beta_init=None, epsilon = 1e-8,max_number_iterations=25):
        # load self variables
        X= self.X; XT = self.XT
        observations = self.observations_vector
        number_basis_elts=self.number_basis_elts
        N = self.number_of_observations

        # initiate beta
        if type(beta_init) == np.ndarray:
            beta = np.copy(beta_init)
        else:
            beta = np.random.random(self.number_basis_elts)-0.5
        beta_old = 100
        iterations = 0

        # start Newtons Raphson algorithm
        while np.linalg.norm(beta-beta_old)>epsilon and iterations <max_number_iterations:
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
        return beta

    def log_beta_SGD(self, beta_init=None, 
                 epoch_start = 0,epochs = 100,
                 M=5):
        # load self variables
        X= self.X; XT = self.XT
        observations = self.observations_vector
        number_basis_elts=self.number_basis_elts
        N = self.number_of_observations
        if type(beta_init) == np.ndarray:
            beta = np.copy(beta_init)
        else:
            beta = np.ones(self.number_basis_elts)*0.001

        # start stochastic gradient descent algorithm
        for epoch in range(epoch_start,epoch_start+epochs):
            batches = get_partition(N,M)
            directional_derivative_beta = 0
            for j in range(M):
                k = np.random.randint(M)
                batch = batches[k]
                for i in batch:
                    Xi= X[i,:]
                    directional_derivative_beta += (observations[i]-sigmoid(np.dot(Xi,beta)))*Xi
                beta += directional_derivative_beta*self.learn.gamma(epoch,M,j)

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