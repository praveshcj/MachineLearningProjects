# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 22:28:17 2019

@author: Pravesh Jain
"""
import numpy as np
import matplotlib.pyplot as plt
fh = open("iris.data", "r")
content = fh.read()
#print(content.split())
list = content.split()
list1 =[]
count =0
X= []
Y =[]
for i in list:
    k = i.split(",")
    X.insert(count,k[0:4])
    Y.append(k[4])
    count= count+1
print(X[2])
X1 = np.zeros((150,4),dtype=float)
count =0
for i in X:
    temp = np.zeros((1,4),dtype=float)
    count1 =0
    for a in i:
        temp[0][count1] =  float(a)
        count1 =count1+1
    X1[count] = temp
    count = count+1
X_train = X1[0:33,:]
X_train = np.append(X_train,X1[50:83,:],axis=0)
X_train = np.append(X_train,X1[100:133,:],axis =0)
X_test = X1[33:50,:]
X_test = np.append(X_test,X1[83:100,:],axis=0)
X_test = np.append(X_test,X1[133:150,:],axis =0)
X_train= X_train.T
X_test = X_test.T
Y1 = np.zeros((150,1))
Y2 = np.zeros((150,3))
count=0
for i in Y:
    if(i=='Iris-setosa'):
        Y1[count] = 1
        Y2[count][0] = 1
    elif(i=='Iris-versicolor'):
        Y1[count] =2
        Y2[count][1] = 1  
    else:
        Y1[count]=3
        Y2[count][2] = 1
    count = count+1
Y_train = Y2[0:33,:]
Y_train = np.append(Y_train,Y2[50:83,:],axis=0)
Y_train = np.append(Y_train,Y2[100:133,:],axis =0)
Y_test = Y1[33:50,:]
Y_test = np.append(Y_test,Y1[83:100,:],axis=0)
Y_test = np.append(Y_test,Y1[133:150,:],axis =0)
Y_train = Y_train.T
Y_test = Y_test.T
nx = 3;
nl =5;
ny =3;

def layer_sizes(X1, Y2):
    """
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)
    
    Returns:
    n_x -- the size of the input layer
    n_h -- the size of the hidden layer
    n_y -- the size of the output layer
    """
    ### START CODE HERE ### (≈ 3 lines of code)
    n_x = 4 # 
    n_h = 5
    n_y = 3 
    return (n_x, n_h, n_y)

def initialize_parameters(n_x, n_h, n_y):
    
    np.random.seed(2) # we set up a seed so that our output matches ours although the initialization is random.
    
    W1 = np.random.randn(n_h,n_x) * 0.01
    b1 = np.zeros((n_h,1))    
    W2 = np.random.randn(n_h,n_h) * 0.01
    b2 = np.zeros((n_h,1))
    W3 = np.random.randn(n_y,n_h) * 0.01
    b3 = np.zeros((n_y,1))
    
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return parameters


def sigmoid(Z):
    t = 1/(1 + np.exp(-Z))
    return t

def forward_propagation(X,parameters):
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]
    b3 = parameters["b3"]
    Z1 = np.dot(W1,X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2,A1) +b2
    A2 = np.tanh(Z2)
    Z3 = np.dot(W3,A2) +b3
    A3 = sigmoid(Z3)

    
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2,
             "Z3": Z3,
             "A3":A3}
    
    return A3, cache
def compute_cost(A2, Y,parameters):
    
    m = Y.shape[0] # number of example

    # Compute the cross-entropy cost
    logprobs =  np.multiply(np.log(A2),Y) + np.multiply(np.log(1-A2),(1-Y))
    cost = - np.sum(logprobs) /m
    #print(cost)
    
    cost = np.squeeze(cost)     # makes sure cost is the dimension we expect. 
                                # E.g., turns [[17]] into 17 
    assert(isinstance(cost, float))
    
    return cost

def backward_propagation(cache, X, Y, parameters):
    m = X.shape[1]
    
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]    

    A1 = cache["A1"]
    A2 = cache["A2"]
    A3 = cache["A3"]
    dZ3 = A3 - Y
    dW3 = np.dot(dZ3, A2.T)/m
    db3 = np.sum(dZ3,axis =1,keepdims=True)/m
    dZ2 = np.dot(W3.T,dZ3)*(1-np.power(A2,2))
    dW2 = np.dot(dZ2,A1.T)/m
    db2 = np.sum(dZ2,axis =1,keepdims=True)/m
    dZ1 = np.dot(W2.T,dZ2)*(1-np.power(A1,2))
    dW1 = np.dot(dZ1,X.T)/m
    db1 = np.sum(dZ1,axis =1,keepdims=True)/m
    #print(A3)
    ### END CODE HERE ###
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2,
             "dW3": dW3,
             "db3": db3}
    
    return grads

# GRADED FUNCTION: nn_model
def update_parameters(grads, learning_rate, parameters):
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]
    b3 = parameters["b3"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    dW3 = grads["dW3"]
    db3 = grads["db3"]

    #print(W1)
    W1 = W1 - learning_rate*dW1
    b1 = b1 - learning_rate*db1
    W2 = W2 - learning_rate*dW2
    b2 = b2 - learning_rate*db2
    W3 = W3 - learning_rate*dW3
    b3 = b3 - learning_rate*db3
    #print(W1)
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return parameters

def nn_model(X, Y, n_h, num_iterations = 50000, print_cost=True):
    np.random.seed(3)
    parameters = initialize_parameters(4,5,3)
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]
    b3 = parameters["b3"]
    # Loop (gradient descent)

    for i in range(0, num_iterations):
         
        # Forward propagation. Inputs: "X, parameters". Outputs: "A3, cache".
        A3, cache = forward_propagation(X,parameters)
        
        # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
        cost = compute_cost(A3,Y,parameters)
 
        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        grads = backward_propagation(cache, X,Y,parameters)
 
        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
        parameters = update_parameters(grads, 0.08,parameters)
        
    
        # Print the cost every 1000 iterations
        if print_cost and i % 500 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    return parameters
parameters = nn_model(X_train,Y_train,5,50000,True)
Y_out,cache = forward_propagation(X_test,parameters)    
Y_out1 = np.argmax(Y_out,axis=0).reshape((1,Y_out.shape[1])) +1
count=0
count1 =0
for i in range(0,Y_out1.shape[1]):
    if(Y_out1[0,i] == Y_test[0,i]):
        count1 += 1
    count += count
predict = (count1* 100)/Y_out1.shape[1] 
print("% Accuracy = ",predict)
   
