import numpy as np
from matplotlib import pyplot as plt
import math
N = 25
X = np.reshape(np.linspace(0, 0.9, N), (N, 1))
Y = np.cos(10*X**2) + 0.1 * np.sin(100*X)

plt.figure()
# #plt.axes([-0.3,1.3,0,2])
# plt.plot(X,Y, '+')
# plt.xlabel("$x$")
# plt.ylabel("$y$")


def poly_features(X, K):

    # X: inputs of size N x 1
    # K: degree of the polynomial
    # computes the feature matrix Phi (N x (K+1))

    X = X.flatten()
    N = X.shape[0]

    #initialize Phi
    Phi = np.zeros((N, K+1))

    # Compute the feature matrix in stages
    for i in range(len(Phi)):
        for j in range(len(Phi[0])):
            Phi[i,j] = X[i] ** j
    print(Phi)
    return Phi

def nonlinear_features_maximum_likelihood(Phi, y):
    # Phi: features matrix for training inputs. Size of N x D
    # y: training targets. Size of N by 1
    # returns: maximum likelihood estimator theta_ml. Size of D x 1

    kappa = 1e-08 # 'jitter' term; good for numerical stability

    D = Phi.shape[1]
    print(Phi.shape)

    # maximum likelihood estimate
    theta_ml = np.linalg.inv(Phi.T @ Phi + kappa * np.identity(D)) @ Phi.T @ y


    return theta_ml


def poly_features_trig(X, K):

    # X: inputs of size N x 1
    # K: degree of the polynomial
    # computes the feature matrix Phi (N x (K+1))

    X = X.flatten()
    N = X.shape[0]

    #initialize Phi
    Phi = np.zeros((N, (2*K+1)))

    # Compute the feature matrix in stages
    for i in range(len(Phi[0])):
        if (i == 0):
            Phi[:,i] = 1
            #print(i,Phi[:,i])
        elif (i % 2 == 0):
            Phi[:,i] = np.cos(2*math.pi*(i/2)*X)
            #print(i,Phi[:,i])
        else:
            Phi[:,i] = np.sin(2*math.pi*((i+1)/2)*X)
            #print(i,Phi[:,i])
    return Phi

def leave_one_out_cross(X,Y,orders):
    av_errors = []
    rmses = []
    for i in orders:
        errors = []
        print(i)
        K = i
        for j in range(len(X)):
            x = X[j]
            y = np.cos(10*x**2) + 0.1 * np.sin(100*x)
            X_new = np.delete(X,j)
            Phi = poly_features_trig(X_new, K)
            Y_new = np.delete(Y,j)
            theta_ml = nonlinear_features_maximum_likelihood(Phi, Y_new)
            #print(theta_ml.shape)
            Phi_test = poly_features_trig(x, K)
            #print(Phi_test.shape)
            y_pred = Phi_test @ theta_ml
            #print(y_pred, y)
            errors.append(av_s_test_error(y_pred,y))
            #print(x,y,y_pred,errors[j])
        av_errors.append(np.mean(errors))
        rmse_temp = 0
        Phi = poly_features_trig(X, K)
        theta_ml = nonlinear_features_maximum_likelihood(Phi, Y)
        for k in range(len(X)):
            rmse_temp += (Y[k]-poly_features_trig(X[k],K) @ theta_ml) ** 2
        rmse = rmse_temp / N
        rmses.append(rmse.flatten())
        print(rmse)
    #     print(rmses)
    # print(rmses)
    return av_errors, rmses



def av_s_test_error(pred, actual):
    return (pred-actual)**2

orders = [0,1,2,3,4,5,6,7,8,9,10]
errors, rmse = leave_one_out_cross(X,Y,orders)
#print(rmses)
plt.plot(orders,errors)
plt.plot(orders, rmse)
plt.show()


# polys = [1,11]
# for i in polys:
#     print(i)
#     K = i
#     Phi = poly_features_trig(X, K) # N x (K+1) feature matrix
#     theta_ml = nonlinear_features_maximum_likelihood(Phi, Y) # maximum likelihood estimator
#     # test inputs
#     Xtest = np.linspace(-1,1.2,500).reshape(-1,1)
#     # feature matrix for test inputs
#     Phi_test = poly_features_trig(Xtest, K)
#     y_pred = Phi_test @ theta_ml # predicted y-values
#
#     plt.plot(Xtest, y_pred)
#     plt.xlabel("$x$")
#     plt.ylabel("$y$")
#
# plt.show()
