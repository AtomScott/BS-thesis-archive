# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 12:43:31 2019

@author: Lincon
"""

import torch
#import torch.nn as nn
#import geoopt
#import geoopt.manifolds as mf
#import Grassmann as Gr
#from torch.distributions.multivariate_normal import MultivariateNormal
#from scipy.io import loadmat
#import h5py
import numpy as np
import matplotlib.pyplot as plt
a= 

def norm(x): 
    # norm of a vector
    if torch.is_tensor(x): 
        return x.pow(2).sum(0).sqrt() 
    else: 
        return np.sqrt(np.sum(x**2, axis=0))
    

#def normalize(A): 
#    for i in range(A.shape[2]):
#            A[:,:,i] = A[:,j,i]/norm(A[:,:,i])
#    return A
def normalize(x): return x/norm(x)

def normalize_dataset(A):
    # normalize dataset stored in tensor A, 
    # dim 0 assumed to be feature space dimension
    
    # takes the norm along dim 0, repeat into vector and reciprocal (invert) 
    norm_A = A.norm(dim=0).expand(A.shape).reciprocal()
    return A * norm_A

def hottaBasisVectors(X,sub_dim):
    d,n = X.shape
    if d < n:
        C = torch.mm(X, X.transpose(0,1))
        matrank = torch.matrix_rank(C)
        tmp_val, tmp_vec = torch.eig(C, eigenvectors = True)
        value, index = torch.sort(tmp_val,descending = True)
        eig_vec = tmp_vec[:,index[:matrank]]
        eig_val = value[:matrank]
        eig_vec  =  eig_vec[:,:sub_dim]
        eig_val  =  value[:sub_dim]
    else:
        C = torch.mm(X.transpose(0,1), X)
        matrank = torch.matrix_rank(C)
        tmp_val, tmp_vec = torch.eig(C, eigenvectors = True)
        
        # second column is zero if the eig vals are real
        tmp_val = tmp_val[:,0]

        value, index = torch.sort(tmp_val,descending = True)
#        tmp_vec = tmp_vec[:,index[:matrank]]
#        eig_vec = torch.mm(X, tmp_vec)
        eig_vec = torch.zeros((X.shape[0],matrank))
        for i in range(matrank):
            eig_vec[:,i] = (X.mv(tmp_vec[:,index[i]])).div((value[i]).sqrt())
        eig_vec  =  normalize_dataset(eig_vec[:,:sub_dim])
        eig_val  =  value[:sub_dim]

    return eig_vec, eig_val

def dataset_PCA(data,sub_dim):
    # data should be a tensor shaped dim x n_samples

    sx = data.shape
    data = data.reshape(sx[0],sx[1],-1)
    n_sets = data.shape[2]
    
    eig_vec = torch.zeros((sx[0],sub_dim,n_sets))
    eig_val = torch.zeros((sub_dim,n_sets))
    for i in range(n_sets):
        eig_vec[:,:,i], eig_val[:,i]= hottaBasisVectors(data[:,:,i], sub_dim)
    
    eig_vec = eig_vec.reshape((sx[0],sub_dim,*sx[2:]))
    eig_val = eig_val.reshape((sub_dim,*sx[2:]))

    return eig_vec, eig_val

def collapse(X, n):
    sx = X.shape
    new_sx = (*sx[0:n],sx[n]*sx[n+1],*sx[n+2:])
    return X.reshape(new_sx)

def imshow(img, img_size = None):
#    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if npimg.shape.__len__() == 1:
        if img_size == None:
            n = int(np.sqrt(npimg.shape))
            npimg = npimg.reshape((n,n))
        else:
            npimg = npimg.reshape(img_size)

    
#    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    fig = plt.figure()
    plt.imshow(npimg)
    plt.show()

def is_orthogonal(X):
    # is matrix X orthogonal?
    tp = ((X.transpose(0,1).matmul(X) - torch.eye(X.shape[1])) > 0.01)
    return False if tp.sum() > 0 else True

def is_normalized(X):
    # are the columns of X normalized?
    tp = torch.norm(X, dim=0).mean()
    return False if tp > 1.001 or tp < 0.999 else True

#def gen_label(n_samples, n_class):
#    tp = np.arange(n_class)
#    tp2 = np.ones(n_samples, dtype='uint8')
#    
#    return np.kron(tp, tp2)
##    return np.kron(tp2, tp)
    
def gen_label(n_samples, n_class, transpose=False):
    tp = torch.zeros(n_samples, n_class)
    for c in range(n_class):
        for i in range(n_samples):
            tp[i,c] = c
    
#    if transpose == True:
#        tp = tp.t()
    
    return tp.reshape(-1)

def similarity(x, y, transpose = True):
    if transpose == True:
        x = x.t()

    U, S, V = torch.mm(x,y).svd()
    # mean of canonical angles
    return S.pow(2).mean()

def similarity_matrix(X, Y):
    X = X.reshape((*X.shape[0:2],-1))
    Y = Y.reshape((*Y.shape[0:2],-1))
    
    S = torch.zeros((X.shape[2], Y.shape[2]))
    for i in range(X.shape[2]):
        for j in range(Y.shape[2]):
             S[i,j] = similarity(X[:,:,i], Y[:,:,j])
    
    return S

def compute_accuracy(S, labels_trn, labels_eva):
    a, nearest = S.max(0); # calculate the nearest neighbor to each test point
    predics = labels_trn[nearest]; # prediction is the class of the nearest n.
    H = predics == labels_eva; # check whether the prediction matches true label
    
    accuracy = 100*(H.mean()); #accuracy
    # Results.orth_degree = 1 - mean(mean(S));
    return accuracy

def sqrd_euclidean_vec(x, y):
     """
       Computes the squared euclidean distance between vectors x and y

        Parameters:
        -----------
        x and y: tensor of shape dim

        Returns:
        --------
        d: squared euclidean distance
       """
     return torch.sum((x - y).pow(2))

def similarity_vec(x, y):
     """
       Computes the similarity (squared cosine distance)
        between vectors x and y

        Parameters:
        -----------
        x and y: tensor of shape dim

        Returns:
        --------
        s: similarity
       """
     return normalize(x).dot(normalize(y)).pow(2)

def sqrd_euclidean_mat(X, Y):
     """
       Computes the squared euclidean distance between matrices X and Y

        Parameters:
        -----------
        X and Y: tensor of shape dim x n_samples

        Returns:
        --------
        d: squared euclidean distance
       """
     return torch.sum((X - Y).pow(2))

def similarity_mat(X, Y):
     """
       Computes the similarity (squared cosine distance)
        between matrices X and Y

        Parameters:
        -----------
        X and Y: tensor of shape dim x n_samples

        Returns:
        --------
        s: similarity
       """
     return normalize_dataset(X).t().mm(normalize_dataset(Y)).pow(2)

def separability_stats(X, labels = None):
    """
       Calculation of separability statistics for vectors:
        fisher ratio, similarity ratio and orthogonality degree

        Parameters:
        -----------
        X: tensor of shape dim x n_samples
        labels: class categories for each sample of X

        Returns:
        --------
        F:
    """

    dim, n_samples = X.shape
    n_class = labels.unique().shape[0]

    ######## Averages ###########

    class_means = torch.zeros(dim, n_class)
    for c in range(n_class):
        class_means[:,c] = X[:,labels == c].mean(1)

    global_mean = class_means.mean(1)

    ######## Fisher Ratio ###########

    euc_W = torch.zeros(n_class)
    sim_W = torch.zeros(n_class)
    
    global_mean_rep = global_mean.unsqueeze(1).expand(class_means.shape)
    euc_B = sqrd_euclidean_mat(global_mean_rep, class_means)
    sim_B = similarity_mat(global_mean_rep, class_means)

    for c in range(n_class):
        tp_X = X[:,labels == c]
        euc_W[c] = sqrd_euclidean_mat(class_means[:,c].unsqueeze(1).expand(tp_X.shape), tp_X).mean()
        sim_W[c] = similarity_mat(class_means[:,c].unsqueeze(1).expand(tp_X.shape), tp_X).mean()

    fisher_ratio = euc_B.mean() / euc_W.mean()
    sim_ratio = sim_W.mean() / sim_B.mean()
    orth_degree = 1 - sim_B.mean(1)

    return fisher_ratio, sim_ratio, orth_degree

