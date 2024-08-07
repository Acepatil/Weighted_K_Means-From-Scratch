import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame

def init_centroids_kmeanspp(X, k):
    n = X.shape[0]
    centroids = np.zeros((k, X.shape[1]))
    centroids[0] = X[np.random.randint(0, n)]
    for i in range(1, k):
        distances = np.min([np.linalg.norm(X - centroids[j], axis=1)**2 for j in range(i)], axis=0)
        prob = distances / distances.sum()
        centroids[i] = X[np.random.choice(n, p=prob)]
    
    return centroids

def init_matrixes(m,n,k):
    D=np.zeros(m)
    U=np.zeros((n,k))
    W_D=np.zeros((n,k))

    return D,U,W_D

def dist(X,Z,i,j,l):
    return(X[i,j]-Z[l,j])**2

def compute_W_D_i_l(m,beta,X,Z,W,i,l):
    distm=0
    for j in range(m):
        distm+=(W[j]**beta)*dist(X,Z,i,j,l)
    return distm

def modify_W_D(n,k,m,beta,X,Z,W,W_D): 
    for i in range(n):
        for l in range(k):
            W_D[i,l]=compute_W_D_i_l(m,beta,X,Z,W,i,l)

def find_min_W_D(W_D,i,k):
    min_index=0
    min=W_D[i,0]
    for l in range(k):
        if min>W_D[i,l]:
            min=W_D[i,l]
            min_index=l
    return min_index


def modify_U(n,k,U,W_D):
    for i in range(n):
        min_dist=find_min_W_D(W_D,i,k)
        for l in range(k):
            if l!=min_dist:
                U[i,l]=0
            else:
                U[i,l]=1

def modify_Z(n,k,m,X,Z,U):
    for l in range(k):
        for j in range(m):
            num=0
            den=0
            for i in range(n):
                num+=U[i,l]*X[i,j]
                den+=U[i,l]
            if den!=0:
                Z[l,j]=num/den

def modify_D(n,k,m,X,U,Z,D):
    for j in range(m):
        D[j]=0
        for l in range(k):
            for i in range(n):
                D[j]+=U[i,l]*dist(X,Z,i,j,l)

def find_min_D(m,D):
    min_index=0
    min=D[0]
    for j in range(m):
        if min>D[j]:
            min=D[j]
            min_index=j
    return min_index

def modify_W(m,beta,W,D):
    if beta==1:
        find_mn=find_min_D(D,m)
        for j in range(m):
            if j==find_mn:
                W[j]=1
            else:
                W[j]=0
    elif beta<=0 or beta >1:
        for j in range(m):
            if D[j]==0: W[j]=0
            else:
                for j in range(m):
                    if D[j]!=0:
                        value=0
                        for i in range(m):
                            if D[i]!=0:
                                value+=(D[j]/D[i])**(1/beta-1)
                        W[j]=1/value
    else:
        raise Exception("The value of beta is not valid")

def init_weights(m):
    W=np.zeros(m)
    for i in range(m):
        W[i]=1/m
    return W


def show_clusters(X, cluster, cg):
    df = DataFrame(dict(x=X[:,0], y=X[:,1], label=cluster))
    colors = {0:'blue', 1:'orange', 2:'green', 3:'yellow', 4:'black', 5:'pink', 6:'purple'}
    fig, ax = plt.subplots(figsize=(8, 8))
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
    ax.scatter(cg[:, 0], cg[:, 1], marker='*', s=150, c='#ff2222')
    plt.xlabel('X_1')
    plt.ylabel('X_2')
    plt.show()

def K_Means(n,k,m,X,beta,max_iter):
    # Random Intialization
    Z=init_centroids_kmeanspp(X,k)
    W=init_weights(m)
    D,U,W_D=init_matrixes(m,n,k)
    # Break Point
    modify_W_D(n,k,m,beta,X,Z,W,W_D)
    modify_U(n,k,U,W_D)
    modify_D(n,k,m,X,U,Z,D)
    modify_W(m,beta,W,D)
    i=0
    while (i<max_iter):
        modify_Z(n,k,m,X,Z,U)
        modify_W_D(n,k,m,beta,X,Z,W,W_D)
        modify_U(n,k,U,W_D)
        modify_D(n,k,m,X,U,Z,D)
        modify_W(n,k,m,beta,U,Z,W_D,W,D)
        i+=1
    return Z,W,i