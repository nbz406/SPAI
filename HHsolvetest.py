import warnings
warnings.filterwarnings("ignore")

import numpy as np
import scipy
from numpy.linalg import inv
from scipy.sparse import csr_matrix
import math
import random

def householder(A):
    (n,m)=A.shape
    p=min(n,m)
    alpha=np.zeros(m)
    for j in range(0,p):
        alpha[j]=np.linalg.norm(A[j:,j])*np.sign(A[j,j])
        if (alpha[j]!=0):
            beta=1/math.sqrt(2*alpha[j]*(alpha[j]+A[j,j]))
            A[j,j]=beta*(A[j,j]+alpha[j])
            A[j+1:,j]=beta*A[j+1:,j]
            for k in range(j+1,m):
                gamma=2*np.inner(A[j:,j], A[j:,k])
                A[j:,k]=A[j:,k]-gamma*A[j:,j]
    return A,alpha

def solve_householder(H,alpha,b):
    (n,m)=H.shape
    b=b.copy()
    x=np.zeros(n)
    # b=Q^t b.
    for j in range(0,n):
        b[j:]=b[j:]-2*(H[j:,j].dot(b[j:]))*H[j:,j]
    # Auflösen von Rx=b.
    for i in range(0,n):
        j=n-1-i
        b[j]=b[j]-H[j,j+1:].dot(x[j+1:])
        x[j]=-b[j]/alpha[j]
    return x

def construct_Q(A):
    (m,n)=A.shape
    Q=np.identity(m)
    for k in range(0,n):
        v = np.matrix(A[k:,k])
        Qv = Q[:, k:] * v
        Q[:, k:] = Q[:, k:] - 2 * Qv * v.T
    return Q

def construct_R(A, alpha):
    (m,n)=A.shape
    R=np.zeros((m,n))
    for j in range(0,n):
        for i in range(0,j+1):
            if i == j:
                R[i,j] = -alpha[j]
            else:
                R[i,j]=A[i,j]
    return R

m=7
n=2
np.random.seed(n)
A=np.random.random([n,n])
A = np.matrix([[random.random() for e in range(0,n)] for e in range(0,m)])
b=np.random.random(n)
np.set_printoptions(precision=3,suppress=True)

Q,R = np.linalg.qr(A, mode="complete")
H,alpha=householder(A.copy())

print("Q: \n",Q, "\n")
print("R: \n",R, "\n")
print("H: \n",H, "\n")
print("alpha: \n",alpha, "\n")

print("construct_Q(H): \n",construct_Q(H))
print("construct_R(H,alpha): \n",construct_R(H,alpha))
print("A: \n",A)

#x=solve_householder(H,alpha,b)
#print(np.linalg.norm(A.dot(x)-b))