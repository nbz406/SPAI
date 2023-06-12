import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

import fast_matrix_market
import numpy as np
import scipy
from numpy.linalg import inv
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
import random
import heapq
import functools
import math

from io import StringIO

from typing import Union

np.set_printoptions(precision=20,suppress=True)

def solveUpperTriangularMatrix(R, b):
    # The solution will be here
    for step in range(len(b) - 1, 0 - 1, -1):
        if R[step,step] == 0:
            if b[step] != 0:
                return "No solution"
            else:
                return "Infinity solutions"
        else:
            b[step] = b[step] / R[step,step]

        for row in range(step - 1, 0 - 1, -1):
            b[row] -= R[row,step] * b[step]
    return b

def householder(A):
    (n1,n2)=A.shape
    p=min(n1,n2)
    alpha=np.zeros(n2)
    for j in range(0,p):
        alpha[j]=np.linalg.norm(A[j:,j])*np.sign(A[j,j])
        if (alpha[j]!=0):
            beta=1/math.sqrt(2*alpha[j]*(alpha[j]+A[j,j]))
            A[j,j]=beta*(A[j,j]+alpha[j])
            A[j+1:,j]=beta*A[j+1:,j]
            for k in range(j+1,n2):
                vTA = A[j:,k].T * A[j:,j]
                A[j:,k]=A[j:,k]-2*A[j:,j]*vTA
    return A,alpha

def construct_Q(A):
    (m,n)=A.shape
    Q=np.identity(m)
    for k in range(0,n):
        v = np.matrix(A[k:,k])
        Qv = Q[:, k:] * v
        Q[:, k:] = Q[:, k:] - 2 * Qv * v.T
    return Q

def apply_QT(A, X):
    (m,n)=A.shape
    for k in range(0,n):
        v = np.matrix(A[k:,k])
        vTX = v.T * X[k:, k:]
        X[k:, k:] = X[k:, k:] - 2 * v * vTX
    return X

def apply_QT(A, X):
    (m,n)=A.shape
    QTX = X
    for k in range(0,n):
        v = np.matrix(A[k:,k])
        vTX = v.T * QTX[k:, :]
        X[k:, :] = QTX[k:, :] - 2 * v * vTX
    return X

def apply_Q(A, X):
    (m,n)=A.shape
    Q=X
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

def construct_R_1(A, alpha):
    (m,n)=A.shape
    R=np.zeros((n,n))
    for j in range(0,n):
        for i in range(0,j+1):
            if i == j:
                R[i,j] = -alpha[j]
            else:
                R[i,j]=A[i,j]
    return R

def permutation(set, n, settilde, ntilde, mode="col"):
    setsettilde = list(zip(list(np.arange(0,n + ntilde)),list(set) + list(settilde)))
    sor = sorted(setsettilde, key=lambda x: x[1])

    swaps, sortedset = [[i for i, j in sor], [j for i, j in sor]]
    return swaps, sortedset

n_most_profitable_indices = 5 # bounded from 3 to 8
epsilon = 0.000000000000001 # 0.1 to 0.5 according to https://mediatum.ub.tum.de/doc/1107998/426923.pdf#page=64&zoom=100,117,534
maxiter = 10 # 1 to 5

matname = "orsirr_2"

# M is just a sparse identity matrix
# A is a csc format of the matrix that we wish to invert
A = scipy.sparse.csc_matrix(fast_matrix_market.mmread("CudaRuntimeTest\\"+matname+".mtx"))

N = A.shape[0]
M = scipy.sparse.identity(N, format='csc')
#M[1,0] = 1
#M[2,0] = 1
#
#M[2,1] = 1
#M[3,1] = 1

#for j in range(0, M.shape[0]):
#    for i in range(0, min(M.shape[0] - j, 3)):
#        M[i+j,j] = 1
B = A * M - np.identity(M.shape[1])

Bnormbefore = np.linalg.norm(B)
Mnz = scipy.sparse.csr_matrix.count_nonzero(M)
Anz = scipy.sparse.csr_matrix.count_nonzero(A)
ratiobefore = Mnz / Anz

#AInv = scipy.sparse.linalg.inv(A)
np.set_printoptions(precision=10, linewidth=800)
maxn1 = 0
maxn2 = 0
minn1 = maxiter*[M.shape[1]]
minn2 = maxiter*[M.shape[1]]

iterations = M.shape[1]*[0]

from joblib import Parallel, delayed
def process(k):
    print(k)
    iter = 0
    # For each column
    m_k = M[:,k]

    # Create e_k column
    e_k = np.matrix([0]*N).T
    e_k[k] = 1

    # Calculate J
    J = m_k.nonzero()[0] # gets row inds of nonzero
    n2 = J.size

    # Calculate A(.,J)
    A_J = A[:,J]

    # Calculate I from A(.,J)
    I = np.unique(A_J.nonzero()[0])
    n1 = I.size

    # Reduced matrix A_IJ (A hat) an n1 x n2 matrix
    A_IJ = A[np.ix_(I, J)].todense()
    
    Q, R = np.linalg.qr(A_IJ, mode="complete")
    R = R[:n2,:n2]

    # Compute mhat_k
    try:
        chat_k = np.matrix(Q.T[:,list(I).index(k)])
    except:
        chat_k = np.zeros(n2)
    mhat_k = scipy.linalg.solve_triangular(R, chat_k[0:n2])

    # Compute residual r
    rI = A_IJ * mhat_k - e_k[I]
    r = np.zeros((M.shape[0],1))
    r[I] = rI
    try:
        list(I).index(k)
    except:
        r[k] = -1
    r_norm = np.linalg.norm(r)
    
    while iter < maxiter:
        iter += 1
        print("iter: ",iter)
        # Calculate L
        L = np.union1d(I,k) #np.nonzero(r)[0] # Lk ← Ik ∪ {k} ? from paper https://mediatum.ub.tum.de/doc/1107998/426923.pdf#page=64&zoom=100,117,534

        # Calculate Jtilde: All of the the new column indices of A that appear in all
        # L rows but not in J
        Jtilde = np.array([],dtype=int)
        for l in L:
            A_l = A[l,:]
            NZofA_l = np.unique(A_l.nonzero()[1])
            N_l = np.setdiff1d(NZofA_l, J)
            Jtilde = np.union1d(Jtilde,N_l)

        # Calculate the new norm of the modified residual and record the indices j
        avg_rho = 0
        j_rho_pairs = []
        for j in Jtilde:
            Ae_j = A[:,j].todense()
            Ae_jnorm = np.linalg.norm(Ae_j)
            rTAe_j = r.T * Ae_j ## if A[:,j] (row_inds[col_ptr[j]]) overlap with nonzeros of r, then mult, otherwise don't
            rho_jsquared = r_norm*r_norm - (rTAe_j * rTAe_j) / (Ae_jnorm * Ae_jnorm)
            avg_rho += rho_jsquared
            j_rho_pairs.append((rho_jsquared[0,0],j))

        avg_rho = avg_rho / len(j_rho_pairs)

        # Creates min heap to quickly find indices with lowest error.
        heap = []
        for pair in j_rho_pairs:
                heapq.heappush(heap, (pair[0], pair[1]))

        # Select the remaining 5 indices that create the lowest residuals
        pops = 0
        Jtilde = []
        while len(heap) > 0 and pops < n_most_profitable_indices:
            pair = heapq.heappop(heap)
            if (pair[0] < avg_rho):
                Jtilde.append(pair[1])
                pops += 1

        # Update Q, R
        n2tilde = len(Jtilde)
        Jtilde = np.sort(Jtilde) # Needed for calculation of permutation matrices

        Itilde = np.setdiff1d(np.unique(A[:,Jtilde].nonzero()[0]), I)
        n1tilde = len(Itilde)

        AIJtilde = A[np.ix_(I, Jtilde)]
        AItildeJtilde = A[np.ix_(Itilde,Jtilde)]

        Au = Q.T * AIJtilde
        B_1 = Au[:n2,:]
        B_2 = np.vstack((Au[n2:n1,:], AItildeJtilde.todense()))

        QB, RB = np.linalg.qr(B_2, mode="complete")
        RB = RB[:n2tilde,:n2tilde]

        # Construct R and Q by stacking and matrix products
        R = np.hstack((np.vstack((R, np.zeros((n2tilde, n2)))), np.vstack((B_1, RB)))) # don't need entire R
        q = np.hstack((np.vstack((Q[:,n2:], np.zeros((n1tilde,n1-n2)))), np.vstack((np.zeros((n1,n1tilde)), np.identity(n1tilde)))))
        Q = np.hstack((np.vstack((Q[:,:n2], np.zeros((n1tilde,n2)))), q * QB))

        # New J and I
        J = np.append(J,Jtilde) # J U Jtilde
        n2 = J.size
        I = np.append(I,Itilde) # Itilde
        n1 = I.size

        try:
            chat_k = np.matrix(Q.T[:,list(I).index(k)])
        except:
            chat_k = np.zeros(n2)
        mhat_k = scipy.linalg.solve_triangular(R, chat_k[0:n2])

        rI = A[np.ix_(I,J)] * mhat_k - e_k[I]
        r[I] = rI

        try:
            list(I).index(k)
        except:
            r[k] = -1

        r_norm = np.linalg.norm(rI)

        print("norm:", r_norm)

    if iter <= maxiter:
        iter = iter
    else:
        iter = maxiter+1

    m_k[J] = mhat_k
    # Place result column in matrix
    return m_k, iter
    
results = Parallel(n_jobs=16)(delayed(process)(i) for i in range(0,M.shape[0]))

for k in range(0,M.shape[0]):
    M[:,k] = results[k][0]
    iterations[k] = results[k][1]

#for k in range(0,M.shape[0]):
#    print("column: ", k)
#    iter = 0
#    # For each column
#    m_k = M[:,k]
#
#    # Create e_k column
#    e_k = np.matrix([0]*N).T
#    e_k[k] = 1
#
#    # Calculate J
#    J = m_k.nonzero()[0] # gets row inds of nonzero
#    n2 = J.size
#
#    # Calculate A(.,J)
#    A_J = A[:,J]
#
#    # Calculate I from A(.,J)
#    I = np.unique(A_J.nonzero()[0])
#    n1 = I.size
#
#    # Reduced matrix A_IJ (A hat) an n1 x n2 matrix
#    A_IJ = A[np.ix_(I, J)].todense()
#    
#    Q, R = np.linalg.qr(A_IJ, mode="complete")
#    R = R[:n2,:n2]
#
#    # Compute mhat_k
#    try:
#        chat_k = np.matrix(Q.T[:,list(I).index(k)])
#    except:
#        chat_k = np.zeros(n2)
#    mhat_k = scipy.linalg.solve_triangular(R, chat_k[0:n2])
#
#    # Compute residual r
#    rI = A_IJ * mhat_k - e_k[I]
#    r = np.zeros((M.shape[0],1))
#    r[I] = rI
#    try:
#        list(I).index(k)
#    except:
#        r[k] = -1
#    r_norm = np.linalg.norm(r)
#    
#    while r_norm > epsilon and iter < maxiter:
#        iter += 1
#        print("iter: ",iter)
#        # Calculate L
#        L = np.union1d(I,k) #np.nonzero(r)[0] # Lk ← Ik ∪ {k} ? from paper https://mediatum.ub.tum.de/doc/1107998/426923.pdf#page=64&zoom=100,117,534
#
#        # Calculate Jtilde: All of the the new column indices of A that appear in all
#        # L rows but not in J
#        Jtilde = np.array([],dtype=int)
#        for l in L:
#            A_l = A[l,:]
#            NZofA_l = np.unique(A_l.nonzero()[1])
#            N_l = np.setdiff1d(NZofA_l, J)
#            Jtilde = np.union1d(Jtilde,N_l)
#
#        # Calculate the new norm of the modified residual and record the indices j
#        # Could get A_J and each column is a j
#        avg_rho = 0
#        j_rho_pairs = []
#        for j in Jtilde:
#            Ae_j = A[:,j].todense()
#            Ae_jnorm = np.linalg.norm(Ae_j)
#            rTAe_j = r.T * Ae_j ## if A[:,j] (row_inds[col_ptr[j]]) overlap with nonzeros of r, then mult, otherwise don't
#            rho_jsquared = r_norm*r_norm - (rTAe_j * rTAe_j) / (Ae_jnorm * Ae_jnorm)
#            avg_rho += rho_jsquared
#            j_rho_pairs.append((rho_jsquared[0,0],j))
#
#        avg_rho = avg_rho / len(j_rho_pairs)
#
#        # Creates min heap to quickly find indices with lowest error.
#        heap = []
#        for pair in j_rho_pairs:
#                heapq.heappush(heap, (pair[0], pair[1]))
#
#        # Select the remaining 5 indices that create the lowest residuals
#        pops = 0
#        Jtilde = []
#        while len(heap) > 0 and pops < n_most_profitable_indices:
#            pair = heapq.heappop(heap)
#            if (pair[0] < avg_rho):
#                Jtilde.append(pair[1])
#                pops += 1
#
#        # Update Q, R
#        n2tilde = len(Jtilde)
#        Jtilde = np.sort(Jtilde) # Needed for calculation of permutation matrices
#
#        Itilde = np.setdiff1d(np.unique(A[:,Jtilde].nonzero()[0]), I)
#        n1tilde = len(Itilde)
#
#        AIJtilde = A[np.ix_(I, Jtilde)]
#        AItildeJtilde = A[np.ix_(Itilde,Jtilde)]
#
#        Au = Q.T * AIJtilde
#        B_1 = Au[:n2,:]
#        B_2 = np.vstack((Au[n2:n1,:], AItildeJtilde.todense()))
#
#        QB, RB = np.linalg.qr(B_2, mode="complete")
#        RB = RB[:n2tilde,:n2tilde]
#
#        # Construct R and Q by stacking and matrix products
#        R = np.hstack((np.vstack((R, np.zeros((n2tilde, n2)))), np.vstack((B_1, RB)))) # don't need entire R
#        q = np.hstack((np.vstack((Q[:,n2:], np.zeros((n1tilde,n1-n2)))), np.vstack((np.zeros((n1,n1tilde)), np.identity(n1tilde)))))
#        Q = np.hstack((np.vstack((Q[:,:n2], np.zeros((n1tilde,n2)))), q * QB))
#
#        # New J and I
#        J = np.append(J,Jtilde) # J U Jtilde
#        n2 = J.size
#        I = np.append(I,Itilde) # Itilde
#        n1 = I.size
#
#        try:
#            chat_k = np.matrix(Q.T[:,list(I).index(k)])
#        except:
#            chat_k = np.zeros(n2)
#        mhat_k = scipy.linalg.solve_triangular(R, chat_k[0:n2])
#
#        rI = A[np.ix_(I,J)] * mhat_k - e_k[I]
#        r[I] = rI
#
#        try:
#            list(I).index(k)
#        except:
#            r[k] = -1
#
#        r_norm = np.linalg.norm(rI)
#
#        print("norm:", r_norm)
#
#    if iter <= maxiter:
#        iterations[k] = iter
#    else:
#        iterations[k] = maxiter+1
#
#    m_k[J] = mhat_k
#    # Place result column in matrix
#    M[:,k] = m_k
#
#
B = A * M - np.identity(M.shape[1])
normy = np.linalg.norm(B)
print("norm before:", Bnormbefore)
print("norm after:", normy)

Mnz = scipy.sparse.csr_matrix.count_nonzero(M)
Anz = scipy.sparse.csr_matrix.count_nonzero(A)
print("ratio nnz(M)/nnz(A): ", Mnz/Anz)
print("ratio before ", ratiobefore)

text = f"nnz(M) / nnz(A) = {Mnz} / {Anz} = {Mnz/Anz:.2f}. Maxiter = {maxiter}, ε = {epsilon}, ||AM − I|| = {normy:.2f}"

plt.figure(figsize=(15, 10))
values, bins, bars = plt.hist(iterations, edgecolor='white', bins=range(0, maxiter + 2, 1))
bars[maxiter].set_facecolor('red')
plt.xlabel("Iterations")
plt.ylabel("Number of columns")
plt.title = ('Iterations before covergence')
plt.bar_label(bars, fontsize=20, color='navy')
plt.margins(x=0.01, y=0.1)
plt.figtext(.5, 0.02, text, ha='center', size='x-large')
plt.savefig("Histogram_"+matname)

#print(np.linalg.norm(A * AInv - np.identity(M.shape[1])))
