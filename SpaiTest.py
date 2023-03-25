import warnings
warnings.filterwarnings("ignore")

import numpy as np
import scipy
from numpy.linalg import inv
from scipy.sparse import csr_matrix
import random
import heapq
import functools

from io import StringIO
from scipy.io import mmread

N = 1000
n_most_profitable_indices = 5
epsilon = 0.5
maxiter = 10

# M is just a sparse identity matrix
M = scipy.sparse.identity(N, format='csr')
# A is a csr format of the matrix that we wish to invert
A = scipy.sparse.csr_matrix(mmread("CudaRuntimeTest\sherman1.mtx"))
AInv = scipy.sparse.linalg.inv(A)

def permutation(set, n, settilde, ntilde, mode="col"):
    setsettilde = list(zip(list(np.arange(0,n + ntilde)),list(set) + list(settilde)))
    sor = sorted(setsettilde, key=lambda x: x[1])

    swaps, rest = [[i for i, j in sor], [j for i, j in sor]]

    if mode == "col":
        P = np.identity(n + ntilde)[:,swaps]
    elif mode == "row":
        P = np.identity(n + ntilde)[swaps,:]
    return P

for k in range(0, M.shape[1]):
    print("column: ", k)
    iter = 0
    # For each column
    m_k = M[:,k]

    # Create e_k column
    e_k = np.matrix([0]*N).T
    e_k[k] = 1

    # Calculate J
    J = m_k.nonzero()[0] # gets row inds of nonzero
    n2 = J.size

    print(J)

    # Calculate A(.,J)
    A_J = A[:,J]

    # Calculate I from A(.,J)
    I = np.unique(A_J.nonzero()[0])
    n1 = I.size

    # Reduced matrix A_IJ (A hat) an n1 x n2 matrix
    A_IJ = A[np.ix_(I, J)].todense()

    # Compute ehat_k
    ehat_k = e_k[I]

    A_IJ = np.hstack((np.vstack((A_IJ, np.zeros((n1,n2)))),np.vstack((np.zeros((n1,n2)), np.zeros((n1,n2))))))

    print(A_IJ)

    # Compute QR decomp. R_1 upper triangular n1 x n1 matrix. 0 is an (n1 − n2)×n2 zero matrix. Q1 is m×n, Q2 is m×(m − n)
    Q, R = np.linalg.qr(A_IJ, mode="complete")
    R_1 = R[:n2,:n2]

    Q = Q[:n1,:n1]

    print("Q: ",Q)
    print("R: ",R_1)

    # Compute mhat_k
    chat_k = Q.T * ehat_k

    mhat_k = inv(R_1) * chat_k[0:n2,:]

    # Scatter to original column
    m_k[J] = mhat_k

    # Compute residual r
    r = A_J * mhat_k - e_k
    r_norm = np.linalg.norm(r)

    while r_norm > epsilon and iter < maxiter:
        iter += 1
        print(iter)
        # Calculate L
        L = np.nonzero(r)[0]

        # Calculate Jtilde: All of the the new column indices of A that appear in all
        # L rows but not in J
        Jtilde = np.array([],dtype=int)
        for l in L:
            A_l = A[l,:]
            NZofA_l = np.unique(A_l.nonzero()[1])
            N_l = np.setdiff1d(NZofA_l, J)
            Jtilde = np.union1d(Jtilde,N_l)

        # Calculate the new norm of the modified residual and record the indices j
        j_rho_pairs = []
        for j in Jtilde:
            e_j = np.matrix([0]*N).T
            e_j[j] = 1
            Ae_j = A * e_j
            Ae_jnorm = np.linalg.norm(Ae_j)
            rTAe_j = r.T * Ae_j
            rho_jsquared = r_norm*r_norm - (rTAe_j * rTAe_j) / (Ae_jnorm * Ae_jnorm)
            j_rho_pairs.append((rho_jsquared[0,0],j))

        # Creates min heap to quickly find indices with lowest error
        heap = []
        for pair in j_rho_pairs:
                heapq.heappush(heap, (pair[0], pair[1]))

        # Select the remaining 5 indices that create the lowest residuals
        pops = 0
        Jtilde = []
        while len(heap) > 0 and pops < n_most_profitable_indices:
            Jtilde.append(heapq.heappop(heap)[1])
            pops += 1

        # Update Q, R
        n2tilde = len(Jtilde)

        J = np.array(J)
        I = np.array(I)
        Jtilde = np.sort(Jtilde) # Needed for calculation of permutation matrices
        Itilde = np.setdiff1d(np.unique(A[:,np.union1d(Jtilde,J)].nonzero()[0]), I)
            
        n1tilde = len(Itilde)

        AIJtilde = A[np.ix_(I, Jtilde)]
        AItildeJtilde = A[np.ix_(Itilde,Jtilde)]

        # Find permutation matrices
        Pc = permutation(J, n2, Jtilde, n2tilde, mode="col")
        Pr = permutation(I, n1, Itilde, n1tilde, mode="row")
        Pc = Pc.T
        Pr = Pr.T

        Au = Q.T * AIJtilde
        B_1 = Au[:n2,:]
        B_2 = np.vstack((Au[n2:n1,:], AItildeJtilde.todense()))

        # QR decompose B2
        Q_B, R_B = np.linalg.qr(B_2, mode="complete") 

        # Matrices to use in new QR
        In1tilde = np.identity(n1tilde)
        In2 = np.identity(n2)
        n1tilden1zeros = np.zeros((n1tilde, n1))
        n1n1tildezeros = np.zeros((n1, n1tilde))

        # for the new Q, which is a matrix product
        firstmat17 = np.hstack((np.vstack((Q, n1tilden1zeros)), np.vstack((n1n1tildezeros, In1tilde))))
        secondmat17 = np.hstack((np.vstack((In2, np.zeros((n1-n2+n1tilde,n2)))), np.vstack((np.zeros((n2,n1-n2+n1tilde)), Q_B))))

        # Construct R and Q by stacking and matrix products
        R = np.hstack((np.vstack((R_1, np.zeros((n1tilde + n1 - n2, n2)))), np.vstack((B_1, R_B))))
        Q = firstmat17 * secondmat17

        # New J and I
        J = np.union1d(Jtilde,J) # J U Jtilde
        n2 = J.size
        I = np.union1d(Itilde, I) # Itilde
        n1 = I.size
        R_1 = R[0:n2,:]
            
        etilde_k = Pr * e_k[I]
        
        chat_k = Q.T * etilde_k # selects the 
        mtilde_k = inv(R_1) * chat_k[0:n2,:]

        m_k[J] = Pc * mtilde_k

        # Permute Q and R to be used in next iteration
        Q = Pr.T * Q #
        R_1 = R_1 * Pc.T
        
        # Compute residual r
        r = A[:,J] * m_k[J] - e_k
        r_norm = np.linalg.norm(r)

    # Place result column in matrix
    M[:,k] = m_k

print(np.linalg.norm(A * M - np.identity(M.shape[1])))
print(np.linalg.norm(A * AInv - np.identity(M.shape[1])))