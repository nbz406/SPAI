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

N = 50
n_most_profitable_indices = 5
epsilon = 0.01
maxiter = 100

m = mmread("CudaRuntimeTest\sherman1.mtx")
print(m)