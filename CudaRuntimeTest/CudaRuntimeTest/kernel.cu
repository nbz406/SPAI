
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <cublasLt.h>

#include "Utils.h"

#define IDX2C(i,j,ld) (((j)*(ld))+(i))
#define iszero(val) (abs(val) < 0.00000001)

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

void sequential_spai(double* A, double* M, const int N)
{
    double* e_k = static_cast<double*>(malloc(N * N * sizeof(double)));

    // Populate M with ones in diagonal. Column major, so outer loop is over columns. Also populates e_k.
    for (int j = 0; j < N; j++)
    {
        for (int i = 0; i < N; i++)
        {
            if (i == j) {
                M[IDX2C(i, j, N)] = 1;
				e_k[IDX2C(i, j, N)] = 1;
			}
            else
            {
                M[IDX2C(i, j, N)] = 0;
                e_k[IDX2C(i, j, N)] = 0;
            }
        }
    }
    int nnz = 0;
    // for each column
    for (int k = 0; k < N; k++)
    {
        int iteration = 0;
        // each column is M from IDX2C(0, k, N) to IDX2C(0, k, N-1);
        // e_k[IDX2C(1 to N-1, k, N)]
        
        // Calculate J, an array of nonzero row indices of M
        std::vector<int> J;
        for (int i = 0; i < N; i++)
        {
            if (!iszero(M[IDX2C(i,k,N)]))
                J.emplace_back(i);
        }
        // Column major CSR can get indices from offsets array.

    }
}

int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }


    int n_rows, n_cols, nnz;
    int *csc_col_ptr_A, *csc_row_ind_A, *csc_col_ptr_M, *csc_row_ind_M;
    double* csc_val_A, *csc_val_M;

    try
    {
        const char* file_name = "../sherman1.mtx";
        Utils::read_matrix_market_file_col_major_sparse(file_name, n_rows, n_cols, nnz, csc_col_ptr_A,
            csc_val_A, csc_row_ind_A);
        printf("%s, %d x %d matrix loaded.\n", file_name, n_rows, n_cols);
    }
    catch (const std::exception&)
    {
        return EXIT_FAILURE;
    }

    Utils::create_identity_csc(n_rows, csc_col_ptr_M, csc_val_M, csc_row_ind_M);

    for (int k = 0; k < n_cols; k++)
    {
	    const int beg = csc_col_ptr_M[k]; // points to start of sub-array with column k
	    const int end = csc_col_ptr_M[k + 1]; // points to end of sub-array with column k

        // Construct J by getting all row indices from beg to end 
        int n2 = end - beg;
        int* J = static_cast<int*>(malloc(sizeof(int) * n2));
        for (int j = 0; j < n2; j++)
        {
            J[j] = csc_row_ind_M[j + beg];
        }

        // Construct I: nonzero rows of A[:,J]
        // We have to allocate n2 = reduce (+) (map (\j -> shape_A[j]) J) = reduce (+) (map (\j -> csc_col_ptr_A[j+1]-csc_col_ptr_A[j]) J)
        int n1 = 0;
    	for (int j = 0; j < n2; j++)
        {
            const int col_ind = J[j];
            n1 += csc_col_ptr_A[col_ind + 1] - csc_col_ptr_A[col_ind];
        }

        // get indices from csc_row_ind_A starting from the column pointers from csc_col_ptr_A[J]
        int* I = static_cast<int*>(malloc(sizeof(int) * n1));
        int i_ind = 0;
        for (int j = 0; j < n2; j++)
        {
            const int col_ind = J[j];
            for (int i = csc_col_ptr_A[col_ind]; i < csc_col_ptr_A[col_ind + 1]; i++)
            {
                I[i_ind] = csc_row_ind_A[i];
                i_ind++;
            }
        }

        // Construct dense A[I,J] in column major format to be used in batched QR decomposition.
        double* A_hat = static_cast<double*>(malloc(sizeof(double) * n1 * n2));
        for (int j = 0; j < n2; j++)
        {
            const int col_ind = J[j];
            const int col_start = csc_col_ptr_A[col_ind];
            const int col_end = csc_col_ptr_A[col_ind + 1];
            int* beg = &csc_row_ind_A[col_start];
            int* end = &csc_row_ind_A[col_end];
	        for (int i = 0; i < n1; i++)
	        {
                const int row_ind = I[i];
                // Need to search for row ind I[i] in csc_row_ind_A: idx and return the val at this idx.
                // Indices in csc_row_ind_A are sorted per column, so binary search works
                int* itr = std::lower_bound(beg, end, row_ind);
                const int index = std::distance(beg, itr);
                A_hat[IDX2C(i, j, n1)] = csc_val_A[col_start + index];
	        }
        }
        for (int i = 0; i < n1; i++)
        {
            for (int j = 0; j < n2; j++)
	        {
                printf(" %e ", A_hat[IDX2C(i, j, n1)]);
	        }
            printf("\n");
        }
        printf("\n");

        // Free all allocations
        free(A_hat);
        free(I);
        free(J);
    }


    free(csc_col_ptr_A); free(csc_val_A); free(csc_row_ind_A);
    free(csc_col_ptr_M); free(csc_val_M); free(csc_row_ind_M);
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
