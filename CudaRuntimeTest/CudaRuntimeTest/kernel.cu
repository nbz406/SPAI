
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstdio>
#include "cublas_v2.h"

#include "Utils.h"

#define IDX2C(i,j,ld) (((j)*(ld))+(i))
#define iszero(val) (abs(val) < 0.00000001)

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

#define cudacall(call)                                                                                                          \
    do                                                                                                                          \
    {                                                                                                                           \
        cudaError_t err = (call);                                                                                               \
        if(cudaSuccess != err)                                                                                                  \
        {                                                                                                                       \
            fprintf(stderr,"CUDA Error:\nFile = %s\nLine = %d\nReason = %s\n", __FILE__, __LINE__, cudaGetErrorString(err));    \
            cudaDeviceReset();                                                                                                  \
            exit(EXIT_FAILURE);                                                                                                 \
        }                                                                                                                       \
    }                                                                                                                           \
    while (0)

#define cublascall(call)                                                                                        \
    do                                                                                                          \
    {                                                                                                           \
        cublasStatus_t status = (call);                                                                         \
        if(CUBLAS_STATUS_SUCCESS != status)                                                                     \
        {                                                                                                       \
            fprintf(stderr,"CUBLAS Error:\nFile = %s\nLine = %d\nCode = %d\n", __FILE__, __LINE__, status);     \
            cudaDeviceReset();                                                                                  \
            exit(EXIT_FAILURE);                                                                                 \
        }                                                                                                       \
                                                                                                                \
    }                                                                                                           \
    while(0)


cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

	//const int arraySize = 5;
    //const int a[arraySize] = { 1, 2, 3, 4, 5 };
    //const int b[arraySize] = { 10, 20, 30, 40, 50 };
    //int c[arraySize] = { 0 };
    //
    //// Add vectors in parallel.
    //cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "addWithCuda failed!");
    //    return 1;
    //}
    //
    //printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
    //    c[0], c[1], c[2], c[3], c[4]);
    //
    //// cudaDeviceReset must be called before exiting in order for profiling and
    //// tracing tools such as Nsight and Visual Profiler to show complete traces.
    //cudaStatus = cudaDeviceReset();
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaDeviceReset failed!");
    //    return 1;
    //}

int main()
{
    cublasHandle_t handle;
    cublascall(cublasCreate(&handle));

    int n_rows, n_cols, nnz;
    int *csc_col_ptr_A, *csc_row_ind_A, *csc_col_ptr_M, *csc_row_ind_M;
    double* csc_val_A, *csc_val_M;

    try
    {
        const char* file_name = "../orsirr_2.mtx";
        Utils::read_matrix_market_file_col_major_sparse(file_name, n_rows, n_cols, nnz, csc_col_ptr_A,
            csc_val_A, csc_row_ind_A);
        printf("%s, %d x %d matrix loaded.\n", file_name, n_rows, n_cols);
    }
    catch (const std::exception&)
    {
        return EXIT_FAILURE;
    }

    Utils::create_identity_csc(n_rows, csc_col_ptr_M, csc_val_M, csc_row_ind_M);

    int Js_size = 0;
    int Is_size = 0;
    int A_hats_size = 0;

    int* n2s = static_cast<int*>(malloc(sizeof(int) * n_cols));
    int* n1s = static_cast<int*>(malloc(sizeof(int) * n_cols));
    int* Js_inds = static_cast<int*>(malloc(sizeof(int) * n_cols));
    int* Is_inds = static_cast<int*>(malloc(sizeof(int) * n_cols));
    int* A_hats_inds = static_cast<int*>(malloc(sizeof(int) * n_cols));

    for (int k = 0; k < n_cols; k++)
    {
        const int beg = csc_col_ptr_M[k]; // points to start of sub-array with column k
        const int end = csc_col_ptr_M[k + 1]; // points to end of sub-array with column k

        // Construct J by getting all row indices from beg to end
        const int n2 = end - beg;
        n2s[k] = n2;
    }

    // scan n2s
    for (int k = 0; k < n_cols; k++)
    {
        Js_inds[k] = Js_size;
        Js_size += n2s[k];
    }

    int* Js = static_cast<int*>(malloc(sizeof(int) * Js_size));

    for (int k = 0; k < n_cols; k++)
    {
        const int beg = csc_col_ptr_M[k]; // points to start of sub-array with column k
        for (int j = 0; j < n2s[k]; j++)
        {
            Js[Js_inds[k]] = csc_row_ind_M[j + beg];
        }

        // Construct I: nonzero rows of A[:,J]
        // We have to allocate n2 = reduce (+) (map (\j -> shape_A[j]) J) = reduce (+) (map (\j -> csc_col_ptr_A[j+1]-csc_col_ptr_A[j]) J)
        int n1 = 0;
        for (int j = 0; j < n2s[k]; j++)
        {
            const int col_ind = Js[Js_inds[k]];
            n1 += csc_col_ptr_A[col_ind + 1] - csc_col_ptr_A[col_ind];
        }
        n1s[k] = n1;
    }

    int Qs_size = 0;
    // Scan n1s
    int* Qs_inds = static_cast<int*>(malloc(sizeof(int) * n_cols));

    for (int k = 0; k < n_cols; k++)
    {
        const int n1 = n1s[k];
        Is_inds[k] = Is_size;
        Is_size += n1;
        A_hats_inds[k] = A_hats_size;
        A_hats_size += n1 * n2s[k];
        Qs_inds[k] = Qs_size;
        Qs_size += n1 * n1;
    }


    int* ind_of_col_k_in_I_k = static_cast<int*>(malloc(sizeof(int)*n_cols));
    int* Is = static_cast<int*>(malloc(sizeof(int) * Is_size));
    double* A_hats = static_cast<double*>(malloc(sizeof(double) * A_hats_size));

    for (int k = 0; k < n_cols; k++)
    {
        // Get indices from csc_row_ind_A starting from the column pointers from csc_col_ptr_A[J]
        // Construct dense A[I,J] in column major format to be used in batched QR decomposition.
        for (int j = 0; j < n2s[k]; j++)
        {
            const int col_ind = Js[Js_inds[k]];
            const int col_start = csc_col_ptr_A[col_ind];
            const int col_end = csc_col_ptr_A[col_ind + 1];
            for (int i = col_start; i < col_end; i++)
            {
                const int row_ind = csc_row_ind_A[i];
                const int I_ind = i - col_start;
                if (row_ind == k)
                {
                    // Keep track of index of column k in I_k
                    ind_of_col_k_in_I_k[k] = I_ind;
                }
                Is[Is_inds[k] + I_ind] = row_ind;
                A_hats[A_hats_inds[k] + IDX2C(i - col_start, j, n1s[k])] = csc_val_A[i];
            }
        }
    }

    // Householder QR decomposition in place, will only need space for diagonal of Ahat: min(n1,n2)
    int* alphas_inds = static_cast<int*>(malloc(sizeof(int) * n_cols));
    int alphas_size = 0;
    for (int k = 0; k < n_cols; k++)
    {
        alphas_inds[k] = alphas_size;
        alphas_size += std::min(n1s[k], n2s[k]);
    }

    double* alphas = static_cast<double*>(malloc(sizeof(double) * alphas_size));

    for (int k = 0; k < n_cols; k++) {
        const int A_hat_ind = A_hats_inds[k];
        const int n1 = n1s[k];
        const int n2 = n2s[k];
        const int p = std::min(n1, n2);
        for (int j = 0; j < p; j++)
        {
            //alpha[j]=np.linalg.norm(A[j:,j])*np.sign(A[j,j])
            double A_col_norm_squared = 0;
            const int sign = sgn(A_hats[A_hat_ind]);
            for (int i = j; i < n1; i++)
            {
                A_col_norm_squared += pow(A_hats[A_hat_ind + IDX2C(i, j, n1)], 2);
            }
            const double alpha_j = sign * sqrt(A_col_norm_squared);
            alphas[alphas_inds[k] + j] = alpha_j;

            if (!iszero(alpha_j))
            {
                const double A_jj = A_hats[A_hat_ind + IDX2C(j, j, n1)];
                const double beta = 1 / sqrt(2 * alpha_j * (alpha_j + A_jj));
                A_hats[A_hat_ind + IDX2C(j, j, n1)] = beta * (A_jj + alpha_j);
                //A[j+1:,j]=beta*A[j+1:,j] : rest of column multiply by beta.
                for (int i = j + 1; i < n1; i++)
                {
                    A_hats[A_hat_ind + IDX2C(i, j, n1)] *= beta;
                }
                for (int l = j + 1; l < n1; l++)
                {
                    // vTA = A[j:,l].T * A[j:,j]
                    double vTA = 0;
                    for (int i = j; i < n1; i++)
                    {
                        vTA += A_hats[A_hat_ind + IDX2C(i, l, n1)] * A_hats[A_hat_ind + IDX2C(i, j, n1)];
                    }
                    for (int i = j; i < n1; i++)
                    {
                        A_hats[A_hat_ind + IDX2C(i, l, n1)] = A_hats[A_hat_ind + IDX2C(i, l, n1)] - 2 * A_hats[A_hat_ind + IDX2C(i, j, n1)] * vTA;
                    }
                }
            }
        }
    }

    // construct Q (needed for update) and R in place (in A_hats


	
    for (int k = 0; k < n_cols; k++)
    {
        const int batchsize = 1;
        const int ltau = std::max(1, std::min(n1s[k], n2s[k]));

        double* dA_hat;     cudacall(cudaMalloc((void**)&dA_hat, batchsize * n1s[k] * n2s[k] * sizeof(*dA_hat)));
        cudacall(cudaMemcpy(dA_hat, &A_hats[A_hats_inds[k]], n1s[k] * n2s[k] * sizeof(dA_hat), cudaMemcpyHostToDevice));

        double* d_TAU;      cudacall(cudaMalloc((void**)&d_TAU, batchsize * ltau * sizeof(double)));

        double* h_A_hatArray[batchsize], * h_TauArray[batchsize];

        for (int i = 0; i < batchsize; i++)
        {
            h_A_hatArray[i] = dA_hat + i * n1s[k] * n2s[k];
            h_TauArray[i] = d_TAU + i * ltau;
        }

        double** d_Aarray, ** d_TauArray;
        cudaMalloc((void**)&d_Aarray, sizeof(h_A_hatArray));
        cudaMalloc((void**)&d_TauArray, sizeof(h_TauArray));

        cudaMemcpy(d_Aarray, h_A_hatArray, sizeof(h_A_hatArray), cudaMemcpyHostToDevice);
        cudaMemcpy(d_TauArray, h_TauArray, sizeof(h_TauArray), cudaMemcpyHostToDevice);
        int info;
        cublascall(cublasDgeqrfBatched(handle, n1s[k], n2s[k], d_Aarray, n1s[k], d_TauArray, &info, batchsize));

        double* tau = static_cast<double*>(malloc(sizeof(double) * ltau));
        cudacall(cudaMemcpy(&A_hats[A_hats_inds[k]], dA_hat, batchsize * n1s[k] * n2s[k] * sizeof(double), cudaMemcpyDeviceToHost));
        cudacall(cudaMemcpy(tau, d_TAU, batchsize * ltau * sizeof(double), cudaMemcpyDeviceToHost));



        // Free all allocations
        cudacall(cudaFree(dA_hat));
        cudacall(cudaFree(d_TAU));
        cudacall(cudaFree(d_Aarray));
        cudacall(cudaFree(d_TauArray));
    }

    free(A_hats_inds); free(A_hats);
    free(n1s); free(Js); free(Js_inds);
    free(n2s); free(Is); free(Is_inds);
    free(csc_col_ptr_A); free(csc_val_A); free(csc_row_ind_A);
    free(csc_col_ptr_M); free(csc_val_M); free(csc_row_ind_M);
    cublasDestroy_v2(handle);
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
