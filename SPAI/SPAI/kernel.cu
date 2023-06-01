
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstdio>
#include "cublas_v2.h"

#include <fstream>
#include <iostream>

#include "cub/device/device_radix_sort.cuh"

#define IDX2C(i,j,ld) (((j)*(ld))+(i))
#define iszero(val) (abs(val) < 0.00000001)
#define padded true

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

double* read_matrix_market_file_col_major(const char* file_name, int& n_rows, int& n_cols)
{
    std::ifstream file(file_name);
    int num_rows, num_cols, num_lines;

    if (!file)
    {
        std::cout << "Could not read file" << std::endl;
        throw std::exception("Could not read file");
    }

    // ignore headers
    while (file.peek() == '%')
        file.ignore(2048, '\n');

    file >> num_rows >> num_cols >> num_lines;

    double* matrix = (double*)(malloc(sizeof(double) * num_rows * num_cols));
    std::fill_n(matrix, num_rows * num_cols, 0.0);

    for (int l = 0; l < num_lines; l++)
    {
        double data;
        int row, col;
        file >> row >> col >> data;
        matrix[(row - 1) + (col - 1) * num_rows] = data;
    }

    file.close();
    n_rows = num_rows;
    n_cols = num_rows;
    return matrix;
}

void read_matrix_market_file_col_major_sparse(const char* file_name, int& n_rows, int& n_cols, int& nnz, int*& csc_col_ptr_A,
    double*& csc_val_A, int*& csc_row_ind_A)
{
    std::ifstream file(file_name);

    if (!file)
    {
        std::cout << "Could not read file" << std::endl;
        throw std::exception("Could not read file");
    }

    // ignore headers
    while (file.peek() == '%')
        file.ignore(2048, '\n');

    file >> n_rows >> n_cols >> nnz;

    csc_val_A = (double*)(malloc(sizeof(double) * nnz));
    csc_col_ptr_A = (int*)(malloc(sizeof(int) * (n_cols + 1)));
    csc_row_ind_A = (int*)(malloc(sizeof(int) * nnz));

    // Construct shape array: number of non-zeros in columns:
    int prev_col = 1;
    int total_nnz = 0;
    csc_col_ptr_A[0] = 0;
    for (int l = 0; l < nnz; l++)
    {
        double data;
        int row, col;
        file >> row >> col >> data;

        csc_val_A[l] = data;
        csc_row_ind_A[l] = row - 1;

        while (col - prev_col > 0)
        {
            csc_col_ptr_A[prev_col] = total_nnz;
            prev_col++;
        }
        total_nnz++;
        prev_col = col;
    }
    csc_col_ptr_A[prev_col] = total_nnz;
}

void create_csr_from_csc(int n_rows, int n_cols, int nnz, int*& csc_col_ptr_A, double*& csc_val_A,
    int*& csc_row_ind_A, int*& csr_row_ptr_A, double*& csr_val_A, int*& csr_col_ind_A)
{
    csr_val_A = (double*)(malloc(sizeof(double) * nnz));
    csr_col_ind_A = (int*)(malloc(sizeof(int) * nnz));
    csr_row_ptr_A = (int*)(malloc(sizeof(int) * (n_rows + 1)));

    //compute number of non-zero entries per column of A 
    std::fill(csr_row_ptr_A, csr_row_ptr_A + n_rows, 0);

    for (int n = 0; n < nnz; n++) {
        csr_row_ptr_A[csc_row_ind_A[n]]++;
    }

    //cumsum the nnz per column to get csr_row_ptr_A[]
    for (int col = 0, cumsum = 0; col < n_cols; col++) {
        int temp = csr_row_ptr_A[col];
        csr_row_ptr_A[col] = cumsum;
        cumsum += temp;
    }
    csr_row_ptr_A[n_cols] = nnz;

    for (int row = 0; row < n_rows; row++) {
        for (int jj = csc_col_ptr_A[row]; jj < csc_col_ptr_A[row + 1]; jj++) {
            int col = csc_row_ind_A[jj];
            int dest = csr_row_ptr_A[col];

            csr_col_ind_A[dest] = row;
            csr_val_A[dest] = csc_val_A[jj];

            csr_row_ptr_A[col]++;
        }
    }

    for (int col = 0, last = 0; col <= n_cols; col++) {
        int temp = csr_row_ptr_A[col];
        csr_row_ptr_A[col] = last;
        last = temp;
    }
}

void create_identity_csc(int N, int*& csc_col_ptr_I, double*& csc_val_I, int*& csc_row_ind_I)
{
    csc_val_I = (double*)(malloc(sizeof(double) * N));
    csc_col_ptr_I = (int*)(malloc(sizeof(int) * (N + 1)));
    csc_row_ind_I = (int*)(malloc(sizeof(int) * N));

    for (int i = 0; i < N; i++)
    {
        csc_val_I[i] = 1.0;
        csc_col_ptr_I[i] = i;
        csc_row_ind_I[i] = i;
    }
    csc_col_ptr_I[N] = N;
}

void create_identity_plus_minus_csc(int N, int per_col, int*& csc_col_ptr_I, double*& csc_val_I, int*& csc_row_ind_I)
{
    csc_val_I = (double*)(malloc(sizeof(double) * ((N - per_col) * per_col + per_col * (per_col - 1) / 2)));
    csc_col_ptr_I = (int*)(malloc(sizeof(int) * (N + 1)));
    csc_row_ind_I = (int*)(malloc(sizeof(int) * ((N - per_col) * per_col + per_col * (per_col - 1) / 2)));

    int i;
    for (i = 0; i < N - per_col; i++)
    {
        csc_col_ptr_I[i] = i * per_col;
        for (int j = 0; j < per_col; j++)
        {
            csc_val_I[per_col * i + j] = 1;
            csc_row_ind_I[per_col * i + j] = i + j;
        }
    }

    for (int i = N - per_col, ii = 0; i < N; i++, ii += N + 1 - i)
    {
        csc_col_ptr_I[i] = per_col * (N - per_col) + ii;
        for (int j = 0; j < N - i; j++)
        {
            csc_val_I[per_col * (N - per_col) + ii + j] = 1;
            csc_row_ind_I[per_col * (N - per_col) + ii + j] = i + j;
        }
    }

    csc_col_ptr_I[N] = ((N - per_col) * per_col + per_col * (per_col - 1) / 2) + per_col;
}

int main()
{
#if padded
    cublasHandle_t handle;
    cublascall(cublasCreate(&handle));

    int n_rows, n_cols, nnz;
    int* csc_col_ptr_A, * csc_row_ind_A, * csc_col_ptr_M, * csc_row_ind_M;
    int* csr_row_ptr_A, * csr_col_ind_A;
    double* csc_val_A, * csc_val_M, * csr_val_A;

    try
    {
        const char* file_name = "../orsirr_2.mtx";
        read_matrix_market_file_col_major_sparse(file_name, n_rows, n_cols, nnz, csc_col_ptr_A,
            csc_val_A, csc_row_ind_A);
        printf("%s, %d x %d matrix loaded.\n", file_name, n_rows, n_cols);
    }
    catch (const std::exception&)
    {
        return EXIT_FAILURE;
    }
    create_csr_from_csc(n_rows, n_cols, nnz, csc_col_ptr_A, csc_val_A, csc_row_ind_A, csr_row_ptr_A, csr_val_A, csr_col_ind_A);

    create_identity_plus_minus_csc(n_rows, 3, csc_col_ptr_M, csc_val_M, csc_row_ind_M);

    int Js_size = 0;
    int Is_size = 0;
    int A_hats_size = 0;

    // for Cublas padding
    int minn1 = n_cols;
    int minn2 = n_cols;
    int maxn1 = 0;
    int maxn2 = 0;


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
        maxn2 = n2 > maxn2 ? n2 : maxn2;
        minn2 = n2 < minn2 ? n2 : minn2;
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
            Js[Js_inds[k] + j] = csc_row_ind_M[j + beg];
        }
    }

    int* buffer_inds = static_cast<int*>(malloc(sizeof(int) * n_cols));
    int total_max_I_size = 0;

    for (int k = 0; k < n_cols; k++)
    {
        // Construct I: nonzero rows of A[:,J]
        // We need a union of all row indices of columns:
        // Construct array of all columns J_0 + J_1 +... and associate each J_i with another sub-array
        // with i as elements. Stable sort both arrays with first array as keys and sort again with second array as keys.
        int size = 0;
        for (int j = 0; j < n2s[k]; j++)
        {
            const int c_ind = Js[Js_inds[k] + j];
            size += csc_col_ptr_A[c_ind + 1] - csc_col_ptr_A[c_ind];
        }

        buffer_inds[k] = total_max_I_size;
        total_max_I_size += size;
    }

    // should save pointers to rows of of A[:,J_i]'s in array.
    int* h_vals = static_cast<int*>(malloc(sizeof(int) * total_max_I_size));
    int* h_keys = static_cast<int*>(malloc(sizeof(int) * total_max_I_size));
    for (int k = 0; k < n_cols; k++)
    {
        int ind = 0;
        for (int j = 0; j < n2s[k]; j++)
        {
            const int c_ind = Js[Js_inds[k] + j];
            const int c_ptr = csc_col_ptr_A[c_ind];
            const int col_size = csc_col_ptr_A[c_ind + 1] - c_ptr;
            for (int i = 0; i < col_size; i++)
            {
                h_vals[buffer_inds[k] + ind + i] = csc_row_ind_A[c_ptr + i];
                h_keys[buffer_inds[k] + ind + i] = k;
            }
            ind += csc_col_ptr_A[c_ind + 1] - csc_col_ptr_A[c_ind];
        }
    }

    //int* d_vals_in, * d_keys_in;
    //int* d_vals_out, * d_keys_out;
    //cudaMalloc((void**)&d_vals_in, sizeof(int) * total_max_I_size);
    //cudaMalloc((void**)&d_vals_out, sizeof(int) * total_max_I_size);
    //cudaMalloc((void**)&d_keys_in, sizeof(int) * total_max_I_size);
    //cudaMalloc((void**)&d_keys_out, sizeof(int) * total_max_I_size);
    //
    //cudaMemcpy(d_vals_in, h_vals, sizeof(int) * total_max_I_size, cudaMemcpyHostToDevice);
    //cudaMemcpy(d_keys_in, h_keys, sizeof(int) * total_max_I_size, cudaMemcpyHostToDevice);
    //
    //void* d_temp_storage = NULL;
    //size_t   temp_storage_bytes = 0;
    //cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_vals_in, d_vals_out, total_max_I_size);
    //
    //cudaMalloc(&d_temp_storage, temp_storage_bytes);
    //// Run sorting operation on values
    //cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
    //    d_keys_in, d_keys_out, d_vals_in, d_vals_out, total_max_I_size);
    //
    //// Run sorting operation on keys swapping keys, pairs and ins, outs.
    //cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
    //    d_vals_out, d_vals_in, d_keys_out, d_keys_in, total_max_I_size);
    //
    //cudacall(cudaMemcpy(h_vals, d_keys_in, sizeof(int) * total_max_I_size, cudaMemcpyDeviceToHost));
    //cudacall(cudaMemcpy(h_keys, d_vals_in, sizeof(int) * total_max_I_size, cudaMemcpyDeviceToHost));
    //
    //
    //cudacall(cudaFree(d_keys_in)); cudacall(cudaFree(d_vals_in)); cudacall(cudaFree(d_keys_out));; cudacall(cudaFree(d_vals_out));
    printf("S");
    /*
    int* output_buffer = static_cast<int*>(malloc(sizeof(int) * total_max_I_size));
    int* input_buffer = static_cast<int*>(malloc(sizeof(int) * total_max_I_size));


    for (int k = 0; k < n_cols; k++)
    {
        for (int j = 0; j < n2s[k]; j++)
        {
            // get max col size
            const int col_ind = Js[Js_inds[k] + j];
            const int col_ptr = csc_col_ptr_A[col_ind];
            const int col_size = csc_col_ptr_A[col_ind + 1] - col_ptr;

        }
    }

    for (int k = 0; k < n_cols; k++)
    {
        const int b_ind = buffer_inds[k];
        const int c_1_ind = Js[Js_inds[k]];
        int col_1_ptr = csc_col_ptr_A[c_1_ind];
        int c_1_size = csc_col_ptr_A[c_1_ind + 1] - col_1_ptr;

        std::memcpy(&output_buffer[b_ind], &csc_row_ind_A[col_1_ptr], sizeof(int) * c_1_size);
        int pos = c_1_size;

        for (int j = 1; j < n2s[k]; j++)
        {
            const int c_2_ind = Js[Js_inds[k] + j];
            int col_2_ptr = csc_col_ptr_A[c_2_ind];
            const int c_2_size = csc_col_ptr_A[c_2_ind + 1] - col_2_ptr;

            std::memcpy(&input_buffer[b_ind], &output_buffer[b_ind], sizeof(int) * c_1_size);

            int pos1 = 0;
            int pos2 = 0;
            pos = 0;
            while ((pos1 < c_1_size) && (pos2 < c_2_size)) {
                uint32_t v1 = input_buffer[b_ind + pos1];
                uint32_t v2 = csc_row_ind_A[col_2_ptr + pos2];
                size_t cmp_v1_lt_v2 = v1 <= v2;
                size_t cmp_v1_gt_v2 = v1 >= v2;
                output_buffer[b_ind + pos++] = cmp_v1_lt_v2 ? v1 : v2;
                pos1 += cmp_v1_lt_v2;
                pos2 += cmp_v1_gt_v2;
            }
            while (pos1 < c_1_size) {
                output_buffer[b_ind + pos++] = input_buffer[b_ind + pos1++];
            }
            while (pos2 < c_2_size) {
                output_buffer[b_ind + pos++] = csc_row_ind_A[col_2_ptr + pos2++];
            }

            c_1_size = pos;
        }
        n1s[k] = pos;

        maxn1 = pos > maxn1 ? pos : maxn1;
        minn1 = pos < minn1 ? pos : minn1;
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

    const int A_size = maxn1 * maxn2;
    const int Q_size = maxn1 * maxn1;

    int* ind_of_col_k_in_I_k = static_cast<int*>(malloc(sizeof(int) * n_cols));
    int* Is = static_cast<int*>(malloc(sizeof(int) * Is_size));
    double* A_hats = static_cast<double*>(malloc(sizeof(double) * A_size * n_cols));
    for (int k = 0; k < n_cols; k++)
    {
        // Get indices from csc_row_ind_A starting from the column pointers from csc_col_ptr_A[J]
        // Construct dense A[I,J] in column major format to be used in batched QR decomposition.
        int A_hats_ind = A_size * k;
        for (int j = 0; j < maxn2; j++)
        {
            for (int i = 0; i < maxn1; i++)
            {
                if (i > n1s[k]-1 || j > n2s[k]-1)
                {
                    if (!(i == j))
                    {
                        A_hats[A_hats_ind + IDX2C(i, j, maxn1)] = 1; // Maybe doesn't yield same solution !!!!!!!!
                    }
                    else
                    {
                        A_hats[A_hats_ind + IDX2C(i, j, maxn1)] = 0;
                    }
                }
                else
                {
                    const int col_ind = Js[Js_inds[k] + j];
                    const int col_start = csc_col_ptr_A[col_ind];
                    const int row_ind = csc_row_ind_A[col_start + i];
                    if (row_ind == k)
                    {
                        // Keep track of index of column k in I_k
                        ind_of_col_k_in_I_k[k] = i;
                    }
                    Is[Is_inds[k] + i] = row_ind;
                    A_hats[A_hats_ind + IDX2C(i, j, maxn1)] = csc_val_A[col_start + i];
                }
            }
        }
    }

    const int batch_count = n_cols;
    const int ltau = std::max(1, std::min(maxn1, maxn2));

    double* dA_hat;     cudacall(cudaMalloc((void**)&dA_hat, batch_count * A_size * sizeof(double)));
    double* d_TAU;      cudacall(cudaMalloc((void**)&d_TAU, batch_count * ltau * sizeof(double)));
    cudacall(cudaMemcpy(dA_hat, A_hats, batch_count * A_size * sizeof(double), cudaMemcpyHostToDevice));

    double **h_A_hatArray = (double**)(malloc(batch_count * sizeof(double*)));
    double **h_TauArray = (double**)(malloc(batch_count * sizeof(double*)));

    for (int i = 0; i < batch_count; i++)
    {
        h_A_hatArray[i] = dA_hat + i * A_size;
        h_TauArray[i] = d_TAU + i * ltau;
    }

    double** d_Aarray, **d_TauArray;
    cudaMalloc((void**)&d_Aarray, batch_count * sizeof(double*));
    cudaMalloc((void**)&d_TauArray, batch_count * sizeof(double*));

    cudaMemcpy(d_Aarray, h_A_hatArray, batch_count * sizeof(double*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_TauArray, h_TauArray, batch_count * sizeof(double*), cudaMemcpyHostToDevice);

    int info;
    cublascall(cublasDgeqrfBatched(handle, maxn1, maxn2, d_Aarray, maxn1, d_TauArray, &info, batch_count));

    double* taus = static_cast<double*>(malloc(batch_count * ltau * sizeof(double)));
    cudacall(cudaMemcpy(A_hats, dA_hat, batch_count * A_size * sizeof(double), cudaMemcpyDeviceToHost));
    cudacall(cudaMemcpy(taus, d_TAU, batch_count * ltau * sizeof(double), cudaMemcpyDeviceToHost));

    // Free all allocations
    cudacall(cudaFree(dA_hat));
    cudacall(cudaFree(d_TAU));
    cudacall(cudaFree(d_Aarray));
    cudacall(cudaFree(d_TauArray));

    // Construct Q and solve upper triangular
    // If same sized arrays across columns, we can transpose and make accesses coalesced (or just use CUBLAS)

    // Q = I_n1
    double* Qs = static_cast<double*>(malloc(sizeof(double) * n_cols * Q_size));
    for (int k = 0; k < n_cols; k++) {
        for (int j = 0; j < maxn1; j++) {
            for (int i = 0; i < maxn1; i++) {
                if (i == j) {
                    Qs[k * Q_size + IDX2C(i, j, maxn1)] = 1;
                } else {
                    Qs[k * Q_size + IDX2C(i, j, maxn1)] = 0;
                }
            }
        }
    }

    for (int k = 0; k < maxn2; k++)
    {
        const int k_to_m = maxn1 - k;
        double* Qvs = static_cast<double*>(malloc(n_cols * k_to_m * sizeof(double)));
        double* vs = static_cast<double*>(malloc(n_cols * k_to_m * sizeof(double)));
        for (int kk = 0; kk < n_cols; kk++)
        {
            //Q(1 : maxn1, k : maxn1) = Q(1 : maxn1, k : maxn1) - βQ(1 : maxn1, k : maxn1)vvH
            // Q * v is maxn1 x (maxn1 - k) * (maxn1 - k) x 1
            // beta*v: v = np.matrix(A[k:,k]), beta = tau[kk * ltau + k];
            const double beta = taus[kk * ltau + k];
            vs[k_to_m * kk] = 1;
            for (int i = 1; i < k_to_m; i++)
            {
                vs[kk * k_to_m + i] = A_hats[A_size * kk + IDX2C(k + i, k, maxn1)];
            }

            for (int i = 0; i < maxn1; i++)
            {
                Qvs[k_to_m * kk + i] = 0;
            }
            for (int j = 0; j < k_to_m; j++)
            {
                for (int i = 0; i < maxn1; i++)
                {
                    Qvs[k_to_m * kk + i] += beta * Qs[kk * Q_size + IDX2C(i, j + k, maxn1)] * vs[kk * k_to_m + j];
                }
            }

            // gerBatched https://hipblas.readthedocs.io/en/latest/functions.html
            // Q = Q - QvvT
            for (int j = 0; j < k_to_m; j++)
            {
                for (int i = 0; i < maxn1; i++)
                {
                    Qs[kk * Q_size + IDX2C(i, j + k, maxn1)] -= Qvs[k_to_m * kk + i] * vs[kk * k_to_m + j];
                }
            }
        }
    }
    // chat_k: index of k in I'th row of Q transposed and first n2 elements.
    for (int k = 0; k < 20; k++)
    {
        for (int i = 0; i < maxn1; i++)
        {
            for (int j = 0; j < maxn1; j++)
            {
                printf("%.7f ", Qs[k * Q_size + IDX2C(i, j, maxn1)]);
            }
            printf("\n");
        }
    }

    double* e_hats = static_cast<double*>(malloc(sizeof(double) * n_cols * maxn1));
    for (int k = 0; k < 20; k++)
    {
        printf("ehat_%d:\n", k);
        for (int j = 0; j < maxn1; j++)
        {
            e_hats[k * maxn1 + j] = Qs[k * Q_size +  IDX2C(ind_of_col_k_in_I_k[k], j, maxn1)];
            printf("%f\n", e_hats[k * maxn1 + j]);
        }
    }
                    cudacall(cudaMalloc((void**)&dA_hat, batch_count * A_size * sizeof(double)));
    double* d_B;    cudacall(cudaMalloc((void**)&d_B, batch_count * maxn1 * sizeof(double)));
    cudacall(cudaMemcpy(dA_hat, A_hats, batch_count * A_size * sizeof(double), cudaMemcpyHostToDevice));
    cudacall(cudaMemcpy(d_B, e_hats, batch_count * maxn1 * sizeof(double), cudaMemcpyHostToDevice));

    //double** h_A_hatArray = (double**)(malloc(batch_count * sizeof(double*)));
    double** h_BArray = (double**)(malloc(batch_count * sizeof(double*)));

    for (int i = 0; i < batch_count; i++)
    {
        h_A_hatArray[i] = dA_hat + i * A_size;
        h_BArray[i] = d_B + i * maxn1;
    }

    double** d_BArray;
    cudaMalloc((void**)&d_Aarray, batch_count * sizeof(double*));
    cudaMalloc((void**)&d_BArray, batch_count * sizeof(double*));

    cudaMemcpy(d_Aarray, h_A_hatArray, batch_count * sizeof(double*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_BArray, h_BArray, batch_count * sizeof(double*), cudaMemcpyHostToDevice);

    double alpha = 1;
    cublascall(cublasDtrsmBatched(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, maxn1, maxn2, &alpha, d_Aarray, maxn1, d_BArray, maxn1, batch_count));

    double* mhat_k = static_cast<double*>(malloc(batch_count * maxn1 * sizeof(double)));
    cudacall(cudaMemcpy(A_hats, dA_hat, batch_count * A_size * sizeof(double), cudaMemcpyDeviceToHost));
    cudacall(cudaMemcpy(mhat_k, d_B, batch_count * maxn1 * sizeof(double), cudaMemcpyDeviceToHost));

    // Free all allocations
    cudacall(cudaFree(dA_hat));
    cudacall(cudaFree(d_B));
    cudacall(cudaFree(d_Aarray));
    cudacall(cudaFree(d_BArray));


    // Scatter mhat_k to J.
    // # Compute residual r. Need Indices of nonzeros.
    // rI = A_J[I] * mhat_k - e_k[I]
    //     r = np.zeros((M.shape[0], 1))
    //     r[I] = rI
    //     r_norm = np.linalg.norm(r)

    //double* rs = static_cast<double*>(malloc(sizeof(double) * maxn1 * n_cols));
    //double* r_norms = static_cast<double*>(malloc(sizeof(double) * n_cols));
    //for (int k = 0; k < n_cols; k++)
    //{
    //    r_norms[k] = 0;
    //}
    */
#else
    cublasHandle_t handle;
    cublascall(cublasCreate(&handle));

    int n_rows, n_cols, nnz;
    int* csc_col_ptr_A, * csc_row_ind_A, * csc_col_ptr_M, * csc_row_ind_M;
    int* csr_row_ptr_A, * csr_col_ind_A;
    double* csc_val_A, * csc_val_M, * csr_val_A;

    try
    {
        const char* file_name = "../orsirr_2.mtx";
        read_matrix_market_file_col_major_sparse(file_name, n_rows, n_cols, nnz, csc_col_ptr_A,
            csc_val_A, csc_row_ind_A);
        printf("%s, %d x %d matrix loaded.\n", file_name, n_rows, n_cols);
    }
    catch (const std::exception&)
    {
        return EXIT_FAILURE;
    }

    create_identity_csc(n_rows, csc_col_ptr_M, csc_val_M, csc_row_ind_M);
    create_csr_from_csc(n_rows, n_cols, nnz, csc_col_ptr_A, csc_val_A, csc_row_ind_A, csr_row_ptr_A, csr_val_A, csr_col_ind_A);

    int Js_size = 0;
    int Is_size = 0;
    int A_hats_size = 0;

    // for Cublas padding
    int minn1 = n_cols;
    int minn2 = n_cols;
    int maxn1 = 0;
    int maxn2 = 0;


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
        maxn2 = n2 > maxn2 ? n2 : maxn2;
        minn2 = n2 < minn2 ? n2 : minn2;
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
        maxn1 = n1 > maxn1 ? n1 : maxn1;
        minn1 = n1 < minn1 ? n1 : minn1;
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

    int* ind_of_col_k_in_I_k = static_cast<int*>(malloc(sizeof(int) * n_cols));
    int* Is = static_cast<int*>(malloc(sizeof(int) * Is_size));
    double* A_hats = static_cast<double*>(malloc(sizeof(double) * A_hats_size));

    for (int k = 0; k < n_cols; k++)
    {
        // Get indices from csc_row_ind_A starting from the column pointers from csc_col_ptr_A[J]
        // Construct dense A[I,J] in column major format to be used in batched QR decomposition.
        for (int j = 0; j < n2s[k]; j++)
        {
            const int col_ind = Js[Js_inds[k] + j];
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
                for (int l = j + 1; l < n2; l++)
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

    // construct Q (needed for update) and R in place (in A_hats)
#endif

    //free(A_hats_inds); free(A_hats);
    free(n1s); free(Js); free(Js_inds);
    //free(n2s); free(Is); free(Is_inds);
    free(csc_col_ptr_A); free(csc_val_A); free(csc_row_ind_A);
    free(csr_row_ptr_A); free(csr_val_A); free(csr_col_ind_A);
    free(csc_col_ptr_M); free(csc_val_M); free(csc_row_ind_M);
    cublasDestroy_v2(handle);
    return 0;
}