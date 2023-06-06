
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cub/device/device_segmented_radix_sort.cuh"
#include "cub/device/device_segmented_reduce.cuh"
#include "cub/device/device_radix_sort.cuh"
#include "cub/device/device_scan.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#include "cublas_v2.h"

#include <fstream>
#include <iostream>
#include <queue>
#include <vector>

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


double rand_from(double min, double max)
{
    const double range = (max - min);
    const double div = RAND_MAX / range;
    return min + (rand() / div);
}

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
    csc_val_I = (double*)(malloc(sizeof(double) * ((N - per_col) * per_col + per_col * (per_col - 1) / 2 + per_col)));
    csc_col_ptr_I = (int*)(malloc(sizeof(int) * (N + 1)));
    csc_row_ind_I = (int*)(malloc(sizeof(int) * ((N - per_col) * per_col + per_col * (per_col - 1) / 2 + per_col)));

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

template <typename T1, typename T2>
struct MinHeap {
    cuda::std::pair<T1, T2>* arr;
    // Current Size of the Heap
    int size;
    // Maximum capacity of the heap
    int capacity;
};

__host__ __device__ int parent(int i) {
    // Get the index of the parent
    return (i - 1) / 2;
}

__host__ __device__ int left_child(int i) {
    return (2 * i + 1);
}

__host__ __device__ int right_child(int i) {
    return (2 * i + 2);
}

template <typename T1, typename T2>
__host__ __device__ cuda::std::pair<T1, T2> get_min(MinHeap<T1,T2>* heap) {
    // Return the root node element,
    // since that's the minimum
    return heap->arr[0];
}

template <typename T1, typename T2>
__host__ __device__ MinHeap<T1,T2>* init_minheap(int capacity) {
    MinHeap<T1,T2>* minheap = (MinHeap<T1, T2>*)calloc(1, sizeof(MinHeap<T1, T2>));
    minheap->arr = (cuda::std::pair<T1, T2>*)calloc(capacity, sizeof(cuda::std::pair<T1, T2>));
    minheap->capacity = capacity;
    minheap->size = 0;
    return minheap;
}
template <typename T1, typename T2>
__host__ __device__ MinHeap<T1,T2>* insert_minheap(MinHeap<T1,T2>* heap, T1 element, T2 extra) {
    // Inserts an element to the min heap
    // We first add it to the bottom (last level)
    // of the tree, and keep swapping with it's parent
    // if it is lesser than it. We keep doing that until
    // we reach the root node. So, we will have inserted the
    // element in it's proper position to preserve the min heap property
    if (heap->size == heap->capacity) {
        fprintf(stderr, "Cannot insert %d. Heap is already full!\n", element);
        return heap;
    }
    // We can add it. Increase the size and add it to the end
    heap->size++;
    heap->arr[heap->size - 1] = cuda::std::pair<T1, T2>(element, extra);

    // Keep swapping until we reach the root
    int curr = heap->size - 1;
    // As long as you aren't in the root node, and while the 
    // parent of the last element is greater than it
    while (curr > 0 && heap->arr[parent(curr)].first > heap->arr[curr].first) {
        // Swap
        cuda::std::pair<T1, T2> temp = heap->arr[parent(curr)];
        heap->arr[parent(curr)] = heap->arr[curr];
        heap->arr[curr] = temp;
        // Update the current index of element
        curr = parent(curr);
    }
    return heap;
}

template <typename T1, typename T2>
__host__ __device__ MinHeap<T1,T2>* heapify(MinHeap<T1,T2>* heap, int index) {
    // Rearranges the heap as to maintain
    // the min-heap property
    if (heap->size <= 1)
        return heap;

    int left = left_child(index);
    int right = right_child(index);

    // Variable to get the smallest element of the subtree
    // of an element an index
    int smallest = index;

    // If the left child is smaller than this element, it is
    // the smallest
    if (left < heap->size && heap->arr[left].first < heap->arr[index].first)
        smallest = left;

    // Similarly for the right, but we are updating the smallest element
    // so that it will definitely give the least element of the subtree
    if (right < heap->size && heap->arr[right].first < heap->arr[smallest].first)
        smallest = right;

    // Now if the current element is not the smallest,
    // swap with the current element. The min heap property
    // is now satisfied for this subtree. We now need to
    // recursively keep doing this until we reach the root node,
    // the point at which there will be no change!
    if (smallest != index)
    {
        cuda::std::pair<T1, T2> temp = heap->arr[index];
        heap->arr[index] = heap->arr[smallest];
        heap->arr[smallest] = temp;
        heap = heapify(heap, smallest);
    }

    return heap;
}

template <typename T1, typename T2>
__host__ __device__ MinHeap<T1,T2>* delete_minimum(MinHeap<T1,T2>* heap) {
    // Deletes the minimum element, at the root
    if (!heap || heap->size == 0)
        return heap;

    int size = heap->size;
    cuda::std::pair<T1, T2> last_element = heap->arr[size - 1];

    // Update root value with the last element
    heap->arr[0] = last_element;

    // Now remove the last element, by decreasing the size
    heap->size--;

    // We need to call heapify(), to maintain the min-heap
    // property
    heap = heapify(heap, 0);
    return heap;
}

template <typename T1, typename T2>
__host__ __device__ MinHeap<T1,T2>* delete_element(MinHeap<T1,T2>* heap, int index) {
    // Deletes an element, indexed by index
    // Ensure that it's lesser than the current root
    cuda::std::pair<T1, T2> min = get_min(heap);
    heap->arr[index] = cuda::std::pair<T1, T2>(min.first - 1, min.second);

    // Now keep swapping, until we update the tree
    int curr = index;
    while (curr > 0 && heap->arr[parent(curr)].first > heap->arr[curr].first) {
        cuda::std::pair<T1, T2> temp = heap->arr[parent(curr)];
        heap->arr[parent(curr)] = heap->arr[curr];
        heap->arr[curr] = temp;
        curr = parent(curr);
    }

    // Now simply delete the minimum element
    heap = delete_minimum(heap);
    return heap;
}

template <typename T1, typename T2>
__host__ __device__ void print_heap(MinHeap<T1,T2>* heap) {
    // Simply print the array. This is an
    // inorder traversal of the tree
    printf("Min Heap:\n");
    for (int i = 0; i < heap->size; i++) {
        printf("(%d, %d) -> ", heap->arr[i].first, heap->arr[i].second);
    }
    printf("\n");
}

template <typename T1, typename T2>
__host__ __device__ void free_minheap(MinHeap<T1,T2>* heap) {
    if (!heap)
        return;
    free(heap->arr);
    free(heap);
}



__global__ void set_vs_initial(double* vs, int k_to_m, double* Ahats, int A_size, int* n2s, int maxn1, int k, int n)
{
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int kk = tid / k_to_m;
    const unsigned int i = tid % k_to_m;
    if (tid < n)
    {
        if (n2s[kk] - 1 < k)
            vs[k_to_m * kk + i] = 0;
        else
        {
            if (i == 0)
                vs[k_to_m * kk + i] = 1;
            else
                vs[kk * k_to_m + i] = Ahats[A_size * kk + IDX2C(k + i, k, maxn1)];
        }
    }
}
//if (k < n2tildes[kk])
//    {
//        vs->array[k_to_m * kk] = 1;
//        for (int i = 1; i < k_to_m; i++)
//            vs->array[kk * k_to_m + i] = Anews->array[Anew_size * kk + n2s[kk] * maxnewn1 + n2s[kk] + IDX2C(k + i, k, maxnewn1)];
//    } else
//    {
//        for (int i = 0; i < k_to_m; i++)
//            vs->array[kk * k_to_m + i] = 0;
//    }
//set_vs_loop << <gridSize, blockSize >> > (d_vs->array, k_to_m, d_Anews->array, Anew_size, d_n2s, d_n2tildes, maxnewn1, k, n_cols* k_to_m);
__global__ void set_vs_loop(double* vs, int k_to_m, double* Ahats, int A_size, int* n2s, int* n2tildes, int maxnewn1, int k, int n)
{
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int kk = tid / k_to_m;
    const unsigned int i = tid % k_to_m;
    if (tid < n)
    {
        if (n2tildes[kk] - 1 < k)
            vs[k_to_m * kk + i] = 0;
        else
        {
            if (i == 0)
                vs[k_to_m * kk + i] = 1;
            else
                vs[kk * k_to_m + i] = Ahats[A_size * kk + n2s[kk] * maxnewn1 + n2s[kk] + IDX2C(k + i, k, maxnewn1)];
        }
    }
}

__global__ void set_ptr_array(double** Array, double* A, int size, int col_offset, int row_offset, int n_rows, int n)
{
    const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n)
    {
        Array[idx] = A + idx * size + col_offset * n_rows + row_offset;
    }
}

__global__ void set_ptr_array_with_irregular_column_offsets(double** Array, double* A, int size, int* col_offsets, int col_offset, int row_offset, int n_rows, int n)
{
    const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n)
    {
        Array[idx] = A + idx * size + (col_offsets[idx] + col_offset) * n_rows + row_offset;
    }
}

__global__ void set_ptr_array_with_irregular_column_and_row_offsets(double** Array, double* A, int size, int* col_offsets, int* row_offsets, int n_rows, int n)
{
    const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n)
    {
        Array[idx] = A + idx * size + col_offsets[idx] * n_rows + row_offsets[idx];
    }
}

__global__ void mult_taus(double* arr, double* taus, int ltau, int k, int maxn1, int n)
{
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int kk = tid / maxn1;
    const unsigned int i = tid % maxn1;

    if (kk == 885)
    {
        int x = 1;
    }

    if (tid < n)
    {
        const double v = arr[kk * maxn1 + i];
        const double tau = taus[kk * ltau + k];
    	arr[kk * maxn1 + i] = v * tau;
    }
}

__global__ void init_rs(double* rs, int* id_of_ks, int maxn1, int n)
{
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int k = tid / maxn1;
    const unsigned int i = tid % maxn1;
    if (tid < n)
    {
        if (id_of_ks[k] == i)
            rs[maxn1 * k + i] = 1;
        else
            rs[maxn1 * k + i] = 0;
    }
}

__global__ void set_val_at_offset(int* arr, int* offsets, int n)
{
    const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n)
    {
        arr[offsets[idx]] = 1;
    }
}

template <typename T>
struct dyn_arr // array that can increase in size
{
    int capacity;
    T* array;
};

template <typename T>
__host__ dyn_arr<T>* init_dyn_arr(int capacity)
{
    dyn_arr<T>* arr = (dyn_arr<T>*)malloc(sizeof(dyn_arr<T>));
    arr->array = (T*)malloc(sizeof(T) * capacity);
    arr->capacity = capacity;
    return arr;
}

template <typename T>
__host__ void resize_based_on_write_size(dyn_arr<T>* arr, const unsigned int n_elems)
{
    while (arr->capacity < n_elems)
        arr->capacity *= 2;

    T* temp = (T*)malloc(sizeof(T) * arr->capacity);
    free(arr->array);
    arr->array = temp;
}

template <typename T>
__host__ void free_dyn_arr(dyn_arr<T>* arr)
{
    free(arr->array);
    free(arr);
}

template <typename T>
struct cuda_dyn_arr // array that can increase in size
{
    int capacity;
    T* array;
};

template <typename T>
__host__ cuda_dyn_arr<T>* init_cuda_dyn_arr(int capacity)
{
    cuda_dyn_arr<T>* arr = (cuda_dyn_arr<T>*)malloc(sizeof(cuda_dyn_arr<T>));
    cudacall(cudaMalloc((void**)&arr->array, sizeof(T) * capacity));
    arr->capacity = capacity;
    return arr;
}

template <typename T>
__host__ void cuda_resize_based_on_write_size(cuda_dyn_arr<T>* arr, const unsigned int n_elems)
{
    while (arr->capacity < n_elems)
        arr->capacity *= 2;

	T* temp; cudacall(cudaMalloc((void**)&temp, sizeof(T) * arr->capacity));
    cudacall(cudaFree(arr->array));
    arr->array = temp;
}

template <typename T>
__host__ void free_cuda_dyn_arr(cuda_dyn_arr<T>* arr)
{
    cudacall(cudaFree(arr->array));
    free(arr);
}

__host__ void construct_Ahats(
    int n_cols,
    double* Ahats,
    int A_size,
    int* Js,
    int* n2s,
    int J_size,
    int* Is,
    int* n1s,
    int I_size,
    int maxn1,
    int maxn2,
    int* csc_col_ptr_A,
    int* csc_row_ind_A,
    double* csc_val_A)
{
    for (int k = 0; k < n_cols; k++)
    {
        // Get indices from csc_row_ind_A starting from the column pointers from csc_col_ptr_A[J]
        // Construct dense A[I,J] in column major format to be used in batched QR decomposition.
        int A_hats_ind = A_size * k;
        for (int j = 0; j < maxn2; j++)
        {
            const int col_ind = Js[J_size * k + j];
            const int col_start = csc_col_ptr_A[col_ind];
            const int col_size = csc_col_ptr_A[col_ind + 1] - col_start;
            for (int i = 0; i < maxn1; i++)
            {
                if (i > n1s[k] - 1 || j > n2s[k] - 1)
                {
                    if (i == j)
						Ahats[A_hats_ind + IDX2C(i, j, maxn1)] = 1;
                    else 
                        Ahats[A_hats_ind + IDX2C(i, j, maxn1)] = 0;
                }
                else
                {
                    int col_elem_idx = 0;
                    const int i_ind = Is[I_size * k + i];
                    while (col_elem_idx < col_size && csc_row_ind_A[col_start + col_elem_idx] < i_ind) // Linear search could be optimized to binary search
                        col_elem_idx++;
                    if (i_ind == csc_row_ind_A[col_start + col_elem_idx])
                    {
                        Ahats[A_hats_ind + IDX2C(i, j, maxn1)] = csc_val_A[col_start + col_elem_idx];
                    }
                    else
                    {
                        Ahats[A_hats_ind + IDX2C(i, j, maxn1)] = 0;
                    }
                }
            }
        }
    }
}

int main()
{
    cublasHandle_t handle;
    cublascall(cublasCreate(&handle));

    const int inds_per_iter = 5;
    const int max_iter = 10;
    const double epsilon = 0.2;

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

    //create_identity_plus_minus_csc(n_rows, 3, csc_col_ptr_M, csc_val_M, csc_row_ind_M);
	create_identity_csc(n_rows, csc_col_ptr_M, csc_val_M, csc_row_ind_M);

    // for Cublas padding
    int maxn1 = 0;
    int maxn2 = 0;
    int maxn2tilde = 0;
    int maxn1tilde = 0;


    int* n2s = static_cast<int*>(malloc(sizeof(int) * n_cols));
    int* d_n2s; cudaMalloc((void**)&d_n2s, sizeof(int) * n_cols);
    int* n2tildes = static_cast<int*>(malloc(sizeof(int) * n_cols));
    int* d_n2tildes; cudaMalloc((void**)&d_n2tildes, sizeof(int) * n_cols);

	int* Joffsets = static_cast<int*>(malloc(sizeof(int) * (n_cols+1)));
    int* d_Joffsets; cudaMalloc((void**)&d_Joffsets, sizeof(int) * (n_cols+1));

    int* n1s = static_cast<int*>(malloc(sizeof(int) * n_cols));
    int* d_n1s; cudaMalloc((void**)&d_n1s, sizeof(int) * n_cols);
    int* n1tildes = static_cast<int*>(malloc(sizeof(int) * n_cols));

    int* Ioffsets = static_cast<int*>(malloc(sizeof(int) * (n_cols+1)));
    int* d_Ioffsets; cudaMalloc((void**)&d_Ioffsets, sizeof(int) * (n_cols+1));

    double* A_col_norms = static_cast<double*>(malloc(sizeof(double) * n_cols));

    // A-column norms used later
    for (int k = 0; k < n_cols; k++) // map (^2) reduce (+)
    {
        double acc = 0;
        const int c_ptr = csc_col_ptr_A[k];
        const int c_size = csc_col_ptr_A[k + 1] - c_ptr;
	    for (int i = 0; i < c_size; i++)
	    {
            const double v = csc_val_A[c_ptr + i];
            acc += v*v;
	    }
        A_col_norms[k] = sqrt(acc);
    }

    for (int k = 0; k < n_cols; k++)
    {
        const int beg = csc_col_ptr_M[k]; // points to start of sub-array with column k
        const int end = csc_col_ptr_M[k + 1]; // points to end of sub-array with column k

        // Construct J by getting all row indices from beg to end
        const int n2 = end - beg;
        n2s[k] = n2;
        if (n2 > maxn2)
			maxn2 = n2;
    }

    const int J_size = maxn2 + max_iter * inds_per_iter;
    int* Js = static_cast<int*>(malloc(sizeof(int) * J_size * n_cols));
    int* Jsorteds = static_cast<int*>(malloc(sizeof(int) * J_size * n_cols));
    int *d_Js, *d_Jsorteds;
    cudacall(cudaMalloc((void**)&d_Js, sizeof(int) * J_size * n_cols));
    cudacall(cudaMalloc((void**)&d_Jsorteds, sizeof(int) * J_size * n_cols));

    for (int k = 0; k < n_cols; k++)
    {
        const int beg = csc_col_ptr_M[k]; // points to start of sub-array with column k
        for (int j = 0; j < J_size; j++)
        {
            if (j < n2s[k])
            {
                Js[J_size * k + j] = csc_row_ind_M[j + beg];
                Jsorteds[J_size * k + j] = csc_row_ind_M[j + beg];
            }
            else
            {
                Js[J_size * k + j] = n_cols;
                Jsorteds[J_size * k + j] = n_cols;
            }
        }
    }

    for (int k = 0; k < n_cols + 1; k++)
    {
        Joffsets[k] = k * J_size;
    }
    cudaMemcpy(d_Joffsets, Joffsets, sizeof(int) * (n_cols + 1), cudaMemcpyHostToDevice);
    
    int I_size = 0;
    for (int k = 0; k < n_cols; k++)
    {
        // Construct I: nonzero rows of A[:,J]
        int size = 0;
        for (int j = 0; j < n2s[k]; j++)
        {
            const int c_ind = Js[J_size * k + j];
            size += csc_col_ptr_A[c_ind + 1] - csc_col_ptr_A[c_ind];
        }
        if (size > I_size)
			I_size = size;
    }

    int L_size = I_size + 1;
    dyn_arr<int>* Is = init_dyn_arr<int>(I_size * n_cols);
    dyn_arr<int>* Is_new = init_dyn_arr<int>(1);
    cuda_dyn_arr<int>* d_Is_new = init_cuda_dyn_arr<int>(1);
    dyn_arr<int>* Isorteds = init_dyn_arr<int>(I_size * n_cols);
    cuda_dyn_arr<int>* d_Isorteds = init_cuda_dyn_arr<int>(1);

	int* id_of_ks = static_cast<int*>(malloc(sizeof(int) * n_cols));
    int* d_id_of_ks; cudaMalloc((void**)&d_id_of_ks, sizeof(int)* n_cols);

    for (int k = 0; k < n_cols; k++)
    {
        id_of_ks[k] = -1; // if I doesn't contain k, then stuff needs to be done
        // min heap to store row index, pointer next row index and number of elements left in column 
        MinHeap<int, cuda::std::pair<int,int>>* mh = init_minheap<int, cuda::std::pair<int, int>>(n2s[k]);

        for (int j = 0; j < n2s[k]; j++)
        {
            const int c_ind = Js[J_size * k + j];
            const int c_ptr = csc_col_ptr_A[c_ind];
            const int col_size = csc_col_ptr_A[c_ind + 1] - c_ptr;

        	insert_minheap<int, cuda::std::pair<int, int>>(mh, csc_row_ind_A[c_ptr], cuda::std::pair<int,int>(c_ptr + 1, col_size - 1));
        }
        int prev_ind = -1;
        int n1 = 0;
        int pos_to_ins_k = 0;
        while (mh->size > 0) {
            //cuda::std::pair<int, cuda::std::pair<int, cuda::std::pair<int, int>>> {1, { 2,3 }};
            cuda::std::pair<int, cuda::std::pair<int, int>> curr = get_min<int, cuda::std::pair<int, int>>(mh);
            delete_minimum<int, cuda::std::pair<int, int>>(mh);

            const int row_ind = curr.first;
            const int ptr = curr.second.first;
            const int elems_left = curr.second.second;
            const int I_start = I_size * k;

            if (row_ind != prev_ind)
            {
                Is->array[I_start + n1] = row_ind;
                Isorteds->array[I_start + n1] = row_ind;
                if (row_ind == k)
                    id_of_ks[k] = n1;
                n1++;
            }

            prev_ind = row_ind;
            // checking if the next element belongs to same array as the current array.
            if (elems_left > 0)
                insert_minheap<int, cuda::std::pair<int, int>>(mh, csc_row_ind_A[ptr], cuda::std::pair<int, int>(ptr + 1, elems_left - 1));
        }
        const int i_start = I_size * k;

        // fill out rest of Is_prev->array with large filler vals for sorting
        std::fill(&Is->array[i_start + n1], &Is->array[i_start + I_size], n_cols);

        free_minheap<int, cuda::std::pair<int, int>>(mh);
    	n1s[k] = n1;
        if (n1 > maxn1)
            maxn1 = n1;
    }
    cudaMemcpy(d_id_of_ks, id_of_ks, sizeof(int)* n_cols, cudaMemcpyHostToDevice);

	int A_size = maxn1 * maxn2;
    int Q_size = maxn1 * maxn1;
    
    dyn_arr<double>* Ahats = init_dyn_arr<double>(A_size * n_cols);
    construct_Ahats(n_cols, Ahats->array, A_size, Js, n2s, J_size, Is->array, n1s, I_size, maxn1, maxn2, csc_col_ptr_A, csc_row_ind_A, csc_val_A);

    // QR
    int ltau = std::max(1, std::min(maxn1, maxn2));
    cuda_dyn_arr<double>* d_Ahat = init_cuda_dyn_arr<double>(n_cols * A_size);
    cuda_dyn_arr<double>* d_Taus = init_cuda_dyn_arr<double>(n_cols * ltau);
    dyn_arr<double>* taus = init_dyn_arr<double>(n_cols * ltau);

    cudacall(cudaMemcpy(d_Ahat->array, Ahats->array, n_cols * A_size * sizeof(double), cudaMemcpyHostToDevice));

    double** d_Aarray, ** d_TauArray;
    cudaMalloc((void**)&d_Aarray, n_cols * sizeof(double*));
    cudaMalloc((void**)&d_TauArray, n_cols * sizeof(double*));

    int minGridSize, blockSize, gridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, set_ptr_array, 0, 0);
    gridSize = (n_cols + blockSize - 1) / blockSize;
    set_ptr_array<<<gridSize, blockSize>>>(d_Aarray, d_Ahat->array, A_size, 0, 0, maxn1, n_cols);
    set_ptr_array<<<gridSize, blockSize>>>(d_TauArray, d_Taus->array, ltau, 0, 0, 0, n_cols);

    int info;
    cublascall(cublasDgeqrfBatched(handle, maxn1, maxn2, d_Aarray, maxn1, d_TauArray, &info, n_cols));

    cudacall(cudaMemcpy(Ahats->array, d_Ahat->array, n_cols * A_size * sizeof(double), cudaMemcpyDeviceToHost));
    cudacall(cudaMemcpy(taus->array, d_Taus->array, n_cols * ltau * sizeof(double), cudaMemcpyDeviceToHost));

    // Construct Q and solve upper triangular
    // Q = I_n1
    dyn_arr<double>* Qs = init_dyn_arr<double>(n_cols * Q_size);
    cuda_dyn_arr<double>* d_Qs = init_cuda_dyn_arr<double>(n_cols * Q_size);
    for (int k = 0; k < n_cols; k++) {
	    for (int j = 0; j < maxn1; j++) {
		    for (int i = 0; i < maxn1; i++) {
			    if (i == j) {
			    	Qs->array[k * Q_size + IDX2C(i, j, maxn1)] = 1;
			    } else {
			    	Qs->array[k * Q_size + IDX2C(i, j, maxn1)] = 0;
			    }
		    }
	    }
    }

    cudaMemcpy(d_Qs->array, Qs->array, sizeof(double) * n_cols * Q_size, cudaMemcpyHostToDevice);

	cuda_dyn_arr<double>* d_vs = init_cuda_dyn_arr<double>(n_cols * maxn1);
    cuda_dyn_arr<double>* d_Qvs = init_cuda_dyn_arr<double>(n_cols * maxn1);

    double** d_QsArray; double** d_vsArray; double** d_QvsArray;
    cudaMalloc((void**)&d_QsArray, sizeof(double*)* n_cols);
    cudaMalloc((void**)&d_vsArray, sizeof(double*)* n_cols);
    cudaMalloc((void**)&d_QvsArray, sizeof(double*)* n_cols);

    cudaMemcpy(d_n2s, n2s, sizeof(int)* n_cols, cudaMemcpyHostToDevice);

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, set_ptr_array, 0, 0);
    gridSize = (n_cols + blockSize - 1) / blockSize;
    set_ptr_array << <gridSize, blockSize >> > (d_QvsArray, d_Qvs->array, maxn1, 0, 0, 0, n_cols);
    for (int k = 0; k < maxn2; k++)
    {
        const int k_to_m = maxn1 - k;

        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, set_vs_initial, 0, 0);
        gridSize = (k_to_m * n_cols + blockSize - 1) / blockSize;
        set_vs_initial<<<gridSize, blockSize >>>(d_vs->array, k_to_m, d_Ahat->array, A_size, d_n2s, maxn1, k, k_to_m * n_cols);

        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, set_ptr_array, 0, 0);
        gridSize = (n_cols + blockSize - 1) / blockSize;
        set_ptr_array<<<gridSize, blockSize>>>(d_QsArray, d_Qs->array, Q_size, k, 0, maxn1, n_cols);
        set_ptr_array<<<gridSize, blockSize>>>(d_vsArray, d_vs->array, k_to_m, 0, 0, 0, n_cols);

        double alpha = 1, beta = 0;
        cublasDgemvBatched(handle, CUBLAS_OP_N, maxn1, k_to_m, &alpha, d_QsArray, maxn1, d_vsArray, 1, &beta, d_QvsArray, 1, n_cols);

        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, mult_taus, 0, 0);
        gridSize = (maxn1 * n_cols + blockSize - 1) / blockSize;
        mult_taus<<<gridSize, blockSize>>>(d_Qvs->array, d_Taus->array, ltau, k, maxn1, maxn1 * n_cols);

        beta = 1; alpha = -1;
        cublascall(cublasDgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T, maxn1, k_to_m, 1, &alpha, d_QvsArray, maxn1, d_vsArray, k_to_m, &beta, d_QsArray, maxn1, n_cols));
    }

    cudaMemcpy(Qs->array, d_Qs->array, sizeof(double)* n_cols* Q_size, cudaMemcpyDeviceToHost);

    // Q^T*e_k[I]: index of k'th row of Q transposed and first n2 elements. Will contain result mhat_k
    dyn_arr<double>* mhats = init_dyn_arr<double>(n_cols * maxn2);
    cuda_dyn_arr<double>* d_mhats = init_cuda_dyn_arr<double>(n_cols * maxn2);
	for (int k = 0; k < n_cols; k++)
	{
        const int i = id_of_ks[k];
        if (i >= 0) { // if I contains k, otherwise Q^T * e_k[I] = 0
            for (int j = 0; j < maxn2; j++) {
                if (j < n2s[k])
                    mhats->array[k * maxn2 + j] = Qs->array[k * Q_size + IDX2C(i, j, maxn1)];
                else
                    mhats->array[k * maxn2 + j] = 0;
            }
        }
		else
        {
            for (int j = 0; j < maxn2; j++)
            {
            	mhats->array[k * maxn2 + j] = 0;
            }
        }
	}
    
    cudacall(cudaMemcpy(d_Ahat->array, Ahats->array, n_cols * A_size * sizeof(double), cudaMemcpyHostToDevice));
    cudacall(cudaMemcpy(d_mhats->array, mhats->array, n_cols * maxn2 * sizeof(double), cudaMemcpyHostToDevice));
    
    double** d_mhatsArray;
    cudaMalloc((void**)&d_mhatsArray, n_cols * sizeof(double*));
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, set_ptr_array, 0, 0);
    gridSize = (n_cols + blockSize - 1) / blockSize;
    set_ptr_array << <gridSize, blockSize >> > (d_Aarray, d_Ahat->array, A_size, 0, 0, maxn1, n_cols);
    set_ptr_array << <gridSize, blockSize >> > (d_mhatsArray, d_mhats->array, maxn2, 0, 0, 0, n_cols);

    double alpha = 1;
    cublascall(cublasDtrsmBatched(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, maxn2, 1, &alpha, d_Aarray, maxn1, d_mhatsArray, maxn2, n_cols));
    
    cudacall(cudaMemcpy(mhats->array, d_mhats->array, n_cols * maxn2 * sizeof(double), cudaMemcpyDeviceToHost));

    dyn_arr<double>* rs = init_dyn_arr<double>(n_cols * maxn1);
    cuda_dyn_arr<double>* d_rs = init_cuda_dyn_arr<double>(n_cols * maxn1);
    dyn_arr<double>* AIJs = init_dyn_arr<double>(A_size * n_cols); // For rI = A[I,J] * mhat_k - e_k[I]
    cuda_dyn_arr<double>* d_AIJs = init_cuda_dyn_arr<double>(A_size * n_cols);
    construct_Ahats(n_cols, AIJs->array, A_size, Js, n2s, J_size, Is->array, n1s, I_size, maxn1, maxn2, csc_col_ptr_A, csc_row_ind_A, csc_val_A);
    cudaMemcpy(d_AIJs->array, AIJs->array, sizeof(double) * A_size * n_cols, cudaMemcpyHostToDevice);

    double** d_rsArray;
    cudaMalloc((void**)&d_rsArray, n_cols * sizeof(double*));

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, set_ptr_array, 0, 0);
    gridSize = (n_cols + blockSize - 1) / blockSize;
    set_ptr_array << <gridSize, blockSize >> > (d_rsArray, d_rs->array, maxn1, 0, 0, maxn1, n_cols);
    set_ptr_array << <gridSize, blockSize >> > (d_Aarray, d_AIJs->array, A_size, 0, 0, maxn1, n_cols);

	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, init_rs, 0, 0);
    gridSize = (maxn1 * n_cols + blockSize - 1) / blockSize;
    init_rs << <gridSize, blockSize >> > (d_rs->array, d_id_of_ks, maxn1, maxn1 * n_cols);

    alpha = 1;
	double beta = -1;
    cublasDgemvBatched(handle, CUBLAS_OP_N, maxn1, maxn2, &alpha, d_Aarray, maxn1, d_mhatsArray, 1, &beta, d_rsArray, 1, n_cols);

    cudaMemcpy(rs->array, d_rs->array, sizeof(double) * maxn1 * n_cols, cudaMemcpyDeviceToHost);

    double* r_norms = static_cast<double*>(malloc(sizeof(double) * n_cols));

    // segmented map reduce to calculate norm
    for (int k = 0; k < n_cols; k++)
    {
        double norm = 0;
        for (int i = 0; i < maxn1; i++)
        {
	        if (i < n1s[k])
	        {
                const double val = rs->array[maxn1 * k + i];
                norm += val * val;
	        }
        }
        r_norms[k] = sqrt(norm);
    }

    for (int k = 0; k < n_cols; k++)
        printf("col %d norm: %f\n", k, r_norms[k]);

    // while loop begins omg
    bool above_epsilon = false;
    for (int k = 0; k < n_cols && !above_epsilon; k++)
    {
        above_epsilon |= r_norms[k] > epsilon;
    }

    // initialize arrays to be reused during iterations
    dyn_arr<double>* AIJtildes = init_dyn_arr<double>(1);
    dyn_arr<double>* Anews = init_dyn_arr<double>(1);
    dyn_arr<double>* Qnews = init_dyn_arr<double>(1);
    dyn_arr<double>* B2s = init_dyn_arr<double>(1);
    cuda_dyn_arr<double>* d_AIJtildes = init_cuda_dyn_arr<double>(1);
    cuda_dyn_arr<double>* d_Anews = init_cuda_dyn_arr<double>(1);
    cuda_dyn_arr<double>* d_Qnews = init_cuda_dyn_arr<double>(1);
    cuda_dyn_arr<double>* d_B2s = init_cuda_dyn_arr<double>(1);

    double** d_AIJtildesArray;
    double** d_AnewsArray;
    cudaMalloc((void**)&d_AIJtildesArray, n_cols * sizeof(double*));
    cudaMalloc((void**)&d_AnewsArray, n_cols * sizeof(double*));

    int iter = 0; 
    while (above_epsilon && iter < max_iter)
    {
        iter++;
        printf("iter: %d\n", iter);
        // read each element of A in a coalesced way to get into L3 cache. If cond that cannot be optimized away by compiler
        // forall tid = 0 .. num_nonzeros do
        //   (a,i ) = A[tid]
        //   if (nrver_true_cond)
        //     x[i] = a*2

        for (int k = 0; k < n_cols; k++)
        {
            // heaps should be regular across columns in parallel application
            MinHeap<int, cuda::std::pair<int, int>>* mh = init_minheap<int, cuda::std::pair<int, int>>(std::max(maxn1 + 1, inds_per_iter)); //   reused later so max of 
            MinHeap<double, int>* rho_j_pairs = init_minheap<double, int>(inds_per_iter);
            
	        const int I_start = I_size * k;
            for (int l = 0; l < I_size; l++)
            {
	            if (l < n1s[k])
	            {
                    const int row_ind = Isorteds->array[I_start + l];
                    const int r_ptr = csr_row_ptr_A[row_ind];
                    const int row_size = csr_row_ptr_A[row_ind + 1] - r_ptr;
                    insert_minheap<int, cuda::std::pair<int, int>>(mh, csr_col_ind_A[r_ptr], cuda::std::pair<int, int>(r_ptr + 1, row_size - 1));
	            }
            }

            if (id_of_ks[k] == -1)
            {
                const int row_ind = k;
                const int r_ptr = csr_row_ptr_A[row_ind];
                const int row_size = csr_row_ptr_A[row_ind + 1] - r_ptr;
                insert_minheap<int, cuda::std::pair<int, int>>(mh, csr_col_ind_A[r_ptr], cuda::std::pair<int, int>(r_ptr + 1, row_size - 1));
            }

            int count = 0;
            double avg = 0;
            const int J_start = J_size * k;
            int J_ptr = 0;
            int prev_col = -1;
            while (mh->size > 0) {
                cuda::std::pair<int, cuda::std::pair<int, int>> curr = get_min<int, cuda::std::pair<int, int>>(mh);
                delete_minimum<int, cuda::std::pair<int, int>>(mh);
                const int j = curr.first;
                const int ptr = curr.second.first;
                const int elems_left = curr.second.second;

                while (J_ptr < n2s[k] && Jsorteds[J_start + J_ptr] < j)
                    J_ptr++;
                if (j != prev_col && j != Jsorteds[J_start + J_ptr]) // if new index doesn't belong to J (denoted by negative pointer)
                {
                    // calc Rho from j and residual.
                    const int c_ptr = csc_col_ptr_A[j];
                    const int c_size = csc_col_ptr_A[j + 1] - c_ptr;

                    double rTAej = 0;
                    const int r_size = n1s[k];
                    
                    for (int l = 0; l < r_size; l++)
                    {
                        const int l_ind = Is->array[I_start + l];
                        int i = 0;
                        while (i < c_size && csc_row_ind_A[c_ptr + i] < l_ind) // linear search could be binary
                            i++;
                        if (l_ind == csc_row_ind_A[c_ptr + i])
                            rTAej += rs->array[maxn1 * k + l] * csc_val_A[c_ptr + i];
                    }

                    if (id_of_ks[k] == -1)
                    {
                        const int l_ind = k;
                        int i = 0;
                        while (i < c_size && csc_row_ind_A[c_ptr + i] < l_ind) // linear search could be binary
                            i++;
                        if (l_ind == csc_row_ind_A[c_ptr + i])
                            rTAej += -1 * csc_val_A[c_ptr + i];
                    }

                    const double minus_rho = -(r_norms[k] * r_norms[k] - pow(rTAej / A_col_norms[j], 2));
                    avg -= minus_rho;
                    count++;

                    if (rho_j_pairs->size >= inds_per_iter)
                    {
                        const double min = get_min<double, int>(rho_j_pairs).first;
                        if (minus_rho > min)
                        {
                            delete_minimum<double, int>(rho_j_pairs);
                            insert_minheap<double, int>(rho_j_pairs, minus_rho, j);
                        }
                    }
                	else
                    {
                        insert_minheap<double, int>(rho_j_pairs, minus_rho, j);
                    }
                }

                prev_col = j;
                if (elems_left > 0)
                    insert_minheap<int, cuda::std::pair<int, int>>(mh, csc_row_ind_A[ptr], cuda::std::pair<int, int>(ptr + 1, elems_left -   1));
            }
            avg /= count;
            while (rho_j_pairs->size > 0) // reuse mh
            {
                const cuda::std::pair<double, int> pair = get_min<double, int>(rho_j_pairs);
                delete_minimum<double, int>(rho_j_pairs);
                if (-pair.first < avg)
					insert_minheap<int, cuda::std::pair<int, int>>(mh, pair.second, {});
                //insert_minheap<int, cuda::std::pair<int, int>>(mh, pair.second, {});
            }
            int n2tilde = 0;
            while (mh->size > 0)
            {
                const int col_ind = get_min<int, cuda::std::pair<int, int>>(mh).first;
                delete_minimum<int, cuda::std::pair<int, int>>(mh);
                Js[J_size * k + n2s[k] + n2tilde] = col_ind; // Could also place n2tilde at an equal offset (maxn2 instead of n2s[k]) for regularity
                n2tilde++;
            }
            n2tildes[k] = n2tilde;
            if (n2tilde > maxn2tilde)
                maxn2tilde = n2tilde;
            free_minheap<int, cuda::std::pair<int, int>>(mh);
            free_minheap<double,int>(rho_j_pairs);
        }

        // Get Itilde: New inds not in I from shadow of Jtilde
        int new_I_size = 0;
        for (int k = 0; k < n_cols; k++)
        {
            int size = 0;
            for (int j = 0; j < maxn2tilde; j++)
            {
                if (j < n2tildes[k])
                {
                    const int c_ind = Js[J_size * k + n2s[k] + j];
                    size += csc_col_ptr_A[c_ind + 1] - csc_col_ptr_A[c_ind];
                }
            }
            size += n1s[k];
            if (size > new_I_size)
				new_I_size = size;
        }

        for (int k = 0; k < n_cols + 1; k++)
        {
            Ioffsets[k] = new_I_size * k;
        }
        cudaMemcpy(d_Ioffsets, Ioffsets, sizeof(int)* (n_cols + 1), cudaMemcpyHostToDevice);

        resize_based_on_write_size(Is_new, new_I_size * n_cols); // resize Is_new
        for (int k = 0; k < n_cols; k++)
        {
            std::memcpy(&Is_new->array[new_I_size * k], &Is->array[I_size * k], n1s[k] * sizeof(int));
            std::fill(&Is_new->array[new_I_size * k + n1s[k]], &Is_new->array[new_I_size * (k + 1)], n_cols);
        }
        
        for (int k = 0; k < n_cols; k++)
        {
            // if I doesn't contain k, then stuff needs to be done
            MinHeap<int, cuda::std::pair<int, int>>* mh = init_minheap<int, cuda::std::pair<int, int>>(maxn2tilde);

            for (int j = 0; j < maxn2tilde; j++)
            {
                if (j < n2tildes[k])
                {
                    const int c_ind = Js[J_size * k + n2s[k] + j];
                    const int c_ptr = csc_col_ptr_A[c_ind];
                    const int col_size = csc_col_ptr_A[c_ind + 1] - c_ptr;

                    insert_minheap<int, cuda::std::pair<int, int>>(mh, csc_row_ind_A[c_ptr], cuda::std::pair<int, int>(c_ptr + 1, col_size   - 1));
                }
            }

            int n1tilde = 0;
            const int I_start = I_size * k;
            int I_ptr = 0;
            int prev_row = -1;
            while (mh->size > 0) {
                cuda::std::pair<int, cuda::std::pair<int, int>> curr = get_min<int, cuda::std::pair<int, int>>(mh);
                delete_minimum<int, cuda::std::pair<int, int>>(mh);

                const int i = curr.first;
                const int ptr = curr.second.first;
                const int elems_left = curr.second.second;
                const int Inew_start = new_I_size * k + n1s[k];

                while (I_ptr < n1s[k] && Isorteds->array[I_start + I_ptr] < i)
                    I_ptr++;
                if (i != prev_row && i != Isorteds->array[I_start + I_ptr])
                {
                    Is_new->array[Inew_start + n1tilde] = i;
                    n1tilde++;
                    if (id_of_ks[k] != -1 && i == k)
                        id_of_ks[k] = n1s[k] + n1tilde;
                }
                prev_row = i;
                // checking if the next element belongs to same array as the current array.
                if (elems_left > 0)
                    insert_minheap<int, cuda::std::pair<int, int>>(mh, csc_row_ind_A[ptr], cuda::std::pair<int, int>(ptr + 1, elems_left -   1));
            }

            free_minheap<int, cuda::std::pair<int, int>>(mh);
            n1tildes[k] = n1tilde;
            if (n1tilde > maxn1tilde)
                maxn1tilde = n1tilde;
        }

        // Construct AIJtilde = A[I, Jtilde]
        const int AIJtilde_size = maxn1 * maxn2tilde;
        resize_based_on_write_size(AIJtildes, AIJtilde_size * n_cols);
        for (int k = 0; k < n_cols; k++)
        {
            int AIJtildes_ind = AIJtilde_size * k;
            for (int j = 0; j < maxn2tilde; j++)
            {
                const int col_ind = Js[J_size * k + n2s[k] + j]; // Jtilde starts after J in Js
                const int col_start = csc_col_ptr_A[col_ind];
                const int col_size = csc_col_ptr_A[col_ind + 1] - col_start;
                for (int i = 0; i < maxn1; i++)
                {
                    if (j > n2tildes[k] - 1)
                        AIJtildes->array[AIJtildes_ind + IDX2C(i, j, maxn1)] = 0;
                    else
                    {
                        int col_elem_idx = 0;
                        const int i_ind = Is_new->array[new_I_size * k + i];
                        while (col_elem_idx < col_size && csc_row_ind_A[col_start + col_elem_idx] < i_ind) // linear search can be binary
                            col_elem_idx++;
                        if (i_ind == csc_row_ind_A[col_start + col_elem_idx])
                        {
                            AIJtildes->array[AIJtildes_ind + IDX2C(i, j, maxn1)] = csc_val_A[col_start + col_elem_idx];
                        }
                        else
                        {
                            AIJtildes->array[AIJtildes_ind + IDX2C(i, j, maxn1)] = 0;
                        }
                    }
                }
            }
        }

        int maxnewn1 = 0;
        int maxnewn2 = 0;
        int maxB2n = 0;
        for (int k = 0; k < n_cols; k++)
        {
            const int n2 = n2s[k];
            const int newn1 = n1s[k] + n1tildes[k];
            const int newn2 = n2 + n2tildes[k];
            const int B2n = newn1 - n2;
            if (newn1 > maxnewn1)
                maxnewn1 = newn1;
            if (newn2 > maxnewn2)
                maxnewn2 = newn2;
            if (B2n > maxB2n)
                maxB2n = B2n;
        }

    	const int Anew_size = maxnewn1 * maxnewn2;
        resize_based_on_write_size(Anews, Anew_size * n_cols);
        // Au starts at n2s[k] * maxnewn1
        // B2 starts at n2s[k] * maxnewn1 + (n1s[k]-n2s[k])
        // AItildeJtildes starts at n2s[k] * maxnewn1 + n1s[k]
        // ld is maxnew1 for all

    	// Au = Q.T * AIJtilde
        cuda_resize_based_on_write_size(d_Qs, n_cols* Q_size);
        cuda_resize_based_on_write_size(d_Anews, n_cols* Anew_size);
        cuda_resize_based_on_write_size(d_AIJtildes, n_cols* AIJtilde_size);

        cudacall(cudaMemcpy(d_Qs->array, Qs->array, n_cols * Q_size * sizeof(double), cudaMemcpyHostToDevice));
        cudacall(cudaMemcpy(d_AIJtildes->array, AIJtildes->array, n_cols * AIJtilde_size * sizeof(double), cudaMemcpyHostToDevice));
        cudacall(cudaMemcpy(d_n2s, n2s, n_cols * sizeof(int), cudaMemcpyHostToDevice));

        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, set_ptr_array_with_irregular_column_offsets, 0, 0);
        gridSize = (n_cols + blockSize - 1) / blockSize;
        set_ptr_array_with_irregular_column_offsets << <gridSize, blockSize >> > (d_AnewsArray, d_Anews->array, Anew_size, d_n2s, 0, 0, maxnewn1, n_cols);
        set_ptr_array << <gridSize, blockSize >> > (d_QsArray, d_Qs->array, Q_size, 0, 0, maxn1, n_cols);
        set_ptr_array << <gridSize, blockSize >> > (d_AIJtildesArray, d_AIJtildes->array, AIJtilde_size, 0, 0, maxn1, n_cols);

        double alpha = 1, beta = 0;
        cublasDgemmBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, maxn1, maxn2tilde, maxn1, &alpha, d_QsArray, maxn1, d_AIJtildesArray, maxn1, &beta, d_AnewsArray, maxnewn1, n_cols);
        
        cudacall(cudaMemcpy(Anews->array, d_Anews->array, n_cols * Anew_size * sizeof(double), cudaMemcpyDeviceToHost));

        for (int k = 0; k < n_cols; k++)
        {
            int AItildeJtildes_ind = Anew_size * k + n2s[k] * maxnewn1 + n1s[k];
            for (int j = 0; j < maxn2tilde; j++)
            {
                const int col_ind = Js[J_size * k + n2s[k] + j]; // Jtilde starts after J in Js
                const int col_start = csc_col_ptr_A[col_ind];
                const int col_size = csc_col_ptr_A[col_ind + 1] - col_start;
                for (int i = 0; i < maxn1tilde; i++)
                {
                    if (i > n1tildes[k] - 1 || j > n2tildes[k] - 1)
                    {
                        // do nothing
                    }
                    else
                    {
                        int col_elem_idx = 0;
                        const int i_ind = Is_new->array[new_I_size * k + n1s[k] + i]; // Itilde starts after I in Is_new->array
                        while (col_elem_idx < col_size && csc_row_ind_A[col_start + col_elem_idx] < i_ind)
                            col_elem_idx++;
                        if (i_ind == csc_row_ind_A[col_start + col_elem_idx])
                        {
                            Anews->array[AItildeJtildes_ind + IDX2C(i, j, maxnewn1)] = csc_val_A[col_start + col_elem_idx];
                        }
                        else
                        {
                            Anews->array[AItildeJtildes_ind + IDX2C(i, j, maxnewn1)] = 0;
                        }
                    }
                }
            }
        }

        // QR time !!!!!!!!!!!
        int B2_size = maxB2n * maxn2tilde;
        resize_based_on_write_size(B2s, n_cols * B2_size);
        cuda_resize_based_on_write_size(d_B2s, n_cols * B2_size);
        std::fill(&B2s->array[0], &B2s->array[n_cols * B2_size], 0);
        for (int k = 0; k < n_cols; k++)
        {
	        for (int j = 0; j < maxn2tilde; j++)
	        {
                if (j < n2tildes[k])
                    memcpy(&B2s->array[B2_size * k + maxB2n * j], &Anews->array[Anew_size * k + maxnewn1 * (j + n2s[k]) + n2s[k]], (n1s[k] + n1tildes[k] - n2s[k]) * sizeof(double));
	        }
        }

        cudaMemcpy(d_B2s->array, B2s->array, sizeof(double) * n_cols * B2_size, cudaMemcpyHostToDevice);

        ltau = std::max(1, std::min(maxB2n, maxn2tilde));
        cuda_resize_based_on_write_size(d_Taus, ltau* n_cols);
        resize_based_on_write_size(taus, ltau* n_cols);

        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, set_ptr_array_with_irregular_column_and_row_offsets, 0, 0);
        gridSize = (n_cols + blockSize - 1) / blockSize;
        set_ptr_array << <gridSize, blockSize >> > (d_AnewsArray, d_B2s->array, B2_size, 0, 0, maxB2n, n_cols);
        set_ptr_array << <gridSize, blockSize >> > (d_TauArray, d_Taus->array, ltau, 0, 0, 0, n_cols);

        int info;
        cublascall(cublasDgeqrfBatched(handle, maxB2n, maxn2tilde, d_AnewsArray, maxB2n, d_TauArray, &info, n_cols));

        cudacall(cudaMemcpy(B2s->array, d_B2s->array, n_cols * B2_size * sizeof(double), cudaMemcpyDeviceToHost));
        cudacall(cudaMemcpy(taus->array, d_Taus->array, n_cols * ltau * sizeof(double), cudaMemcpyDeviceToHost));

        // copy B2 into A
        for (int k = 0; k < n_cols; k++)
        {
            for (int j = 0; j < maxn2tilde; j++)
            {
                if (j < n2tildes[k])
                    memcpy(&Anews->array[Anew_size * k + maxnewn1 * (j + n2s[k]) + n2s[k]], &B2s->array[B2_size * k + maxB2n * j], (n1s[k] + n1tildes[k] - n2s[k]) * sizeof(double));
            }
        }

        // copy prev A int new A // !!!!!!!!!!!!!!! If you can't use previous householder, then only copy R-part (maxn2) and not
        // the householder part (maxn1)
        for (int k = 0; k < n_cols; k++)
        {
            for (int j = 0; j < maxn2; j++)
            {
	            if (j < n2s[k])
	            {
                    memcpy(&Anews->array[Anew_size * k + maxnewn1 * j], &Ahats->array[A_size * k + maxn1 * j], sizeof(double) * maxn1);
	            }
            }
            for (int ij = 0; ij < std::min(maxnewn2, maxnewn1); ij++)
            {
                if (ij > std::min(n2s[k] + n2tildes[k] - 1, n1s[k] + n1tildes[k] - 1))
                    Anews->array[Anew_size * k + maxnewn1 * ij + ij] = 1;
            }
        }

        const int Qnew_size = maxnewn1 * maxnewn1;
        resize_based_on_write_size(Qnews, n_cols* Qnew_size);
        cuda_resize_based_on_write_size(d_Qnews, n_cols* Qnew_size);
        std::fill(Qnews->array, &Qnews->array[n_cols * Qnew_size], 0);
        for (int k = 0; k < n_cols; k++) {
            for (int ij = 0; ij < maxnewn1; ij++)
            {
            	Qnews->array[Qnew_size * k + maxnewn1 * ij + ij] = 1;
            }
            // Copy in previous Q and zeros
            for (int j = 0; j < maxn1; j++)
            {
            	memcpy(&Qnews->array[Qnew_size * k + maxnewn1 * j], &Qs->array[Q_size * k + maxn1 * j], sizeof(double) * maxn1);
            }
        }
        cuda_resize_based_on_write_size(d_Qnews, Qnew_size* n_cols);
        cudaMemcpy(d_Qnews->array, Qnews->array, Qnew_size* n_cols * sizeof(double), cudaMemcpyHostToDevice);

        // You only apply housholder to last n1 - n2 + n1tilde columns, so matrix starts at maxn1tilde * n2
        cuda_resize_based_on_write_size(d_vs, n_cols * maxB2n);
        cuda_resize_based_on_write_size(d_Qvs, n_cols * maxB2n); // Should be possible to do like before the loop but ran out of time
        
        cudaMemcpy(d_n2tildes, n2tildes, sizeof(int)* n_cols, cudaMemcpyHostToDevice);

        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, set_ptr_array, 0, 0);
        gridSize = (n_cols + blockSize - 1) / blockSize;
        set_ptr_array << <gridSize, blockSize >> > (d_QvsArray, d_Qvs->array, maxB2n, 0, 0, 0, n_cols);
        for (int k = 0; k < maxB2n; k++)
        {
            const int k_to_m = maxB2n - k;

            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, set_vs_initial, 0, 0);
            gridSize = (k_to_m * n_cols + blockSize - 1) / blockSize;
            set_vs_initial << <gridSize, blockSize >> > (d_vs->array, k_to_m, d_B2s->array, B2_size, d_n2tildes, maxB2n, k, k_to_m * n_cols);

        	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, set_ptr_array_with_irregular_column_offsets, 0, 0);
            gridSize = (n_cols + blockSize - 1) / blockSize;
            set_ptr_array_with_irregular_column_offsets << <gridSize, blockSize >> > (d_QsArray, d_Qnews->array, Qnew_size, d_n2s, k, 0, maxnewn1, n_cols);
            set_ptr_array << <gridSize, blockSize >> > (d_vsArray, d_vs->array, k_to_m, 0, 0, 0, n_cols);

            double alpha = 1, beta = 0;
            cublasDgemvBatched(handle, CUBLAS_OP_N, maxB2n, k_to_m, &alpha, d_QsArray, maxnewn1, d_vsArray, 1, &beta, d_QvsArray, 1, n_cols);

            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, mult_taus, 0, 0);
            gridSize = (maxB2n * n_cols + blockSize - 1) / blockSize;
            mult_taus << <gridSize, blockSize >> > (d_Qvs->array, d_Taus->array, ltau, k, maxB2n, maxB2n * n_cols);

            beta = 1; alpha = -1;
            //cublascall(cublasDgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T, maxB2n, k_to_m, 1, &alpha, d_QvsArray, maxB2n, d_vsArray, k_to_m, &beta, d_QsArray, maxn1, n_cols));

            cublasDgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T, maxB2n, k_to_m, 1, &alpha, d_QvsArray, maxB2n, d_vsArray, k_to_m, &beta, d_QsArray, maxnewn1, n_cols);
        }

        cudaMemcpy(Qnews->array, d_Qnews->array, sizeof(double) * n_cols * Qnew_size, cudaMemcpyDeviceToHost);

        // chat_k: index of k in I'th row of Q transposed and first n2 elements.
        resize_based_on_write_size(mhats, n_cols* maxnewn2);
        cuda_resize_based_on_write_size(d_mhats, n_cols* maxnewn2);
        for (int k = 0; k < n_cols; k++)
        {
            const int i = id_of_ks[k];
            if (i >= 0) { // if I contains k, otherwise Q^T * e_k[I] = 0
                for (int j = 0; j < maxnewn2; j++)
                {
                    if (j < n2s[k] + n2tildes[k])
                        mhats->array[k * maxnewn2 + j] = Qnews->array[k * Qnew_size + IDX2C(i, j, maxnewn1)];
                    else
                        mhats->array[k * maxnewn2 + j] = 0;
                }
            }
            else
            {
                for (int j = 0; j < maxnewn2; j++)
                    mhats->array[k * maxnewn2 + j] = 0;
            }
        }
        
        cudacall(cudaMemcpy(d_mhats->array, mhats->array, n_cols * maxnewn2 * sizeof(double), cudaMemcpyHostToDevice));
        cudacall(cudaMemcpy(d_Anews->array, Anews->array, n_cols * Anew_size * sizeof(double), cudaMemcpyHostToDevice));

        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, set_ptr_array, 0, 0);
        gridSize = (n_cols + blockSize - 1) / blockSize;
        set_ptr_array << <gridSize, blockSize >> > (d_AnewsArray, d_Anews->array, Anew_size, 0, 0, 0, n_cols);
        set_ptr_array << <gridSize, blockSize >> > (d_mhatsArray, d_mhats->array, maxnewn2, 0, 0, 0, n_cols);

        alpha = 1;
        cublascall(cublasDtrsmBatched(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, maxnewn2, 1, &alpha, d_AnewsArray, maxnewn1, d_mhatsArray, maxnewn2, n_cols));
        
        cudacall(cudaMemcpy(mhats->array, d_mhats->array, n_cols * maxnewn2 * sizeof(double), cudaMemcpyDeviceToHost));


        // Sort Js and Is to be used in next iteration
        resize_based_on_write_size(Isorteds, n_cols * new_I_size);
        cuda_resize_based_on_write_size(d_Isorteds, n_cols * new_I_size);
        cuda_resize_based_on_write_size(d_Is_new, n_cols * new_I_size);

        cudacall(cudaMemcpy(d_Js, Js, sizeof(int)* n_cols* J_size, cudaMemcpyHostToDevice));
        cudacall(cudaMemcpy(d_Is_new->array, Is_new->array, sizeof(int)* n_cols* new_I_size, cudaMemcpyHostToDevice));

        void* d_temp_storage = NULL;
        size_t   temp_storage_bytes = 0;
        cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_Js, d_Jsorteds, n_cols*J_size, n_cols, d_Joffsets, d_Joffsets + 1);

        // Allocate temporary storage
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        // Run sorting operation
        cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_Js, d_Jsorteds, n_cols * J_size, n_cols, d_Joffsets, d_Joffsets + 1);

        cudaFree(d_temp_storage);
        d_temp_storage = NULL;
        temp_storage_bytes = 0;
        cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_Is_new->array, d_Isorteds->array, n_cols* new_I_size, n_cols, d_Ioffsets, d_Ioffsets + 1);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_Is_new->array, d_Isorteds->array, n_cols* new_I_size, n_cols, d_Ioffsets, d_Ioffsets + 1);

        cudaMemcpy(Jsorteds, d_Jsorteds, sizeof(int) * J_size * n_cols, cudaMemcpyDeviceToHost);
        cudaMemcpy(Isorteds->array, d_Isorteds->array, sizeof(int) * new_I_size * n_cols, cudaMemcpyDeviceToHost);

        // Update n1s and n2s and copy over prev to new
        for (int k = 0; k < n_cols; k++)
        {
            n1s[k] += n1tildes[k];
            n2s[k] += n2tildes[k];
        }
        maxn1 = maxnewn1;
        maxn2 = maxnewn2;
        I_size = new_I_size;
        Q_size = Qnew_size;
        A_size = Anew_size;

        resize_based_on_write_size(Is, n_cols* I_size);
        resize_based_on_write_size(Qs, n_cols* Q_size);
        resize_based_on_write_size(Ahats, n_cols* A_size);

        memcpy(Is->array, Is_new->array, sizeof(int)* n_cols* I_size);
        memcpy(Qs->array, Qnews->array, sizeof(double)* n_cols* Q_size);
        memcpy(Ahats->array, Anews->array, sizeof(double)* n_cols* A_size);

        resize_based_on_write_size(rs, n_cols* maxnewn1);
        cuda_resize_based_on_write_size(d_rs, n_cols* maxnewn1);
        resize_based_on_write_size(AIJs, n_cols* Anew_size);
        cuda_resize_based_on_write_size(d_AIJs, n_cols* Anew_size);

        construct_Ahats(n_cols, AIJs->array, A_size, Js, n2s, J_size, Is->array, n1s, I_size, maxn1, maxn2, csc_col_ptr_A, csc_row_ind_A, csc_val_A);
        cudaMemcpy(d_AIJs->array, AIJs->array, sizeof(double) * A_size * n_cols, cudaMemcpyHostToDevice);

        
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, set_ptr_array, 0, 0);
        gridSize = (n_cols + blockSize - 1) / blockSize;
        set_ptr_array << <gridSize, blockSize >> > (d_rsArray, d_rs->array, maxn1, 0, 0, maxn1, n_cols);
        set_ptr_array << <gridSize, blockSize >> > (d_Aarray, d_AIJs->array, A_size, 0, 0, maxn1, n_cols);
        
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, init_rs, 0, 0);
        gridSize = (maxn1 * n_cols + blockSize - 1) / blockSize;
        init_rs << <gridSize, blockSize >> > (d_rs->array, d_id_of_ks, maxn1, maxn1 * n_cols);
        
        alpha = 1;
        beta = -1;
        cublasDgemvBatched(handle, CUBLAS_OP_N, maxn1, maxn2, &alpha, d_Aarray, maxn1, d_mhatsArray, 1, &beta, d_rsArray, 1, n_cols);
        
        cudaMemcpy(rs->array, d_rs->array, sizeof(double) * maxn1 * n_cols, cudaMemcpyDeviceToHost);

        // segmented map reduce to calculate norm
        for (int k = 0; k < n_cols; k++)
        {
            double norm = 0;
            for (int i = 0; i < maxn1; i++)
            {
                if (i < n1s[k])
                {
                    const double val = rs->array[maxn1 * k + i];
                    norm += val * val;
                }
            }
            r_norms[k] = sqrt(norm);
        }

        //for (int k = 0; k < n_cols; k++)
        //    printf("col %d norm: %f\n", k, r_norms[k]);

        above_epsilon = false;
        for (int k = 0; k < n_cols && !above_epsilon; k++)
        {
            above_epsilon |= r_norms[k] > epsilon;
        }
    }

    // scatter to m_k by associating mhats with J, sorting key-value pairs and eliminating non-valid entries (n_cols)
    free(csc_row_ind_M); free(csc_val_M);
    int offset = 0;
    for (int k = 0; k < n_cols; k++)
    {
        csc_col_ptr_M[k] = offset;
        offset += n2s[k];
    }
    csc_col_ptr_M[n_cols] = offset;
    csc_val_M = (double*)malloc(sizeof(double) * offset);
    csc_row_ind_M = (int*)malloc(sizeof(int) * offset);

    for (int k = 0; k < n_cols; k++)
    {
	    std::vector<std::pair <int, double> > vect;

        // Entering values in vector of pairs
        for (int i = 0; i < n2s[k]; i++)
            vect.push_back(std::make_pair(Js[J_size * k + i], mhats->array[maxn2 * k + i]));

        // Using simple sort() function to sort
        sort(vect.begin(), vect.end());

        for (int i = 0; i < n2s[k]; i++)
        {
            csc_row_ind_M[csc_col_ptr_M[k] + i] = vect[i].first;
            csc_val_M[csc_col_ptr_M[k] + i] = vect[i].second;
        }
    }

    // Sorting pairs can be done in parallel, but the code below doesn't work:

    //double* d_csc_val_M_in; cudaMalloc((void**)&d_csc_val_M_in, sizeof(double)* offset);
    //double* d_csc_val_M; cudaMalloc((void**)&d_csc_val_M, offset * sizeof(double));
    //int* d_csc_row_ind_M_in; cudaMalloc((void**)&d_csc_row_ind_M_in, sizeof(int)* offset);
    //int* d_csc_row_ind_M; cudaMalloc((void**)&d_csc_row_ind_M, sizeof(int)* offset);
    //
    //for (int k = 0; k < n_cols; k++)
    //{
    //    cudaMemcpy(&d_csc_val_M_in[csc_col_ptr_M[k]], &mhats->array[maxn2 * k], n2s[k] * sizeof(double), cudaMemcpyHostToDevice);
    //    cudaMemcpy(&d_csc_row_ind_M_in[csc_col_ptr_M[k]], &Js[J_size * k], n2s[k] * sizeof(int), cudaMemcpyHostToDevice);
    //}
    //
    //void* d_temp_storage = NULL;
    //size_t   temp_storage_bytes = 0;
    //cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_csc_row_ind_M_in, d_csc_row_ind_M, d_csc_val_M_in, d_csc_val_M, offset, n_cols, d_Joffsets, d_Joffsets + 1);
    //
	//// Allocate temporary storage
    //cudaMalloc(&d_temp_storage, temp_storage_bytes);
    //// Run sorting operation
    //cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_csc_row_ind_M_in, d_csc_row_ind_M, d_csc_val_M_in, d_csc_val_M, offset, n_cols, d_Joffsets, d_Joffsets + 1);
    //
    //
    //cudaMemcpy(csc_row_ind_M, d_csc_row_ind_M, offset * sizeof(int), cudaMemcpyDeviceToHost);
    //cudaMemcpy(csc_val_M, d_csc_val_M, offset * sizeof(double), cudaMemcpyDeviceToHost);


    // sparse matrix matrix product - identity and norm calculation
    double norm = 0;
    for (int j = 0; j < n_cols; j++)
    {
	    for (int i = 0; i < n_cols; i++)
	    {
            const int A_row_ptr = csr_row_ptr_A[i];
            const int A_row_size = csr_row_ptr_A[i + 1] - A_row_ptr;

            const int M_col_ptr = csc_col_ptr_M[j];
            const int M_col_size = csc_col_ptr_M[j + 1] - M_col_ptr;
            int Mi = 0;
            double acc = 0;

            for (int Aj = 0; Aj < A_row_size; Aj++)
            {
                while (Mi < M_col_size && csc_row_ind_M[M_col_ptr + Mi] < csr_col_ind_A[A_row_ptr + Aj])
                {
                    Mi++;
                }
                if (csc_row_ind_M[M_col_ptr + Mi] == csr_col_ind_A[A_row_ptr + Aj])
                {
                    acc += csc_val_M[M_col_ptr + Mi] * csr_val_A[A_row_ptr + Aj];
                }
            }

            //while (Aj < A_row_size && Mi < M_col_size)
		    //{
			//    while (Aj < A_row_size && csr_col_ind_A[A_row_ptr + Aj] < csc_row_ind_M[M_col_ptr + Mi])
			//    {
			//    	Aj++;
            //        while (Mi < M_col_size && csc_row_ind_M[M_col_ptr + Mi] < csr_col_ind_A[A_row_ptr + Aj])
            //        {
            //            Mi++;
            //        }
			//    }
            //	if (csc_row_ind_M[M_col_ptr + Mi] == csr_col_ind_A[A_row_ptr + Aj])
            //	{
            //		acc += csc_val_M[M_col_ptr + Mi] * csr_val_A[A_row_ptr + Aj];
            //	}
            //	Aj++;
            //    Mi++;
		    //}

            if (i == j)
                acc -= 1;

            norm += acc * acc;
	    }
    }

    norm = sqrt(norm);

    printf("SPAI finished. Error is: %f", norm);

    free_cuda_dyn_arr(d_vs); 
    free_cuda_dyn_arr(d_Qvs); 
    free_cuda_dyn_arr(d_AIJs); 
    free_cuda_dyn_arr(d_AIJtildes); 
    free_cuda_dyn_arr(d_Ahat); 
    free_cuda_dyn_arr(d_Anews); 
    free_cuda_dyn_arr(d_B2s);

    //free(A_hats_inds); free(A_hats);
    free(n1s); free(Js);
    //free(n2s); free(Is); free(Is_inds);
    free(csc_col_ptr_A); free(csc_val_A); free(csc_row_ind_A);
    free(csr_row_ptr_A); free(csr_val_A); free(csr_col_ind_A);
    free(csc_col_ptr_M); free(csc_val_M); free(csc_row_ind_M);
    cublasDestroy_v2(handle);
    return 0;
}