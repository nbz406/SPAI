
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cub/device/device_segmented_radix_sort.cuh"
#include "cub/device/device_radix_sort.cuh"
#include "cub/device/device_select.cuh"

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

__global__ void set_offsets(int* offsets, int size, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n)
        offsets[idx] = size * idx;
}

void launchset_offsets(int* array, int size, int n)
{
    int blockSize;   // The launch configurator returned block size 
    int minGridSize; // The minimum grid size needed to achieve the 
                     // maximum occupancy for a full device launch 
    int gridSize;    // The actual grid size needed, based on input size 

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
        set_offsets, 0, 0);
    // Round up according to array size 
    gridSize = (n + blockSize - 1) / blockSize;

    set_offsets <<<gridSize, blockSize>>> (array, size, n);

    cudaDeviceSynchronize();

    // calculate theoretical occupancy
    int maxActiveBlocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks,
        set_offsets, blockSize,
        0);

    int device;
    cudaDeviceProp props;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&props, device);

    float occupancy = (maxActiveBlocks * blockSize / props.warpSize) /
        (float)(props.maxThreadsPerMultiProcessor /
            props.warpSize);

    printf("Launched blocks of size %d. Theoretical occupancy: %f\n",
        blockSize, occupancy);
}

__global__ void set_values_for_key_pair_sort(int* values, int set_size, int n)
{
    const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n)
    {
        values[idx] = idx % set_size;
    }
}

void launchset_set_values_for_key_pair_sort(int* array, int set_size, int n)
{
    int blockSize;   // The launch configurator returned block size 
    int minGridSize; // The minimum grid size needed to achieve the 
                     // maximum occupancy for a full device launch 
    int gridSize;    // The actual grid size needed, based on input size 

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
        set_values_for_key_pair_sort, 0, 0);
    // Round up according to array size 
    gridSize = (n + blockSize - 1) / blockSize;

    set_values_for_key_pair_sort << <gridSize, blockSize >> > (array, set_size, n);

    cudaDeviceSynchronize();

    // calculate theoretical occupancy
    int maxActiveBlocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks,
        set_offsets, blockSize,
        0);

    int device;
    cudaDeviceProp props;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&props, device);

    float occupancy = (maxActiveBlocks * blockSize / props.warpSize) /
        (float)(props.maxThreadsPerMultiProcessor /
            props.warpSize);

    printf("Launched blocks of size %d. Theoretical occupancy: %f\n",
        blockSize, occupancy);
}


int main()
{
#if padded
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
    int minn1 = n_cols;
    int minn2 = n_cols;
    int maxn1 = 0;
    int maxn2 = 0;
    int maxn2tilde = 0;
    int maxn1tilde = 0;


    int* n2s = static_cast<int*>(malloc(sizeof(int) * n_cols));
    int* n2tildes = static_cast<int*>(malloc(sizeof(int) * n_cols));
    int* n1s = static_cast<int*>(malloc(sizeof(int) * n_cols));
    int* n1tildes = static_cast<int*>(malloc(sizeof(int) * n_cols));
    int* l_n1s = static_cast<int*>(malloc(sizeof(int) * n_cols));
    double* A_col_norms = static_cast<double*>(malloc(sizeof(double) * n_cols));

    // A-column norms used later
    for (int k = 0; k < n_cols; k++)
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
    int* colswaps = static_cast<int*>(malloc(sizeof(int) * J_size * n_cols));
    int *d_Js, *d_Jsorteds;
    cudacall(cudaMalloc((void**)&d_Js, sizeof(int) * J_size * n_cols));
    cudacall(cudaMalloc((void**)&d_Jsorteds, sizeof(int) * J_size * n_cols));
    int* d_colswaps, * d_colswaps_in;
    cudacall(cudaMalloc((void**)&d_colswaps, sizeof(int) * J_size * n_cols));
    cudacall(cudaMalloc((void**)&d_colswaps_in, sizeof(int) * J_size * n_cols));


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
                Js[J_size * k + j] = n_cols;
            }
        }
    }
    
    int max_I_size = 0;
    for (int k = 0; k < n_cols; k++)
    {
        // Construct I: nonzero rows of A[:,J]
        int size = 0;
        for (int j = 0; j < n2s[k]; j++)
        {
            const int c_ind = Js[J_size * k + j];
            size += csc_col_ptr_A[c_ind + 1] - csc_col_ptr_A[c_ind];
        }
        if (size > max_I_size)
			max_I_size = size;
    }
    
    int* Is_prev = static_cast<int*>(malloc(sizeof(int) * max_I_size * n_cols));
    int* Isorteds_prev = static_cast<int*>(malloc(sizeof(int) * max_I_size * n_cols));
    int* Ls = static_cast<int*>(malloc(sizeof(int) * (max_I_size * n_cols + n_cols))); // need space for potentially n_cols new entries corresponding to (np.union1d(I,k))
    int* ind_of_col_k_in_I_k = static_cast<int*>(malloc(sizeof(int) * n_cols));

    for (int k = 0; k < n_cols; k++)
    {
        ind_of_col_k_in_I_k[k] = -1; // if I doesn't contain k, then stuff needs to be done
        // min heap to store row index, pointer next row index and number of elements left in column 
        MinHeap<int, cuda::std::pair<int,int>>* mh = init_minheap<int, cuda::std::pair<int, int>>(n2s[k]);

        for (int j = 0; j < n2s[k]; j++)
        {
            const int c_ind = Js[J_size * k + j];
            const int c_ptr = csc_col_ptr_A[c_ind];
            const int col_size = csc_col_ptr_A[c_ind + 1] - c_ptr;

        	insert_minheap<int, cuda::std::pair<int, int>>(mh, csc_row_ind_A[c_ptr], cuda::std::pair<int,int>(c_ptr + 1, col_size - 1));
        }
        int n1 = 0;
        int pos_to_ins_k = 0;
        while (mh->size > 0) {
            //cuda::std::pair<int, cuda::std::pair<int, cuda::std::pair<int, int>>> {1, { 2,3 }};
            cuda::std::pair<int, cuda::std::pair<int, int>> curr = get_min<int, cuda::std::pair<int, int>>(mh);
            delete_minimum<int, cuda::std::pair<int, int>>(mh);

            const int row_ind = curr.first;
            const int ptr = curr.second.first;
            const int elems_left = curr.second.second;
            const int I_start = max_I_size * k;

            if (n1 == 0)
            {
	            Is_prev[I_start] = row_ind;
                Isorteds_prev[I_start] = row_ind;
                if (row_ind == k)// Keep track of index of column k in I_k
                    ind_of_col_k_in_I_k[k] = n1;
            	n1++;
            }
            else if (Is_prev[I_start + n1 - 1] != row_ind)
            {
                Is_prev[I_start + n1] = row_ind;
                Isorteds_prev[I_start + n1] = row_ind;
                if (row_ind == k)
                    ind_of_col_k_in_I_k[k] = n1;
                n1++;
            }

            if (Is_prev[I_start + n1 - 1] <= k && Is_prev[I_start + n1] > k) // k == 0 not accounted for
            {
                pos_to_ins_k = n1;
            }

            // checking if the next element belongs to same array as the current array.
            if (elems_left > 0)
                insert_minheap<int, cuda::std::pair<int, int>>(mh, csc_row_ind_A[ptr], cuda::std::pair<int, int>(ptr + 1, elems_left - 1));
        }
        // copy I[0:ind_of_col_k_in_I_k[k]] ; k ; I[ind_of_col_k_in_I_k[k]:n1] into Ls
        const int ind_of_k = ind_of_col_k_in_I_k[k];
        const int i_start = max_I_size * k;
        const int l_start = i_start + k;
        if (ind_of_k >= 0)
        {
            memcpy(&Ls[l_start], &Is_prev[i_start], n1 * sizeof(int));
            l_n1s[k] = n1;
        }
        else
        {
            memcpy(&Ls[l_start], &Is_prev[i_start], pos_to_ins_k * sizeof(int));
            Ls[l_start + pos_to_ins_k] = k;
            memcpy(&Ls[l_start + pos_to_ins_k + 1], &Is_prev[i_start + pos_to_ins_k], (n1 - pos_to_ins_k) * sizeof(int));
            l_n1s[k] = n1 + 1;
        }
        // fill out rest of Is_prev with large filler vals for sorting
        std::fill(&Is_prev[i_start + n1], &Is_prev[i_start + max_I_size], n_cols);

        free_minheap<int, cuda::std::pair<int, int>>(mh);
    	n1s[k] = n1;
        if (n1 > maxn1)
            maxn1 = n1;
    }

	const int A_size = maxn1 * maxn2;
    int Q_size = maxn1 * maxn1;

    double* A_hats = static_cast<double*>(malloc(sizeof(double) * A_size * n_cols));
    for (int k = 0; k < n_cols; k++)
    {
        // Get indices from csc_row_ind_A starting from the column pointers from csc_col_ptr_A[J]
        // Construct dense A[I,J] in column major format to be used in batched QR decomposition.
        int A_hats_ind = A_size * k;
        for (int j = 0; j < maxn2; j++)
        {
            int col_elem_idx = 0;
            for (int i = 0; i < maxn1; i++)
            {
	            if (i > n1s[k]-1 || j > n2s[k]-1)
	            {
                    A_hats[A_hats_ind + IDX2C(i, j, maxn1)] = 0;
	            }
            	else
	            {
                    const int col_ind = Js[J_size * k + j];
                    const int col_start = csc_col_ptr_A[col_ind];
                    const int i_ind = Is_prev[max_I_size * k + i];
                    while (csc_row_ind_A[col_start + col_elem_idx] < i_ind)
                        col_elem_idx++;
                    if (i_ind == csc_row_ind_A[col_start + col_elem_idx])
                    {
                        A_hats[A_hats_ind + IDX2C(i, j, maxn1)] = csc_val_A[col_start + col_elem_idx];
                        col_elem_idx++;
                    }
                    else
                    {
                        A_hats[A_hats_ind + IDX2C(i, j, maxn1)] = 0;
                    }
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

            // cublasDgemmBatched C = Q, A = Qv, B = vT, alpha = -1
            // Q = Q - QvvT
            for (int j = 0; j < k_to_m; j++)
            {
                for (int i = 0; i < maxn1; i++)
                {
                    Qs[kk * Q_size + IDX2C(i, j + k, maxn1)] -= Qvs[k_to_m * kk + i] * vs[kk * k_to_m + j];
                }
            }
        }
        free(Qvs);
        free(vs);
    }

    // chat_k: index of k in I'th row of Q transposed and first n2 elements.
    double* e_hats = static_cast<double*>(malloc(sizeof(double) * n_cols * maxn2));
	for (int k = 0; k < n_cols; k++)
	{
        const int i = ind_of_col_k_in_I_k[k];
        if (i >= 0) { // if I contains k, otherwise Q^T * e_k[I] = 0
		    for (int j = 0; j < maxn2; j++)
		    {
                e_hats[k * maxn2 + j] = Qs[k * Q_size + IDX2C(i, j, maxn1)];
		    }
        }
		else
        {
            for (int j = 0; j < maxn2; j++)
            {
                e_hats[k * maxn2 + j] = 0;
            }
        }
	}

					cudacall(cudaMalloc((void**)&dA_hat, batch_count * A_size * sizeof(double)));
    double* d_B;    cudacall(cudaMalloc((void**)&d_B, batch_count * maxn2 * sizeof(double)));
    cudacall(cudaMemcpy(dA_hat, A_hats, batch_count * A_size * sizeof(double), cudaMemcpyHostToDevice));
    cudacall(cudaMemcpy(d_B, e_hats, batch_count * maxn2 * sizeof(double), cudaMemcpyHostToDevice));

    double** h_BArray = (double**)malloc(batch_count * sizeof(double*));

    for (int i = 0; i < batch_count; i++)
    {
        h_A_hatArray[i] = dA_hat + i * A_size;
        h_BArray[i] = d_B + i * maxn2;
    }

    double** d_BArray;
    cudaMalloc((void**)&d_Aarray, batch_count * sizeof(double*));
    cudaMalloc((void**)&d_BArray, batch_count * sizeof(double*));

    cudaMemcpy(d_Aarray, h_A_hatArray, batch_count * sizeof(double*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_BArray, h_BArray, batch_count * sizeof(double*), cudaMemcpyHostToDevice);

    double alpha = 1;
    cublascall(cublasDtrsmBatched(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, maxn2, 1, &alpha, d_Aarray, maxn1, d_BArray, maxn2, batch_count));

    // Is it really maxn1 and not maxn2?
    double* mhat_k = static_cast<double*>(malloc(batch_count * maxn2 * sizeof(double)));
    cudacall(cudaMemcpy(A_hats, dA_hat, batch_count * A_size * sizeof(double), cudaMemcpyDeviceToHost));
    cudacall(cudaMemcpy(mhat_k, d_B, batch_count * maxn2 * sizeof(double), cudaMemcpyDeviceToHost));

    // Free all allocations
    cudacall(cudaFree(dA_hat));
    cudacall(cudaFree(d_B));
    cudacall(cudaFree(d_Aarray));
    cudacall(cudaFree(d_BArray));

    double* rs = static_cast<double*>(malloc(sizeof(double) * maxn1 * n_cols));
    double* r_norms = static_cast<double*>(malloc(sizeof(double) * n_cols));
    for (int k = 0; k < n_cols; k++)
    {
        for (int i = 0; i < maxn1; i++)
        {
            rs[maxn1 * k + i] = 0;
        }

        double r_norm = 0;
        for (int j = 0; j < maxn2; j++)
        {
	        int col_elem_idx = 0;
	        for (int i = 0; i < maxn1; i++)
	        {
	        	if (j == n2s[k] - 1 && i < n1s[k])
	        	{
                    const int col_ind = Js[J_size * k + j];
                    const int col_start = csc_col_ptr_A[col_ind];
	        		double res = rs[maxn1 * k + i] + csc_val_A[col_start + col_elem_idx] * mhat_k[maxn2 * k + j];
	        		col_elem_idx++;
	        		if (i == ind_of_col_k_in_I_k[k])
	        			res -= 1;
	        		rs[maxn1 * k + i] = res;
	        		// row finished, now add square to
	        		r_norm += res * res;
	        	}
	        	else if (!(i > n1s[k] - 1 || j > n2s[k] - 1))
	        	{
                    const int col_ind = Js[J_size * k + j];
                    const int col_start = csc_col_ptr_A[col_ind];
	        		const int i_ind = Is_prev[max_I_size * k + i];
	        		if (i_ind == csc_row_ind_A[col_start + col_elem_idx])
	        		{
	        			rs[maxn1 * k + i] += csc_val_A[col_start + col_elem_idx] * mhat_k[maxn2 * k + j];
	        			col_elem_idx++;
	        		}
	        	}
	        }
        }
        r_norms[k] = sqrt(r_norm);
    }
    
    for (int k = 0; k < n_cols; k++)
        printf("col %d norm: %f\n", k, r_norms[k]);

    // while loop begins omg
    bool above_epsilon = false;
    for (int k = 0; k < n_cols; k++)
    {
        above_epsilon |= r_norms[k] > epsilon;
    }

    int iter = 0; 
    while (above_epsilon && iter < max_iter)
    {
        // read each element of A in a coalesced way to get into L3 cache. If cond that cannot be optimized.
        // forall tid = 0 .. num_nonzeros do
        //   (a,i ) = A[tid]
        //   if (nrver_true_cond)
        //     x[i] = a*2

        for (int k = 0; k < n_cols; k++)
        {
            MinHeap<int, cuda::std::pair<int, int>>* mh = init_minheap<int, cuda::std::pair<int, int>>(std::max(l_n1s[k], inds_per_iter)); //   reused later so max of 
            MinHeap<double, int>* rho_j_pairs = init_minheap<double, int>(inds_per_iter);
            
	        const int L_start = max_I_size * k + k;
            for (int l = 0; l < max_I_size + 1; l++)
            {
	            if (l < l_n1s[k])
	            {
                    const int l_ind = Ls[L_start + l];
                    const int r_ptr = csr_row_ptr_A[l_ind];
                    const int row_size = csr_row_ptr_A[l_ind + 1] - r_ptr;
                    insert_minheap<int, cuda::std::pair<int, int>>(mh, csc_row_ind_A[r_ptr], cuda::std::pair<int, int>(r_ptr + 1, row_size   - 1));
	            }
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
                    int l = 0;
                    const int r_size = l_n1s[k];
                    for (int i = 0; i < c_size && l < r_size; i++)
                    {
                        const int row_ind = csc_row_ind_A[c_ptr + i];
                        while (Ls[L_start + l] < row_ind)
                        {
                            l++;
                        }
                        if (Ls[L_start + l] == row_ind)
                        {
                            rTAej += rs[maxn1 * k + l] * csc_val_A[c_ptr + i];
                        }
                    }
                    // Check if calculated correctly for column 1 !!!!!!!!!!!!
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
            }
            int n2tilde = 0;
            while (mh->size > 0)
            {
                const int col_ind = get_min<int, cuda::std::pair<int, int>>(mh).first;
                delete_minimum<int, cuda::std::pair<int, int>>(mh);
                Js[J_size * k + n2s[k] + n2tilde] = col_ind;
                n2tilde++;
            }
            n2tildes[k] = n2tilde;
            if (n2tilde > maxn2tilde)
                maxn2tilde = n2tilde;
            free_minheap<int, cuda::std::pair<int, int>>(mh);
            free_minheap<double,int>(rho_j_pairs);
        }

        // Get Itilde: New inds not in I from shadow of Jtilde
        int max_new_I_size = 0;
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
            if (size > max_new_I_size)
				max_new_I_size = size;
        }

        // maybe keep Itilde/Jtilde in same array as I/J. Jsize = 51. So max I+Itilde = 51 * max column length
        int* Is_new = static_cast<int*>(malloc(sizeof(int) * max_new_I_size * n_cols));

        for (int k = 0; k < n_cols; k++)
        {
            std::memcpy(&Is_new[max_new_I_size * k], &Is_prev[max_I_size * k], n1s[k] * sizeof(int));
            std::fill(&Is_new[max_new_I_size * k + n1s[k]], &Is_new[max_new_I_size * (k + 1)], n_cols);
        }

        free(Is_prev);
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
            const int I_start = max_I_size * k;
            int I_ptr = 0;
            int prev_row = -1;
            while (mh->size > 0) {
                cuda::std::pair<int, cuda::std::pair<int, int>> curr = get_min<int, cuda::std::pair<int, int>>(mh);
                delete_minimum<int, cuda::std::pair<int, int>>(mh);

                const int i = curr.first;
                const int ptr = curr.second.first;
                const int elems_left = curr.second.second;
                const int Inew_start = max_new_I_size * k + n1s[k];

                while (I_ptr < n1s[k] && Isorteds_prev[I_start + I_ptr] < i)
                    I_ptr++;
                if (i != prev_row && i != Isorteds_prev[I_start + I_ptr])
                {
                    Is_new[Inew_start + n1tilde] = i;
                    n1tilde++;
                    if (ind_of_col_k_in_I_k[k] != -1 && i == k)
                        ind_of_col_k_in_I_k[k] = n1s[k] + n1tilde;
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

        int* d_I_offsets, *d_J_offsets;
        cudacall(cudaMalloc((void**)&d_I_offsets, sizeof(int) * (n_cols + 1)));
        cudacall(cudaMalloc((void**)&d_J_offsets, sizeof(int) * (n_cols + 1)));
        launchset_offsets(d_I_offsets, max_new_I_size, n_cols + 1);
        launchset_offsets(d_J_offsets, J_size, n_cols + 1);

        int* d_rowswaps, * d_rowswaps_in;
        cudacall(cudaMalloc((void**)&d_rowswaps, sizeof(int) * max_new_I_size * n_cols));
        cudacall(cudaMalloc((void**)&d_rowswaps_in, sizeof(int) * max_new_I_size * n_cols));
        launchset_set_values_for_key_pair_sort(d_colswaps_in, J_size, J_size * n_cols);
        launchset_set_values_for_key_pair_sort(d_rowswaps_in, max_new_I_size, max_new_I_size* n_cols);

        int* d_Is_new, *d_Isorteds_new;
        cudacall(cudaMalloc((void**)&d_Is_new, sizeof(int) * max_new_I_size * n_cols));
        cudacall(cudaMalloc((void**)&d_Isorteds_new, sizeof(int) * max_new_I_size * n_cols));
        cudaMemcpy(d_Is_new, Is_new, sizeof(int) * max_new_I_size * n_cols, cudaMemcpyHostToDevice);
        cudaMemcpy(d_Js, Js, sizeof(int) * J_size * n_cols, cudaMemcpyHostToDevice);


        // Sort I. First determine temp storage requirements
        void* d_temp_storage = NULL;
        size_t   temp_storage_bytes = 0;
        cub::DeviceSegmentedRadixSort::SortPairs<int, int, int*, int*>(d_temp_storage, temp_storage_bytes, d_Is_new, d_Isorteds_new,    d_rowswaps_in, d_rowswaps, n_cols * max_new_I_size, n_cols, d_I_offsets, d_I_offsets + 1);

        cudacall(cudaMalloc(&d_temp_storage, temp_storage_bytes));

        cub::DeviceSegmentedRadixSort::SortPairs<int, int, int*, int*>(d_temp_storage, temp_storage_bytes, d_Is_new, d_Isorteds_new,    d_rowswaps_in, d_rowswaps, n_cols * max_new_I_size, n_cols, d_I_offsets, d_I_offsets + 1);

        // Sort J.
        cub::DeviceSegmentedRadixSort::SortPairs<int, int, int*, int*>(d_temp_storage, temp_storage_bytes, d_Js, d_Jsorteds,    d_colswaps_in, d_colswaps, n_cols * J_size, n_cols, d_J_offsets, d_J_offsets + 1);

        int* Isorteds_new = static_cast<int*>(malloc(sizeof(int) * max_new_I_size * n_cols));
        int* rowswaps = static_cast<int*>(malloc(sizeof(int) * max_new_I_size * n_cols));
        cudaMemcpy(Isorteds_new, d_Isorteds_new, sizeof(int) * n_cols * max_new_I_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(rowswaps, d_rowswaps, sizeof(int) * n_cols * max_new_I_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(Jsorteds, d_Jsorteds, sizeof(int) * n_cols * J_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(colswaps, d_colswaps, sizeof(int) * n_cols * J_size, cudaMemcpyDeviceToHost);

        // Construct AIJtilde = A[I, Jtilde]
        const int AIJtilde_size = maxn1 * maxn2tilde;
        const int AItildeJtilde_size = maxn1tilde * maxn2tilde;
        double* AIJtildes = static_cast<double*>(malloc(sizeof(double) * AIJtilde_size * n_cols));
        double* AItildeJtildes = static_cast<double*>(malloc(sizeof(double) * AItildeJtilde_size * n_cols));
        for (int k = 0; k < n_cols; k++)
        {
            int AIJtildes_ind = AIJtilde_size * k;
            int AItildeJtildes_ind = AItildeJtilde_size * k;
            for (int j = 0; j < maxn2tilde; j++)
            {
                int col_elem_idx = 0;
                for (int i = 0; i < maxn1; i++)
                {
                    if (i > n1s[k] - 1 || j > n2tildes[k] - 1)
                        AIJtildes[AIJtildes_ind + IDX2C(i, j, maxn1)] = 0;
                    else
                    {
                        const int col_ind = Js[J_size * k + n2s[k] + j]; // Jtilde starts after J in Js
                        const int col_start = csc_col_ptr_A[col_ind];
                        const int i_ind = Is_new[max_new_I_size * k + i];
                        while (csc_row_ind_A[col_start + col_elem_idx] < i_ind)
                            col_elem_idx++;
                        if (i_ind == csc_row_ind_A[col_start + col_elem_idx])
                        {
                            AIJtildes[AIJtildes_ind + IDX2C(i, j, maxn1)] = csc_val_A[col_start + col_elem_idx];
                            col_elem_idx++;
                        }
                        else
                        {
                            AIJtildes[AIJtildes_ind + IDX2C(i, j, maxn1)] = 0;
                        }
                    }
                }
            	col_elem_idx = 0;
                for (int i = 0; i < maxn1tilde; i++)
                {
                    if (i > n1tildes[k] - 1 || j > n2tildes[k] - 1)
                        AItildeJtildes[AItildeJtildes_ind + IDX2C(i, j, maxn1tilde)] = 0;
                    else
                    {
                        const int col_ind = Js[J_size * k + n2s[k] + j]; // Jtilde starts after J in Js
                        const int col_start = csc_col_ptr_A[col_ind];
                        const int i_ind = Is_new[max_new_I_size * k + n1s[k] + i]; // Itilde starts after I in Is_new
                        while (csc_row_ind_A[col_start + col_elem_idx] < i_ind)
                            col_elem_idx++;
                        if (i_ind == csc_row_ind_A[col_start + col_elem_idx])
                        {
                            AItildeJtildes[AItildeJtildes_ind + IDX2C(i, j, maxn1tilde)] = csc_val_A[col_start + col_elem_idx];
                            col_elem_idx++;
                        }
                        else
                        {
                            AItildeJtildes[AItildeJtildes_ind + IDX2C(i, j, maxn1tilde)] = 0;
                        }
                    }
                }
            }
        }

        int maxnewn1 = 0;
        int maxnewn2 = 0;
        for (int k = 0; k < n_cols; k++)
        {
            const int newn1 = n1s[k] + n1tildes[k];
            const int newn2 = n2s[k] + n2tildes[k];
            if (newn1 > maxnewn1)
                maxnewn1 = newn1;
            if (newn2 > maxnewn2)
                maxnewn2 = newn2;
        }
    	const int Anew_size = maxnewn1 * maxnewn2;
        double* Anews = static_cast<double*>(malloc(sizeof(double) * Anew_size * n_cols));
        // Au starts at n2s[k] * maxnewn1
        // B2 starts at n2s[k] * maxnewn1 + (n1s[k]-n2s[k])
        // ld is maxnew1 for both

    	// Au = Q.T * AIJtilde
        const int Au_size = maxn1 * maxn2tilde;
        Q_size = maxn1 * maxn1;
        double* Aus = static_cast<double*>(malloc(sizeof(double) * Au_size * n_cols));

        double* d_Aus, *d_AIJtildes, *d_Qs, *d_Anews;
        cudacall(cudaMalloc((void**)&d_Aus, batch_count * Au_size * sizeof(double))); // result
        cudacall(cudaMalloc((void**)&d_Qs, batch_count * Q_size * sizeof(double))); // left
        cudacall(cudaMalloc((void**)&d_Anews, batch_count * Anew_size * sizeof(double))); // left
        cudacall(cudaMalloc((void**)&d_AIJtildes, batch_count * AIJtilde_size * sizeof(double))); // right

        cudacall(cudaMemcpy(d_Qs, Qs, batch_count * Q_size * sizeof(double), cudaMemcpyHostToDevice));
        cudacall(cudaMemcpy(d_AIJtildes, AIJtildes, batch_count * AIJtilde_size * sizeof(double), cudaMemcpyHostToDevice));

        double** h_QsArray = (double**)malloc(batch_count * sizeof(double*));
        double** h_AIJtildesArray = (double**)malloc(batch_count * sizeof(double*));
        double** h_AusArray = (double**)malloc(batch_count * sizeof(double*));
        double** h_AnewsArray = (double**)malloc(batch_count * sizeof(double*));

        for (int i = 0; i < batch_count; i++)
        {
            h_QsArray[i] = d_Qs + i * Q_size;
            h_AIJtildesArray[i] = d_AIJtildes + i * AIJtilde_size;
            h_AusArray[i] = d_Aus + i * Au_size;
            h_AnewsArray[i] = d_Anews + i * Anew_size + n2s[i] * maxnewn1;
        }

        double** d_QsArray;
    	double** d_AIJtildesArray;
    	double** d_AusArray;
    	double** d_AnewsArray;
    	cudaMalloc((void**)&d_QsArray, batch_count * sizeof(double*));
        cudaMalloc((void**)&d_AIJtildesArray, batch_count * sizeof(double*));
        cudaMalloc((void**)&d_AusArray, batch_count * sizeof(double*));
        cudaMalloc((void**)&d_AnewsArray, batch_count * sizeof(double*));

        cudaMemcpy(d_QsArray, h_QsArray, batch_count * sizeof(double*), cudaMemcpyHostToDevice);
        cudaMemcpy(d_AIJtildesArray, h_AIJtildesArray, batch_count * sizeof(double*), cudaMemcpyHostToDevice);
        cudaMemcpy(d_AusArray, h_AusArray, batch_count * sizeof(double*), cudaMemcpyHostToDevice);
        cudaMemcpy(d_AnewsArray, h_AnewsArray, batch_count * sizeof(double*), cudaMemcpyHostToDevice);

        double alpha = 1, beta = 0;
        cublascall(cublasDgemmBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, maxn1, maxn2tilde, maxn1, &alpha, d_QsArray, maxn1, d_AIJtildesArray, maxn1, &beta, d_AusArray, maxn1, batch_count));

        cublasDgemmBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, maxn1, maxn2tilde, maxn1, &alpha, d_QsArray, maxn1, d_AIJtildesArray, maxn1, &beta, d_AnewsArray, maxnewn1, batch_count);
        
        cudacall(cudaMemcpy(Aus, d_Aus, batch_count * Au_size * sizeof(double), cudaMemcpyDeviceToHost));
        cudacall(cudaMemcpy(Anews, d_Anews, batch_count * Anew_size * sizeof(double), cudaMemcpyDeviceToHost));


        // allocate space for max of n1-n2+n1tilde x maxn2tilde: B2
        int maxB2n = 0;
        for (int k = 0; k < n_cols; k++)
        {
            const int B2n = n1s[k] - n2s[k] + n1tildes[k];
            if (B2n > maxB2n)
                maxB2n = B2n;
        }
        const int B2_size = maxB2n * maxn2tilde;
        double* B2s = static_cast<double*>(malloc(sizeof(double) * B2_size * n_cols));

        for (int k = 0; k < n_cols; k++)
        {
            for (int j = 0; j < maxn2tilde; j++)
            {
                memcpy(&B2s[B2_size * k + j * maxB2n], &Aus[Au_size * k + n2s[k] + j * maxn1], (n1s[k] - n2s[k]) * sizeof(double));
                //memcpy(&B2s[B2_size * k + j * maxB2n + n1s[k] - n2s[k]], &Aus[Au_size * k + n2s[k] + j * maxn1], (n1s[k] - n2s[k]) * sizeof(double));
            }
        }

        printf("S");
        // set new maxn1 and maxn2
        above_epsilon = false;
        for (int k = 0; k < n_cols; k++)
        {
            above_epsilon &= r_norms[k] > epsilon;
        }
    }
#else
	cublasHandle_t handle;
    cublascall(cublasCreate(&handle));

    int n_rows, n_cols, nnz;
    int *csc_col_ptr_A, *csc_row_ind_A, *csc_col_ptr_M, *csc_row_ind_M;
    int* csr_row_ptr_A, * csr_col_ind_A;
    double* csc_val_A, *csc_val_M, *csr_val_A;

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

    int* ind_of_col_k_in_I_k = static_cast<int*>(malloc(sizeof(int)*n_cols));
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
    free(n1s); free(Js);
    //free(n2s); free(Is); free(Is_inds);
    free(csc_col_ptr_A); free(csc_val_A); free(csc_row_ind_A);
    free(csr_row_ptr_A); free(csr_val_A); free(csr_col_ind_A);
    free(csc_col_ptr_M); free(csc_val_M); free(csc_row_ind_M);
    cublasDestroy_v2(handle);
    return 0;
}