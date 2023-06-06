#pragma once

#include <stdio.h>
#include <fstream>
#include <iostream>


class Utils
{
public:
	static double* read_matrix_market_file_col_major(const char* file_name, int& n_rows, int& n_cols);

	static void read_matrix_market_file_col_major_sparse(
		const char* file_name,
		int& n_rows,
		int& n_cols,
		int& nnz,
		int *&csc_col_ptr_A, // return value must be freed
		double *&csc_val_A, // return value must be freed
		int *&csc_row_ind_A // return value must be freed
	);

	static void create_csr_from_csc(
		int n_rows,
		int n_cols,
		int nnz,
		int*& csc_col_ptr_A,
		double*& csc_val_A,
		int*& csc_row_ind_A,
		int*& csr_row_ptr_A, // return value must be freed
		double*& csr_val_A, // return value must be freed
		int*& csr_col_ind_A // return value must be freed
	);

	static void create_identity_csc(
		int N,
		int*& csc_col_ptr_I, // return value must be freed
		double*& csc_val_I, // return value must be freed
		int*& csc_row_ind_I // return value must be freed)
	);

	static void create_identity_plus_minus_csc(
		int N,
		int per_col, int*& csc_col_ptr_I,
		// return value must be freed
		double*& csc_val_I,
		// return value must be freed
		int*& csc_row_ind_I // return value must be freed)
	); 
	
};

typedef struct MinHeap MinHeap;
struct MinHeap {
	int* arr;
	// Current Size of the Heap
	int size;
	// Maximum capacity of the heap
	int capacity;
};