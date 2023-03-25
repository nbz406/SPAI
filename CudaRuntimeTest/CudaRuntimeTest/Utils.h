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

	static void create_identity_csc(
		int N,
		int*& csc_col_ptr_I, // return value must be freed
		double*& csc_val_I, // return value must be freed
		int*& csc_row_ind_I // return value must be freed)
	); 
	
};

