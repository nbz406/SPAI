#pragma once

#include <stdio.h>
#include <fstream>
#include <iostream>


static class Utils
{
public:
	static double* read_matrix_market_file_col_major(const char* file_name, int& n_rows, int& n_cols);

	static void read_matrix_market_file_col_major_sparse(const char* file_name, int& n_rows, int& n_cols, int* offsets, double* vals, int* cols_inds);
	static void read_matrix_market_file_row_major_sparse(const char* file_name, int& n_rows, int& n_cols, int* offsets, double* vals, int* row_inds);
};

