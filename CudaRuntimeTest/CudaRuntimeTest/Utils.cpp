#include "Utils.h"

double* Utils::read_matrix_market_file_col_major(const char* file_name, int& n_rows, int& n_cols)
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

	double* matrix = static_cast<double*>(malloc(sizeof(double) * num_rows * num_cols));
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

void Utils::read_matrix_market_file_col_major_sparse(const char* file_name, int& n_rows, int& n_cols, int& nnz, int*& csc_col_ptr_A,
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

	csc_val_A = static_cast<double*>(malloc(sizeof(double) * nnz));
	csc_col_ptr_A = static_cast<int*>(malloc(sizeof(int) * (n_cols + 1)));
	csc_row_ind_A = static_cast<int*>(malloc(sizeof(int) * nnz));

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

void Utils::create_identity_csc(int N, int*& csc_col_ptr_I, double*& csc_val_I, int*& csc_row_ind_I)
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
