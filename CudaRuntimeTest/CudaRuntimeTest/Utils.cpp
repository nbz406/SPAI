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

void Utils::create_csr_from_csc(int n_rows, int n_cols, int nnz, int*& csc_col_ptr_A, double*& csc_val_A,
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

void Utils::create_identity_plus_minus_csc(int N, int per_col, int*& csc_col_ptr_I, double*& csc_val_I, int*& csc_row_ind_I)
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

	for (int i = N-per_col, ii = 0; i < N; i++, ii += N + 1 - i)
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

