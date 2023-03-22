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

void Utils::read_matrix_market_file_col_major_sparse(const char* file_name, int& n_rows, int& n_cols, int* offsets,
	double* vals, int* cols_inds)
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
}

void Utils::read_matrix_market_file_row_major_sparse(const char* file_name, int& n_rows, int& n_cols, int* offsets,
	double* vals, int* row_inds)
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


}

