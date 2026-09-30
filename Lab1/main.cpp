#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <cstddef>
#include <ctime>
#include <stdexcept>
#include <string>

size_t N = 1000;


using matrix = std::vector<std::vector<int>>;


matrix multiply(const matrix& A, const matrix& B) {
	size_t n = A.size();
	matrix C(n, std::vector<int>(n, 0));
	for (size_t i = 0; i < n; ++i) {
		for (size_t k = 0; k < n; ++k) {
			for (size_t j = 0; j < n; ++j) {
				C[i][j] += A[i][k] * B[k][j];
			}
		}
	}
	return C;
}


matrix generate_matrix(size_t n) {
	std::random_device rd;
	std::mt19937 gen(rd());

	std::uniform_int_distribution<int> dist(-100, 100);

	matrix M(n, std::vector<int>(n));
	for (size_t i = 0; i < n; ++i) {
		for (size_t j = 0; j < n; ++j) {
			M[i][j] = dist(gen);
		}
	}
	return M;
}


void write_matrix(const std::string& file, const matrix& M) {
	std::ofstream f(file);
	size_t n = M.size();
	if (!f) {
		throw std::runtime_error("Cannot open " + file);
	}

	f << n << '\n';
	
	for (size_t i = 0; i < n; ++i) {
		for (size_t j = 0; j < n; ++j) {
			if (j == n - 1) {
				f << M[i][j] << '\n';
				continue;
			}
			f << M[i][j] << ' ';
		}
	}
}





int main() {
	matrix A = generate_matrix(N);
	matrix B = generate_matrix(N);

	write_matrix("assets\\data\\matrix_A", A);
	write_matrix("assets\\data\\matrix_B", B);

	std::clock_t start = std::clock();

	matrix C = multiply(A, B);

	std::clock_t end = std::clock();

	write_matrix("assets\\data\\result_matrix_C", C);

	double time = double(end - start) / CLOCKS_PER_SEC;



	std::cout << "Размерность матрицы - " << N << std::endl;
	std::cout << "Длительность вычеслений - " << time << " сек." << std::endl;



	int rc = std::system("python assets\\scripts\\main.py");
}