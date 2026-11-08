#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <cstddef>
#include <ctime>
#include <stdexcept>
#include <string>
#include <mpi.h>


using matrix = std::vector<std::vector<int>>;


matrix multiply(const matrix& A, const matrix& B) {
	size_t rows = A.size();
	size_t n = B.size();
	matrix C(rows, std::vector<int>(n, 0));
	for (size_t i = 0; i < rows; ++i) {
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


size_t get_rows(size_t N, int p, int size) {
	return (N / size) + (static_cast<size_t>(p) < (N % size) ? 1 : 0);
}


int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);

	int rank;
	int size;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	size_t N = (argc > 1) ? std::atoi(argv[1]) : 1000;

	matrix A;
	matrix C;
	matrix B(N, std::vector<int>(N));

	if (rank == 0) {
		A = generate_matrix(N);
		B = generate_matrix(N);
		C.resize(N, std::vector<int>(N));

		write_matrix("assets\\data\\matrix_A", A);
		write_matrix("assets\\data\\matrix_B", B);
	}


	double start = MPI_Wtime();
	
	for (int i = 0; i < N; ++i) {
		MPI_Bcast(B[i].data(), N, MPI_INT, 0, MPI_COMM_WORLD);
	}

	matrix loc_A;
	for (int i = 0; i < N; ++i) {
		int proc_number = i % size;

		if (proc_number == 0 && rank == 0) {
			loc_A.push_back(A[i]);
		}
		else if (rank == 0) {
			MPI_Send(A[i].data(), N, MPI_INT, proc_number, 0, MPI_COMM_WORLD);
		}
		else if (proc_number == rank) {
			std::vector<int> row(N);
			MPI_Recv(row.data(), N, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			loc_A.push_back(row);
		}
	}

	matrix loc_C = multiply(loc_A, B);

	int loc_index = 0;
	for (int i = 0; i < N; ++i) {
		int proc_number = i % size;

		if (proc_number == 0 && rank == 0) {
			C[i] = loc_C[loc_index++];
		}
		else if (proc_number == rank) {
			MPI_Send(loc_C[loc_index++].data(), N, MPI_INT, 0, 1, MPI_COMM_WORLD);
		}
		else if (rank == 0) {
			MPI_Recv(C[i].data(), N, MPI_INT, proc_number, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
	}

	double end = MPI_Wtime();

	if (rank == 0) {
		write_matrix("assets\\data\\matrix_C", C);

		std::cout << "Количество ядер - " << size << "  ";
		std::cout << "Размерность матриц - " << N << "  ";
		std::cout << "Длительность вычислений - " << (end - start) << "сек.  " << std::endl;
	}

	MPI_Finalize();
	//int rc = std::system("python assets\\scripts\\main.py");
}