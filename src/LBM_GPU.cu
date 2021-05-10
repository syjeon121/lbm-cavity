#include "LBM_GPU.cuh"
#include <cmath>

ofstream fout_GPU("out_GPU.dat");
ofstream fout_GPU_Ux("out_GPU_Ux.dat");
ofstream fout_GPU_Uy("out_GPU_Uy.dat");
ifstream fin_GPU("in_GPU.txt");

LBM_GPU::LBM_GPU()
{
// ============================================================================ //
//  LOAD THE PARAMETERS
// ============================================================================ //
	fin_GPU >> nx;				fin_GPU >> comment;
	fin_GPU >> ny;				fin_GPU >> comment;
	fin_GPU >> Lx;				fin_GPU >> comment;
	fin_GPU >> Ly;				fin_GPU >> comment;
	fin_GPU >> a;				fin_GPU >> comment;
	fin_GPU >> Re;				fin_GPU >> comment;
	fin_GPU >> Ux0;				fin_GPU >> comment;
	fin_GPU >> BLOCK_SIZE_X;	fin_GPU >> comment;
	fin_GPU >> BLOCK_SIZE_Y;	fin_GPU >> comment;
	fin_GPU >> BLOCK_SIZE_Z;	fin_GPU >> comment;
// ============================================================================ //

	
// ============================================================================ //
//  NEW & CUDAMALLOC
// ============================================================================ //
	is_boundary_node = new int[nx*ny];	cudaMalloc((void**)&d_is_boundary_node, nx*ny * sizeof(int));
	U = new float[nx*ny];				cudaMalloc((void**)&d_U, nx*ny * sizeof(float));
	Ux = new float[nx*ny];				cudaMalloc((void**)&d_Ux, nx*ny * sizeof(float));
	Uy = new float[nx*ny];				cudaMalloc((void**)&d_Uy, nx*ny * sizeof(float));
	rho = new float[nx*ny];				cudaMalloc((void**)&d_rho, nx*ny * sizeof(float));
	W = new float[nx*ny];
	UN = new float[nx*ny];				cudaMalloc((void**)&d_UN, nx*ny * sizeof(float));
	UxN = new float[nx*ny];				cudaMalloc((void**)&d_UxN, nx*ny * sizeof(float));
	UyN = new float[nx*ny];				cudaMalloc((void**)&d_UyN, nx*ny * sizeof(float));
	rhoN = new float[nx*ny];			cudaMalloc((void**)&d_rhoN, nx*ny * sizeof(float));
	f = new float[nx*ny*a];				cudaMalloc((void**)&d_f, nx*ny*a * sizeof(float));
	ftemp = new float[nx*ny*a];			cudaMalloc((void**)&d_ftemp, nx*ny*a * sizeof(float));
	fN = new float[nx*ny*a];			cudaMalloc((void**)&d_fN, nx*ny*a * sizeof(float));
	feq = new float[nx*ny*a];			cudaMalloc((void**)&d_feq, nx*ny*a * sizeof(float));
	ex = new float[a];					cudaMalloc((void**)&d_ex, a * sizeof(float));
	ey = new float[a];					cudaMalloc((void**)&d_ey, a * sizeof(float));
	U_p = new float[nx*ny];
	Ux_p = new float[nx*ny];
	Uy_p = new float[nx*ny];
// ============================================================================ //


// ============================================================================ //
//  Microscopic velocity
// ============================================================================ //
	ex[0] = 0.0,	ey[0] = 0.0;
	ex[1] = 1.0,	ey[1] = 0.0;
	ex[2] = 0.0,	ey[2] = 1.0;
	ex[3] = -1.0,	ey[3] = 0.0;
	ex[4] = 0.0,	ey[4] = -1.0;
	ex[5] = 1.0,	ey[5] = 1.0;
	ex[6] = -1.0,	ey[6] = 1.0;
	ex[7] = -1.0,	ey[7] = -1.0;
	ex[8] = 1.0,	ey[8] = -1.0;
	cudaMemcpy(d_ex, ex, a * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ey, ey, a * sizeof(float), cudaMemcpyHostToDevice);
// ============================================================================ //



// ============================================================================ //
//  SET BOUNDARY NODE
// ============================================================================ //
	for (i = 0; i < nx; i++) {
		for (j = 0; j < ny; j++) {
			if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1) is_boundary_node[i + nx*j] = 1;
			else is_boundary_node[i + nx*j] = 0;
		}
	}
	cudaMemcpy(d_is_boundary_node, is_boundary_node, nx*ny * sizeof(int), cudaMemcpyHostToDevice);
// ============================================================================ //




// ============================================================================ //
//  INITIAL CONDITION
// ============================================================================ //
	del_x = Lx / (float)nx;
	del_y = Ly / (float)ny;
	del_t = pow(del_x, 2);

	Ux0_p = Ux0 * (del_x / del_t);
	tau = 3.0*(del_t / pow(del_x, 2))*(Ux0_p * Lx / Re) + 0.5;

	nu = (1.0 / 3.0)*(tau - 0.5);

	for (i = 0; i < nx; i++) {
		for (j = 0; j < ny; j++) {

			rho[i + nx*j] = 1.0;

			f[i + nx*j + nx*ny * 0] = (4.0 / 9.0) * rho[i + nx*j];
			f[i + nx*j + nx*ny * 1] = (1.0 / 9.0) * rho[i + nx*j];
			f[i + nx*j + nx*ny * 2] = (1.0 / 9.0) * rho[i + nx*j];
			f[i + nx*j + nx*ny * 3] = (1.0 / 9.0) * rho[i + nx*j];
			f[i + nx*j + nx*ny * 4] = (1.0 / 9.0) * rho[i + nx*j];
			f[i + nx*j + nx*ny * 5] = (1.0 / 36.0) * rho[i + nx*j];
			f[i + nx*j + nx*ny * 6] = (1.0 / 36.0) * rho[i + nx*j];
			f[i + nx*j + nx*ny * 7] = (1.0 / 36.0) * rho[i + nx*j];
			f[i + nx*j + nx*ny * 8] = (1.0 / 36.0) * rho[i + nx*j];
		}
	}
	cudaMemcpy(d_rho, rho, nx*ny * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_f, f, nx*ny*a * sizeof(float), cudaMemcpyHostToDevice);
// ============================================================================ //


}

__global__ 
void Kernel_Streaming(float* f, float* ftemp, int* is_boundary_node, int nx, int ny, int a) {

	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	int k = blockDim.z * blockIdx.z + threadIdx.z;
	if (i >= nx || j >= ny || k >= a) return;

	int in, ip, jn, jp;


	if (!is_boundary_node[i + nx*j]) {

		in = i - 1;
		ip = i + 1;
		jn = j - 1;
		jp = j + 1;

		ftemp[i + nx*j + nx*ny * 0] = f[i + nx*j + nx*ny * 0];
		ftemp[ip + nx*j + nx*ny * 1] = f[i + nx*j + nx*ny * 1];
		ftemp[i + nx*jp + nx*ny * 2] = f[i + nx*j + nx*ny * 2];
		ftemp[in + nx*j + nx*ny * 3] = f[i + nx*j + nx*ny * 3];
		ftemp[i + nx*jn + nx*ny * 4] = f[i + nx*j + nx*ny * 4];
		ftemp[ip + nx*jp + nx*ny * 5] = f[i + nx*j + nx*ny * 5];
		ftemp[in + nx*jp + nx*ny * 6] = f[i + nx*j + nx*ny * 6];
		ftemp[in + nx*jn + nx*ny * 7] = f[i + nx*j + nx*ny * 7];
		ftemp[ip + nx*jn + nx*ny * 8] = f[i + nx*j + nx*ny * 8];
	}
	else if ((i == 0) && (j > 0 && j < ny - 1)) {				//LEFT

		in = i - 1;
		ip = i + 1;
		jn = j - 1;
		jp = j + 1;

		ftemp[i + nx*j + nx*ny * 0] = f[i + nx*j + nx*ny * 0];
		ftemp[ip + nx*j + nx*ny * 1] = f[i + nx*j + nx*ny * 1];
		ftemp[i + nx*jp + nx*ny * 2] = f[i + nx*j + nx*ny * 2];
		ftemp[i + nx*jn + nx*ny * 4] = f[i + nx*j + nx*ny * 4];
		ftemp[ip + nx*jp + nx*ny * 5] = f[i + nx*j + nx*ny * 5];
		ftemp[ip + nx*jn + nx*ny * 8] = f[i + nx*j + nx*ny * 8];
	}
	else if ((i > 0 && i < nx - 1) && (j == ny - 1)) {			//TOP

		in = i - 1;
		ip = i + 1;
		jn = j - 1;
		jp = j + 1;

		ftemp[i + nx*j + nx*ny * 0] = f[i + nx*j + nx*ny * 0];
		ftemp[ip + nx*j + nx*ny * 1] = f[i + nx*j + nx*ny * 1];
		ftemp[in + nx*j + nx*ny * 3] = f[i + nx*j + nx*ny * 3];
		ftemp[i + nx*jn + nx*ny * 4] = f[i + nx*j + nx*ny * 4];
		ftemp[in + nx*jn + nx*ny * 7] = f[i + nx*j + nx*ny * 7];
		ftemp[ip + nx*jn + nx*ny * 8] = f[i + nx*j + nx*ny * 8];
	}
	else if ((i > 0 && i < nx - 1) && (j == 0)) {				//BOTTOM

		in = i - 1;
		ip = i + 1;
		jn = j - 1;
		jp = j + 1;

		ftemp[i + nx*j + nx*ny * 0] = f[i + nx*j + nx*ny * 0];
		ftemp[ip + nx*j + nx*ny * 1] = f[i + nx*j + nx*ny * 1];
		ftemp[i + nx*jp + nx*ny * 2] = f[i + nx*j + nx*ny * 2];
		ftemp[in + nx*j + nx*ny * 3] = f[i + nx*j + nx*ny * 3];
		ftemp[ip + nx*jp + nx*ny * 5] = f[i + nx*j + nx*ny * 5];
		ftemp[in + nx*jp + nx*ny * 6] = f[i + nx*j + nx*ny * 6];
	}
	else if ((i == nx - 1) && (j > 0 && j < ny - 1)) {			//RIGHT

		in = i - 1;
		ip = i + 1;
		jn = j - 1;
		jp = j + 1;

		ftemp[i + nx*j + nx*ny * 0] = f[i + nx*j + nx*ny * 0];
		ftemp[i + nx*jp + nx*ny * 2] = f[i + nx*j + nx*ny * 2];
		ftemp[in + nx*j + nx*ny * 3] = f[i + nx*j + nx*ny * 3];
		ftemp[i + nx*jn + nx*ny * 4] = f[i + nx*j + nx*ny * 4];
		ftemp[in + nx*jp + nx*ny * 6] = f[i + nx*j + nx*ny * 6];
		ftemp[in + nx*jn + nx*ny * 7] = f[i + nx*j + nx*ny * 7];
	}
	else if ((i == 0) && (j == 0)) {							//BOTTOM-LEFT

		in = i - 1;
		ip = i + 1;
		jn = j - 1;
		jp = j + 1;

		ftemp[i + nx*j + nx*ny * 0] = f[i + nx*j + nx*ny * 0];
		ftemp[ip + nx*j + nx*ny * 1] = f[i + nx*j + nx*ny * 1];
		ftemp[i + nx*jp + nx*ny * 2] = f[i + nx*j + nx*ny * 2];
		ftemp[ip + nx*jp + nx*ny * 5] = f[i + nx*j + nx*ny * 5];
	}
	else if ((i == 0) && (j == ny - 1)) {						//TOP-LEFT

		in = i - 1;
		ip = i + 1;
		jn = j - 1;
		jp = j + 1;

		ftemp[i + nx*j + nx*ny * 0] = f[i + nx*j + nx*ny * 0];
		ftemp[ip + nx*j + nx*ny * 1] = f[i + nx*j + nx*ny * 1];
		ftemp[i + nx*jn + nx*ny * 4] = f[i + nx*j + nx*ny * 4];
		ftemp[ip + nx*jn + nx*ny * 8] = f[i + nx*j + nx*ny * 8];
	}
	else if ((i == nx - 1) && (j == ny - 1)) {					//TOP-RIGHT

		in = i - 1;
		ip = i + 1;
		jn = j - 1;
		jp = j + 1;

		ftemp[i + nx*j + nx*ny * 0] = f[i + nx*j + nx*ny * 0];
		ftemp[in + nx*j + nx*ny * 3] = f[i + nx*j + nx*ny * 3];
		ftemp[i + nx*jn + nx*ny * 4] = f[i + nx*j + nx*ny * 4];
		ftemp[in + nx*jn + nx*ny * 7] = f[i + nx*j + nx*ny * 7];
	}
	else if ((i == nx - 1) && (j == 0)) {						//BOTTOM-RIGHT

		in = i - 1;
		ip = i + 1;
		jn = j - 1;
		jp = j + 1;

		ftemp[i + nx*j + nx*ny * 0] = f[i + nx*j + nx*ny * 0];
		ftemp[i + nx*jp + nx*ny * 2] = f[i + nx*j + nx*ny * 2];
		ftemp[in + nx*j + nx*ny * 3] = f[i + nx*j + nx*ny * 3];
		ftemp[in + nx*jp + nx*ny * 6] = f[i + nx*j + nx*ny * 6];
	}
}
void LBM_GPU::Streaming() {

	dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
	dim3 dimGrid((nx + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (ny + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y, (a + BLOCK_SIZE_Z - 1) / BLOCK_SIZE_Z);
	Kernel_Streaming << < dimGrid, dimBlock >> > (d_f, d_ftemp, d_is_boundary_node, nx, ny, a);

}

__global__ 
void Kernel_BC_bounceback(float* f, float* ftemp, int nx, int ny, int a) {

	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	int k = blockDim.z * blockIdx.z + threadIdx.z;
	if (i >= nx || j >= ny || k >= a) return;
	else if ((i == 0) && (j > 0 && j < ny - 1)){						//LEFT
		ftemp[i + nx*j + nx*ny * 1] = f[i + nx*j + nx*ny * 3];
		ftemp[i + nx*j + nx*ny * 8] = f[i + nx*j + nx*ny * 6];
		ftemp[i + nx*j + nx*ny * 5] = f[i + nx*j + nx*ny * 7];
	}
	else if ((i == nx - 1) && (j > 0 && j < ny - 1)) {			//RIGHT
		ftemp[i + nx*j + nx*ny * 3] = f[i + nx*j + nx*ny * 1];
		ftemp[i + nx*j + nx*ny * 6] = f[i + nx*j + nx*ny * 8];
		ftemp[i + nx*j + nx*ny * 7] = f[i + nx*j + nx*ny * 5];
	}
	else if ((i > 0 && i < nx - 1) && (j == 0)) {				//BOTTOM
		ftemp[i + nx*j + nx*ny * 2] = f[i + nx*j + nx*ny * 4];
		ftemp[i + nx*j + nx*ny * 5] = f[i + nx*j + nx*ny * 7];
		ftemp[i + nx*j + nx*ny * 6] = f[i + nx*j + nx*ny * 8];
	}
	else if ((i == 0) && (j == 0)) {							//BOTTOM-LEFT
		ftemp[i + nx*j + nx*ny * 1] = f[i + nx*j + nx*ny * 3];
		ftemp[i + nx*j + nx*ny * 2] = f[i + nx*j + nx*ny * 4];
		ftemp[i + nx*j + nx*ny * 5] = f[i + nx*j + nx*ny * 7];
		ftemp[i + nx*j + nx*ny * 6] = f[i + nx*j + nx*ny * 8];
		ftemp[i + nx*j + nx*ny * 8] = f[i + nx*j + nx*ny * 6];
	}
	else if ((i == nx - 1) && (j == 0)) {						//BOTTOM-RIGHT
		ftemp[i + nx*j + nx*ny * 2] = f[i + nx*j + nx*ny * 4];
		ftemp[i + nx*j + nx*ny * 6] = f[i + nx*j + nx*ny * 8];
		ftemp[i + nx*j + nx*ny * 3] = f[i + nx*j + nx*ny * 1];
		ftemp[i + nx*j + nx*ny * 5] = f[i + nx*j + nx*ny * 7];
		ftemp[i + nx*j + nx*ny * 7] = f[i + nx*j + nx*ny * 5];
	}
}
void LBM_GPU::BC_bounceback() {

	dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
	dim3 dimGrid((nx + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (ny + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y, (a + BLOCK_SIZE_Z - 1) / BLOCK_SIZE_Z);
	Kernel_BC_bounceback << < dimGrid, dimBlock >> > (d_f, d_ftemp, nx, ny, a);

}

__global__ 
void Kernel_BC_vel(float* ftemp, float* rho, float Ux0, int nx, int ny, int a) {

	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	int k = blockDim.z * blockIdx.z + threadIdx.z;
	if (i >= nx || j >= ny || k >= a) return;

	float rho0, ru;

	if ((i > 0 && i < nx - 1) && (j == ny - 1)) {				//TOP
		rho0 = ftemp[i + nx*j + nx*ny * 0] + ftemp[i + nx*j + nx*ny * 1] + ftemp[i + nx*j + nx*ny * 3]
			+ 2.0*(ftemp[i + nx*j + nx*ny * 2] + ftemp[i + nx*j + nx*ny * 5] + ftemp[i + nx*j + nx*ny * 6]);
		ru = rho0 * Ux0;

		ftemp[i + nx*j + nx*ny * 4] = ftemp[i + nx*j + nx*ny * 2];
		ftemp[i + nx*j + nx*ny * 7] = ftemp[i + nx*j + nx*ny * 5] - (1.0 / 2.0)*ru + (1.0 / 2.0)*(ftemp[i + nx*j + nx*ny * 1] - ftemp[i + nx*j + nx*ny * 3]);
		ftemp[i + nx*j + nx*ny * 8] = ftemp[i + nx*j + nx*ny * 6] + (1.0 / 2.0)*ru - (1.0 / 2.0)*(ftemp[i + nx*j + nx*ny * 1] - ftemp[i + nx*j + nx*ny * 3]);
	}
	else if ((i == 0) && (j == ny - 1)) {							//TOP-LEFT
		ftemp[i + nx*j + nx*ny * 1] = ftemp[i + nx*j + nx*ny * 3];
		ftemp[i + nx*j + nx*ny * 4] = ftemp[i + nx*j + nx*ny * 2];
		ftemp[i + nx*j + nx*ny * 8] = ftemp[i + nx*j + nx*ny * 6];
		ftemp[i + nx*j + nx*ny * 5] = 0.5 * (rho[(i + 1) + nx*(j - 1)] - (ftemp[i + nx*j + nx*ny * 0] + ftemp[i + nx*j + nx*ny * 1] + ftemp[i + nx*j + nx*ny * 2]
			+ ftemp[i + nx*j + nx*ny * 3] + ftemp[i + nx*j + nx*ny * 4] + ftemp[i + nx*j + nx*ny * 6] + ftemp[i + nx*j + nx*ny * 8]));
		ftemp[i + nx*j + nx*ny * 7] = ftemp[i + nx*j + nx*ny * 5];
	}
	else if ((i == nx - 1) && (j == ny - 1)) {						//TOP-RIGHT
		ftemp[i + nx*j + nx*ny * 3] = ftemp[i + nx*j + nx*ny * 1];
		ftemp[i + nx*j + nx*ny * 4] = ftemp[i + nx*j + nx*ny * 2];
		ftemp[i + nx*j + nx*ny * 7] = ftemp[i + nx*j + nx*ny * 5];
		ftemp[i + nx*j + nx*ny * 6] = 0.5 * (rho[(i - 1) + nx*(j - 1)] - (ftemp[i + nx*j + nx*ny * 0] + ftemp[i + nx*j + nx*ny * 1] + ftemp[i + nx*j + nx*ny * 2]
			+ ftemp[i + nx*j + nx*ny * 3] + ftemp[i + nx*j + nx*ny * 4] + ftemp[i + nx*j + nx*ny * 5] + ftemp[i + nx*j + nx*ny * 7]));
		ftemp[i + nx*j + nx*ny * 8] = ftemp[i + nx*j + nx*ny * 6];
	}

}
void LBM_GPU::BC_vel() {

	dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
	dim3 dimGrid((nx + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (ny + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y, (a + BLOCK_SIZE_Z - 1) / BLOCK_SIZE_Z);
	Kernel_BC_vel << < dimGrid, dimBlock >> > (d_ftemp, d_rho, Ux0, nx, ny, a);
}

__global__ 
void Kernel_Eq(float* ftemp, float* feq, float* Ux, float* Uy, float* rho, float* ex, float* ey, int nx, int ny, int a) {

	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	int k = blockDim.z * blockIdx.z + threadIdx.z;
	if (i >= nx || j >= ny || k >= a) return;


	//Calculation of Macroscopic var 
	rho[i + nx*j] = ftemp[i + nx*j + nx*ny * 0] + ftemp[i + nx*j + nx*ny * 1]
		+ ftemp[i + nx*j + nx*ny * 2] + ftemp[i + nx*j + nx*ny * 3] + ftemp[i + nx*j + nx*ny * 4]
		+ ftemp[i + nx*j + nx*ny * 5] + ftemp[i + nx*j + nx*ny * 6] + ftemp[i + nx*j + nx*ny * 7]
		+ ftemp[i + nx*j + nx*ny * 8];

	Ux[i + nx*j] = ftemp[i + nx*j + nx*ny * 1] * ex[1] + ftemp[i + nx*j + nx*ny * 3] * ex[3]
		+ ftemp[i + nx*j + nx*ny * 5] * ex[5] + ftemp[i + nx*j + nx*ny * 6] * ex[6] + ftemp[i + nx*j + nx*ny * 7] * ex[7]
		+ ftemp[i + nx*j + nx*ny * 8] * ex[8];

	Uy[i + nx*j] = ftemp[i + nx*j + nx*ny * 2] * ey[2] + ftemp[i + nx*j + nx*ny * 4] * ey[4]
		+ ftemp[i + nx*j + nx*ny * 5] * ey[5] + ftemp[i + nx*j + nx*ny * 6] * ey[6] + ftemp[i + nx*j + nx*ny * 7] * ey[7]
		+ ftemp[i + nx*j + nx*ny * 8] * ey[8];

	Ux[i + nx*j] /= rho[i + nx*j];
	Uy[i + nx*j] /= rho[i + nx*j];



	feq[i + nx*j + nx*ny * 0] = (4.0 / 9.0) * rho[i + nx*j] * (1.0 - 1.5*(pow(Ux[i + nx*j], 2) + pow(Uy[i + nx*j], 2)));
	feq[i + nx*j + nx*ny * 1] = (1.0 / 9.0) * rho[i + nx*j] * (1.0 + 3.0 * Ux[i + nx*j] + 4.5*pow(Ux[i + nx*j], 2) - 1.5*(pow(Ux[i + nx*j], 2) + pow(Uy[i + nx*j], 2)));
	feq[i + nx*j + nx*ny * 2] = (1.0 / 9.0) * rho[i + nx*j] * (1.0 + 3.0 * Uy[i + nx*j] + 4.5*pow(Uy[i + nx*j], 2) - 1.5*(pow(Ux[i + nx*j], 2) + pow(Uy[i + nx*j], 2)));
	feq[i + nx*j + nx*ny * 3] = (1.0 / 9.0) * rho[i + nx*j] * (1.0 - 3.0 * Ux[i + nx*j] + 4.5*pow(Ux[i + nx*j], 2) - 1.5*(pow(Ux[i + nx*j], 2) + pow(Uy[i + nx*j], 2)));
	feq[i + nx*j + nx*ny * 4] = (1.0 / 9.0) * rho[i + nx*j] * (1.0 - 3.0 * Uy[i + nx*j] + 4.5*pow(Uy[i + nx*j], 2) - 1.5*(pow(Ux[i + nx*j], 2) + pow(Uy[i + nx*j], 2)));
	feq[i + nx*j + nx*ny * 5] = (1.0 / 36.0) * rho[i + nx*j] * (1.0 + 3.0 * (Ux[i + nx*j] + Uy[i + nx*j]) + 4.5*pow(Ux[i + nx*j] + Uy[i + nx*j], 2) - 1.5*(pow(Ux[i + nx*j], 2) + pow(Uy[i + nx*j], 2)));
	feq[i + nx*j + nx*ny * 6] = (1.0 / 36.0) * rho[i + nx*j] * (1.0 + 3.0 * (-Ux[i + nx*j] + Uy[i + nx*j]) + 4.5*pow(-Ux[i + nx*j] + Uy[i + nx*j], 2) - 1.5*(pow(Ux[i + nx*j], 2) + pow(Uy[i + nx*j], 2)));
	feq[i + nx*j + nx*ny * 7] = (1.0 / 36.0) * rho[i + nx*j] * (1.0 + 3.0 * (-Ux[i + nx*j] - Uy[i + nx*j]) + 4.5*pow(-Ux[i + nx*j] - Uy[i + nx*j], 2) - 1.5*(pow(Ux[i + nx*j], 2) + pow(Uy[i + nx*j], 2)));
	feq[i + nx*j + nx*ny * 8] = (1.0 / 36.0) * rho[i + nx*j] * (1.0 + 3.0 * (Ux[i + nx*j] - Uy[i + nx*j]) + 4.5*pow(Ux[i + nx*j] - Uy[i + nx*j], 2) - 1.5*(pow(Ux[i + nx*j], 2) + pow(Uy[i + nx*j], 2)));


}
__global__
void Kernel_Collision(float* fN, float* ftemp, float* feq, int nx, int ny, int a, float tau) {

	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	int k = blockDim.z * blockIdx.z + threadIdx.z;
	if (i >= nx || j >= ny || k >= a) return;


	fN[i + nx*j + nx*ny*k] = ftemp[i + nx*j + nx*ny*k] - (ftemp[i + nx*j + nx*ny*k] - feq[i + nx*j + nx*ny*k]) / tau;

}
void LBM_GPU::Collision() {

	dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
	dim3 dimGrid((nx + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (ny + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y, (a + BLOCK_SIZE_Z - 1) / BLOCK_SIZE_Z);
	Kernel_Eq << < dimGrid, dimBlock >> > (d_ftemp, d_feq, d_Ux, d_Uy, d_rho, d_ex, d_ey, nx, ny, a);
	Kernel_Collision << < dimGrid, dimBlock >> > (d_fN, d_ftemp, d_feq, nx, ny, a, tau);
}

__global__ 
void Kernel_Error(float* ftemp, float* f, float* Ux, float* Uy, float* U, float* rho, float* fN, float* UxN, float* UyN, float* UN, float* rhoN, float* ex, float* ey, int nx, int ny, int a) {
	
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	int k = blockDim.z * blockIdx.z + threadIdx.z;
	if (i >= nx || j >= ny || k >= a) return;


	rho[i + nx*j] = f[i + nx*j + nx*ny * 0] + f[i + nx*j + nx*ny * 1]
		+ f[i + nx*j + nx*ny * 2] + f[i + nx*j + nx*ny * 3] + f[i + nx*j + nx*ny * 4]
		+ f[i + nx*j + nx*ny * 5] + f[i + nx*j + nx*ny * 6] + f[i + nx*j + nx*ny * 7]
		+ f[i + nx*j + nx*ny * 8];

	Ux[i + nx*j] = f[i + nx*j + nx*ny * 1] * ex[1] + f[i + nx*j + nx*ny * 3] * ex[3]
		+ f[i + nx*j + nx*ny * 5] * ex[5] + f[i + nx*j + nx*ny * 6] * ex[6] + f[i + nx*j + nx*ny * 7] * ex[7]
		+ f[i + nx*j + nx*ny * 8] * ex[8];

	Uy[i + nx*j] = f[i + nx*j + nx*ny * 2] * ey[2] + f[i + nx*j + nx*ny * 4] * ey[4]
		+ f[i + nx*j + nx*ny * 5] * ey[5] + f[i + nx*j + nx*ny * 6] * ey[6] + f[i + nx*j + nx*ny * 7] * ey[7]
		+ f[i + nx*j + nx*ny * 8] * ey[8];

	Ux[i + nx*j] /= rho[i + nx*j];
	Uy[i + nx*j] /= rho[i + nx*j];
	U[i + nx*j] = sqrt(pow(Ux[i + nx*j], 2) + pow(Uy[i + nx*j], 2));




	rhoN[i + nx*j] = fN[i + nx*j + nx*ny * 0] + fN[i + nx*j + nx*ny * 1]
		+ fN[i + nx*j + nx*ny * 2] + fN[i + nx*j + nx*ny * 3] + fN[i + nx*j + nx*ny * 4]
		+ fN[i + nx*j + nx*ny * 5] + fN[i + nx*j + nx*ny * 6] + fN[i + nx*j + nx*ny * 7]
		+ fN[i + nx*j + nx*ny * 8];

	UxN[i + nx*j] = fN[i + nx*j + nx*ny * 1] * ex[1] + fN[i + nx*j + nx*ny * 3] * ex[3]
		+ fN[i + nx*j + nx*ny * 5] * ex[5] + fN[i + nx*j + nx*ny * 6] * ex[6] + fN[i + nx*j + nx*ny * 7] * ex[7]
		+ fN[i + nx*j + nx*ny * 8] * ex[8];

	UyN[i + nx*j] = fN[i + nx*j + nx*ny * 2] * ey[2] + fN[i + nx*j + nx*ny * 4] * ey[4]
		+ fN[i + nx*j + nx*ny * 5] * ey[5] + fN[i + nx*j + nx*ny * 6] * ey[6] + fN[i + nx*j + nx*ny * 7] * ey[7]
		+ fN[i + nx*j + nx*ny * 8] * ey[8];

	UxN[i + nx*j] /= rhoN[i + nx*j];
	UyN[i + nx*j] /= rhoN[i + nx*j];
	UN[i + nx*j] = sqrt(pow(UxN[i + nx*j], 2) + pow(UyN[i + nx*j], 2));

}
void LBM_GPU::Error() {

	dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
	dim3 dimGrid((nx + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (ny + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y, (a + BLOCK_SIZE_Z - 1) / BLOCK_SIZE_Z);
	Kernel_Error << < dimGrid, dimBlock >> > (d_ftemp, d_f, d_Ux, d_Uy, d_U, d_rho, d_fN, d_UxN, d_UyN, d_UN, d_rhoN, d_ex, d_ey, nx, ny, a);

	cudaMemcpy(U, d_U, nx*ny * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(UN, d_UN, nx*ny * sizeof(float), cudaMemcpyDeviceToHost);
}

__global__ 
void Kernel_Update(float* fN, float* f, float* Ux, float* Uy, float* U, float* rho, float* ex, float* ey, int nx, int ny, int a) {

	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	int k = blockDim.z * blockIdx.z + threadIdx.z;
	if (i >= nx || j >= ny || k >= a) return;

	f[i + nx*j + nx*ny*k] = fN[i + nx*j + nx*ny*k];

	rho[i + nx*j] = f[i + nx*j + nx*ny * 0] + f[i + nx*j + nx*ny * 1]
		+ f[i + nx*j + nx*ny * 2] + f[i + nx*j + nx*ny * 3] + f[i + nx*j + nx*ny * 4]
		+ f[i + nx*j + nx*ny * 5] + f[i + nx*j + nx*ny * 6] + f[i + nx*j + nx*ny * 7]
		+ f[i + nx*j + nx*ny * 8];

	Ux[i + nx*j] = f[i + nx*j + nx*ny * 1] * ex[1] + f[i + nx*j + nx*ny * 3] * ex[3]
		+ f[i + nx*j + nx*ny * 5] * ex[5] + f[i + nx*j + nx*ny * 6] * ex[6] + f[i + nx*j + nx*ny * 7] * ex[7]
		+ f[i + nx*j + nx*ny * 8] * ex[8];

	Uy[i + nx*j] = f[i + nx*j + nx*ny * 2] * ey[2] + f[i + nx*j + nx*ny * 4] * ey[4]
		+ f[i + nx*j + nx*ny * 5] * ey[5] + f[i + nx*j + nx*ny * 6] * ey[6] + f[i + nx*j + nx*ny * 7] * ey[7]
		+ f[i + nx*j + nx*ny * 8] * ey[8];

	Ux[i + nx*j] /= rho[i + nx*j];
	Uy[i + nx*j] /= rho[i + nx*j];
	U[i + nx*j] = sqrt(pow(Ux[i + nx*j], 2) + pow(Uy[i + nx*j], 2));
}
void LBM_GPU::Update() {


	dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);
	dim3 dimGrid((nx + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (ny + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y, (a + BLOCK_SIZE_Z - 1) / BLOCK_SIZE_Z);
	Kernel_Update << < dimGrid, dimBlock >> > (d_fN, d_f, d_Ux, d_Uy, d_U, d_rho, d_ex, d_ey, nx, ny, a);

}

void LBM_GPU::Print() {

	cudaMemcpy(Ux, d_Ux, nx*ny * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(Uy, d_Uy, nx*ny * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(U, d_U, nx*ny * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(rho, d_rho, nx*ny * sizeof(float), cudaMemcpyDeviceToHost);


// ============================================================================ //
//  VORTICITY
// ============================================================================ //
	//INNER
	for (i = 1; i < nx - 1; i++) {
		for (j = 1; j < ny - 1; j++) {
			W[i + nx*j] = (Uy[(i + 1) + nx*j] - Uy[(i - 1) + nx*j]) / (2.0*del_y) - (Ux[i + nx*(j + 1)] - Ux[i + nx*(j - 1)]) / (2.0*del_y);
		}
	}

	//LEFT BOUNDARY
	i = 0;
	for (j = 1; j < ny - 1; j++) {
		W[i + nx*j] = (Uy[(i + 1) + nx*j] - 0.0) / (del_y)-(Ux[i + nx*(j + 1)] - Ux[i + nx*(j - 1)]) / (2.0*del_y);
	}

	//RIGHT BOUNDARY
	i = nx - 1;
	for (j = 1; j < ny - 1; j++) {
		W[i + nx*j] = (0.0 - Uy[(i - 1) + nx*j]) / (del_y)-(Ux[i + nx*(j + 1)] - Ux[i + nx*(j - 1)]) / (2.0*del_y);
	}

	//TOP BOUNDARY
	j = ny - 1;
	for (i = 1; i < nx - 1; i++) {
		W[i + nx*j] = (Uy[(i + 1) + nx*j] - Uy[(i - 1) + nx*j]) / (2.0*del_y) - (0.0 - Ux[i + nx*(j - 1)]) / (del_y);
	}

	//BOTTOM BOUNDARY
	j = 0;
	for (i = 1; i < nx - 1; i++) {
		W[i + nx*j] = (Uy[(i + 1) + nx*j] - Uy[(i - 1) + nx*j]) / (2.0*del_y) - (Ux[i + nx*(j + 1)] - 0.0) / (del_y);
	}

	//TOP-LEFT CONNER
	i = 0;
	j = ny - 1;
	W[i + nx*j] = (Uy[(i + 1) + nx*j] - 0.0) / (del_y)-(0.0 - Ux[i + nx*(j - 1)]) / (del_y);

	//BOTTOM-LEFT CONNER
	i = 0;
	j = 0;
	W[i + nx*j] = (Uy[(i + 1) + nx*j] - 0.0) / (del_y)-(Ux[i + nx*(j + 1)] - 0.0) / (del_y);

	//TOP-RIGHT CONNER
	i = nx - 1;
	j = ny - 1;
	W[i + nx*j] = (0.0 - Uy[(i - 1) + nx*j]) / (del_y)-(0.0 - Ux[i + nx*(j - 1)]) / (del_y);

	//BOTTOM-RIGHT CONNER
	i = nx - 1;
	j = 0;
	W[i + nx*j] = (0.0 - Uy[(i - 1) + nx*j]) / (del_y)-(Ux[i + nx*(j + 1)] - 0.0) / (del_y);
// ============================================================================ //



// ============================================================================ //
//  NORMALIZATION 
// ============================================================================ //
	for (i = 0; i < nx; i++) {
		for (j = 0; j < ny; j++) {
			Ux_p[i + nx*j] = Ux[i + nx*j] / Ux0;
			Uy_p[i + nx*j] = Uy[i + nx*j] / Ux0;
			U_p[i + nx*j] = U[i + nx*j] / Ux0;
			W[i + nx*j] = W[i + nx*j] / Ux0;
		}
	}
// ============================================================================ //






	fout_GPU << endl;
	fout_GPU << "variables = X Y Ux Uy U rho W" << endl;
	fout_GPU << "zone i=" << nx << " j=" << ny << endl;
	for (j = 0; j < ny; j++) {
		for (i = 0; i < nx; i++) {
			fout_GPU << i << "\t" << j << "\t" << Ux_p[i + nx*j] << "\t" << Uy_p[i + nx*j] << "\t"
				<< U_p[i + nx*j] << "\t" << rho[i + nx*j] << "\t" << W[i + nx*j] << endl;
		}
	}

	fout_GPU_Ux << "variables = X Y Ux " << endl;
	i = nx / 2;
	for (j = 0; j < ny; j++) {
		fout_GPU_Ux << i << "\t" << j << "\t" << Ux_p[i + nx*j] << endl;
	}

	fout_GPU_Uy << "variables = X Y Uy " << endl;
	j = ny / 2;
	for (i = 0; i < nx; i++) {
		fout_GPU_Uy << i << "\t" << j << "\t" << Uy_p[i + nx*j] << endl;
	}

}

LBM_GPU::~LBM_GPU()
{
	cudaFree(d_is_boundary_node);
	cudaFree(d_f);
	cudaFree(d_fN);
	cudaFree(d_ftemp);
	cudaFree(d_feq);
	cudaFree(d_Ux);
	cudaFree(d_Uy);
	cudaFree(d_rho);
	cudaFree(d_ex);
	cudaFree(d_ey);
	cudaFree(d_U);
	cudaFree(d_UN);
	cudaFree(d_UxN);
	cudaFree(d_UyN);
	cudaFree(rhoN);


	delete[] Uy_p;
	delete[] Ux_p;
	delete[] U_p;
	delete[] ey;
	delete[] ex;
	delete[] fN;
	delete[] feq;
	delete[] ftemp;
	delete[] f;
	delete[] rhoN;
	delete[] UyN;
	delete[] UxN;
	delete[] UN;
	delete[] W;
	delete[] rho;
	delete[] Uy;
	delete[] Ux;
	delete[] U;
	delete[] is_boundary_node;
	cout << endl << "Done!" << endl;
}
