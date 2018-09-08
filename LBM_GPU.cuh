#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <fstream>
using namespace std;

class LBM_GPU
{
public:
// ============================================================================ //
//  VARIABLES
// ============================================================================ //
	int nx;								//The number of node in X direction
	int ny;								//The number of node in Y direction
	int a;								//The number of direction in D2Q9 model

	float Lx;							//The length in X direction
	float Ly;							//The length in Y direction
	float del_x;						//The grid step in X direction
	float del_y;						//The grid step in Y direction
	float del_t;						//The time step

	float* Ux;							//The macroscopic velocity in X direction
	float* Uy;							//The macroscopic velocity in Y direction
	float* U;							//The macroscopic velocity
	float* rho;							//The macroscopic density
	float* W;							//The vorticity
	float* Ux_p;						//The macroscopic velocity in X direction (Physical property)
	float* Uy_p;						//The macroscopic velocity in Y direction (Physical property)
	float* U_p;							//The macroscopic velocity (Physical property)

	float* UxN;							//The macroscopic velocity in X direction at n + 1 step
	float* UyN;							//The macroscopic velocity in Y direction at n + 1 step
	float* UN;							//The macroscopic velocity at n + 1 step
	float* rhoN;						//The macroscopic density


	float* f;							//The distribution function at n step
	float* ftemp;						//The distribution function at temp step
	float* feq;							//The equilibrium distribution funtion
	float* fN;							//The distribution function at n + 1 step
	float* ex;							//The microscopic velocity in X direction
	float* ey;							//The microscopic velocity in Y direction

	float tau;							//The relaxation time
	float nu;							//The kinematic viscosity
	float Re;							//The Reynolds number

	float Ux0_p;						//The velocity at inlet boundary (Physical property)
	float Ux0;							//The velocity at inlet boundary (LBM property)
//	float rho0;							//The density at inlet boundary
//	float ru;							//rho*vel

	int* is_boundary_node;
	int i, j, k, in, ip, jn, jp;



	int BLOCK_SIZE_X;					//The Size of block in X direction in GPU
	int BLOCK_SIZE_Y;					//The Size of block in Y direction in GPU
	int BLOCK_SIZE_Z;					//The Size of block in Z direction in GPU
	int* d_is_boundary_node;
	float* d_f;							//The distribution function at n step used in GPU
	float* d_ftemp;						//The distribution function at temp step used in GPU
	float* d_feq;
	float* d_fN;
	float* d_Ux;
	float* d_Uy;
	float* d_U;		
	float* d_UN;	
	float* d_rho;
	float* d_ex;
	float* d_ey;
	float* d_UxN;
	float* d_UyN;
	float* d_rhoN;

	char comment[60];
// ============================================================================ //




// ============================================================================ //
//  FUNCTIONS
// ============================================================================ //
	LBM_GPU();
	~LBM_GPU();

	void Streaming();
	void BC_bounceback();
	void BC_vel();
	void Collision();
	void Error();
	void Update();
	void Print();
// =========================================================================== //
};