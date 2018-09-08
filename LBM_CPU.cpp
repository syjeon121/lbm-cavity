#include "LBM_CPU.h"
ofstream fout_CPU("out_CPU.dat");
ofstream fout_CPU_Ux("out_CPU_Ux.dat");
ofstream fout_CPU_Uy("out_CPU_Uy.dat");
ifstream fin_CPU("in_CPU.txt");

LBM_CPU::LBM_CPU()
{
// ============================================================================ //
//  LOAD THE PARAMETERS
// ============================================================================ //
	fin_CPU >> nx;		fin_CPU >> comment;
	fin_CPU >> ny;		fin_CPU >> comment;
	fin_CPU >> Lx;		fin_CPU >> comment;
	fin_CPU >> Ly;		fin_CPU >> comment;
	fin_CPU >> a;		fin_CPU >> comment;
	fin_CPU >> Re;		fin_CPU >> comment;
	fin_CPU >> Ux0;		fin_CPU >> comment;
// ============================================================================ //
	

// ============================================================================ //
//  NEW & CUDAMALLOC
// ============================================================================ //
	is_boundary_node = new int[nx*ny];
	U = new float[nx*ny];
	Ux = new float[nx*ny];
	Uy = new float[nx*ny];
	rho = new float[nx*ny];
	W = new float[nx*ny];
	UN = new float[nx*ny];
	UxN = new float[nx*ny];
	UyN = new float[nx*ny];
	rhoN = new float[nx*ny];
	f = new float[nx*ny*a];
	ftemp = new float[nx*ny*a];
	fN = new float[nx*ny*a];
	feq = new float[nx*ny*a];
	ex = new float[a];
	ey = new float[a];
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
// ============================================================================ //


}

void LBM_CPU::Streaming() {

	for (i = 0; i < nx; i++) {
		for (j = 0; j < ny; j++) {

			in = i - 1;
			ip = i + 1;
			jn = j - 1;
			jp = j + 1;

			if (!is_boundary_node[i + nx*j]) {
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
			else if ((i == 0) && (j > 0 && j < ny - 1)) {				//INLET
				ftemp[i + nx*j + nx*ny * 0] = f[i + nx*j + nx*ny * 0];
				ftemp[ip + nx*j + nx*ny * 1] = f[i + nx*j + nx*ny * 1];
				ftemp[i + nx*jp + nx*ny * 2] = f[i + nx*j + nx*ny * 2];
				ftemp[i + nx*jn + nx*ny * 4] = f[i + nx*j + nx*ny * 4];
				ftemp[ip + nx*jp + nx*ny * 5] = f[i + nx*j + nx*ny * 5];
				ftemp[ip + nx*jn + nx*ny * 8] = f[i + nx*j + nx*ny * 8];
			}
			else if ((i > 0 && i < nx - 1) && (j == ny - 1)) {			//TOP
				ftemp[i + nx*j + nx*ny * 0] = f[i + nx*j + nx*ny * 0];
				ftemp[ip + nx*j + nx*ny * 1] = f[i + nx*j + nx*ny * 1];
				ftemp[in + nx*j + nx*ny * 3] = f[i + nx*j + nx*ny * 3];
				ftemp[i + nx*jn + nx*ny * 4] = f[i + nx*j + nx*ny * 4];
				ftemp[in + nx*jn + nx*ny * 7] = f[i + nx*j + nx*ny * 7];
				ftemp[ip + nx*jn + nx*ny * 8] = f[i + nx*j + nx*ny * 8];
			}
			else if ((i > 0 && i < nx - 1) && (j == 0)) {				//BOTTOM
				ftemp[i + nx*j + nx*ny * 0] = f[i + nx*j + nx*ny * 0];
				ftemp[ip + nx*j + nx*ny * 1] = f[i + nx*j + nx*ny * 1];
				ftemp[i + nx*jp + nx*ny * 2] = f[i + nx*j + nx*ny * 2];
				ftemp[in + nx*j + nx*ny * 3] = f[i + nx*j + nx*ny * 3];
				ftemp[ip + nx*jp + nx*ny * 5] = f[i + nx*j + nx*ny * 5];
				ftemp[in + nx*jp + nx*ny * 6] = f[i + nx*j + nx*ny * 6];
			}
			else if ((i == nx - 1) && (j > 0 && j < ny - 1)) {			//OUTLET
				ftemp[i + nx*j + nx*ny * 0] = f[i + nx*j + nx*ny * 0];
				ftemp[i + nx*jp + nx*ny * 2] = f[i + nx*j + nx*ny * 2];
				ftemp[in + nx*j + nx*ny * 3] = f[i + nx*j + nx*ny * 3];
				ftemp[i + nx*jn + nx*ny * 4] = f[i + nx*j + nx*ny * 4];
				ftemp[in + nx*jp + nx*ny * 6] = f[i + nx*j + nx*ny * 6];
				ftemp[in + nx*jn + nx*ny * 7] = f[i + nx*j + nx*ny * 7];
			}
			else if ((i == 0) && (j == 0)) {							//BOTTOM-LEFT
				ftemp[i + nx*j + nx*ny * 0] = f[i + nx*j + nx*ny * 0];
				ftemp[ip + nx*j + nx*ny * 1] = f[i + nx*j + nx*ny * 1];
				ftemp[i + nx*jp + nx*ny * 2] = f[i + nx*j + nx*ny * 2];
				ftemp[ip + nx*jp + nx*ny * 5] = f[i + nx*j + nx*ny * 5];
			}
			else if ((i == 0) && (j == ny - 1)) {						//TOP-LEFT
				ftemp[i + nx*j + nx*ny * 0] = f[i + nx*j + nx*ny * 0];
				ftemp[ip + nx*j + nx*ny * 1] = f[i + nx*j + nx*ny * 1];
				ftemp[i + nx*jn + nx*ny * 4] = f[i + nx*j + nx*ny * 4];
				ftemp[ip + nx*jn + nx*ny * 8] = f[i + nx*j + nx*ny * 8];
			}
			else if ((i == nx - 1) && (j == ny - 1)) {					//TOP-RIGHT
				ftemp[i + nx*j + nx*ny * 0] = f[i + nx*j + nx*ny * 0];
				ftemp[in + nx*j + nx*ny * 3] = f[i + nx*j + nx*ny * 3];
				ftemp[i + nx*jn + nx*ny * 4] = f[i + nx*j + nx*ny * 4];
				ftemp[in + nx*jn + nx*ny * 7] = f[i + nx*j + nx*ny * 7];
			}
			else if ((i == nx - 1) && (j == 0)) {						//BOTTOM-RIGHT
				ftemp[i + nx*j + nx*ny * 0] = f[i + nx*j + nx*ny * 0];
				ftemp[i + nx*jp + nx*ny * 2] = f[i + nx*j + nx*ny * 2];
				ftemp[in + nx*j + nx*ny * 3] = f[i + nx*j + nx*ny * 3];
				ftemp[in + nx*jp + nx*ny * 6] = f[i + nx*j + nx*ny * 6];
			}
		}
	}

}

void LBM_CPU::BC_bounceback() {
// ============================================================================ //
//  LEFT BOUNDARY (HALF BOUNCEBACK)
// ============================================================================ //
	i = 0;
	for (j = 1; j < ny - 1; j++) {
		ftemp[i + nx*j + nx*ny * 1] = f[i + nx*j + nx*ny * 3];
		ftemp[i + nx*j + nx*ny * 8] = f[i + nx*j + nx*ny * 6];
		ftemp[i + nx*j + nx*ny * 5] = f[i + nx*j + nx*ny * 7];
	}
// ============================================================================ //

// ============================================================================ //
//  RIGHT BOUNDARY (HALF BOUNCEBACK)
// ============================================================================ //
	i = nx - 1;
	for (j = 1; j < ny - 1; j++) {
		ftemp[i + nx*j + nx*ny * 3] = f[i + nx*j + nx*ny * 1];
		ftemp[i + nx*j + nx*ny * 6] = f[i + nx*j + nx*ny * 8];
		ftemp[i + nx*j + nx*ny * 7] = f[i + nx*j + nx*ny * 5];
	}
// ============================================================================ //

// ============================================================================ //
//	BOTTOM BOUNDARY (HALF BOUNCEBACK)
// ============================================================================ //
	j = 0;
	for (i = 1; i < nx - 1; i++) {
		ftemp[i + nx*j + nx*ny * 2] = f[i + nx*j + nx*ny * 4];
		ftemp[i + nx*j + nx*ny * 5] = f[i + nx*j + nx*ny * 7];
		ftemp[i + nx*j + nx*ny * 6] = f[i + nx*j + nx*ny * 8];
	}
// ============================================================================ //

// ============================================================================ //
//	BOTTOM-LEFT CORNER (HALF BOUNCEBACK)
// ============================================================================ //
	i = 0;
	j = 0;

	ftemp[i + nx*j + nx*ny * 1] = f[i + nx*j + nx*ny * 3];
	ftemp[i + nx*j + nx*ny * 2] = f[i + nx*j + nx*ny * 4];
	ftemp[i + nx*j + nx*ny * 5] = f[i + nx*j + nx*ny * 7];
	ftemp[i + nx*j + nx*ny * 6] = f[i + nx*j + nx*ny * 8];
	ftemp[i + nx*j + nx*ny * 8] = f[i + nx*j + nx*ny * 6];
// ============================================================================ //


// ============================================================================ //
//	BOTTOM-RIGHT CORNER (HALF BOUNCEBACK)
// ============================================================================ //
	i = nx - 1;
	j = 0;

	ftemp[i + nx*j + nx*ny * 2] = f[i + nx*j + nx*ny * 4];
	ftemp[i + nx*j + nx*ny * 6] = f[i + nx*j + nx*ny * 8];
	ftemp[i + nx*j + nx*ny * 3] = f[i + nx*j + nx*ny * 1];
	ftemp[i + nx*j + nx*ny * 5] = f[i + nx*j + nx*ny * 7];
	ftemp[i + nx*j + nx*ny * 7] = f[i + nx*j + nx*ny * 5];
// ============================================================================ //
}

void LBM_CPU::BC_vel() {
// ============================================================================ //
//	TOP BOUNDARY (VELOCITY)
// ============================================================================ //
	j = ny - 1;
	for (i = 1; i < nx - 1; i++) {

		rho0 = ftemp[i + nx*j + nx*ny * 0] + ftemp[i + nx*j + nx*ny * 1] + ftemp[i + nx*j + nx*ny * 3]
			+ 2.0*(ftemp[i + nx*j + nx*ny * 2] + ftemp[i + nx*j + nx*ny * 5] + ftemp[i + nx*j + nx*ny * 6]);
		ru = rho0 * Ux0;

		ftemp[i + nx*j + nx*ny * 4] = ftemp[i + nx*j + nx*ny * 2];
		ftemp[i + nx*j + nx*ny * 7] = ftemp[i + nx*j + nx*ny * 5] - (1.0 / 2.0)*ru + (1.0 / 2.0)*(ftemp[i + nx*j + nx*ny * 1] - ftemp[i + nx*j + nx*ny * 3]);
		ftemp[i + nx*j + nx*ny * 8] = ftemp[i + nx*j + nx*ny * 6] + (1.0 / 2.0)*ru - (1.0 / 2.0)*(ftemp[i + nx*j + nx*ny * 1] - ftemp[i + nx*j + nx*ny * 3]);
	}
// ============================================================================ //


// ============================================================================ //
//	TOP-LEFT CORNER (VELOCITY)
// ============================================================================ //
	i = 0;
	j = ny - 1;

	ftemp[i + nx*j + nx*ny * 1] = ftemp[i + nx*j + nx*ny * 3];
	ftemp[i + nx*j + nx*ny * 4] = ftemp[i + nx*j + nx*ny * 2];
	ftemp[i + nx*j + nx*ny * 8] = ftemp[i + nx*j + nx*ny * 6];
	ftemp[i + nx*j + nx*ny * 5] = 0.5 * (rho[(i+1) + nx*(j-1)] - (ftemp[i + nx*j + nx*ny * 0] + ftemp[i + nx*j + nx*ny * 1] + ftemp[i + nx*j + nx*ny * 2]
		+ ftemp[i + nx*j + nx*ny * 3] + ftemp[i + nx*j + nx*ny * 4] + ftemp[i + nx*j + nx*ny * 6] + ftemp[i + nx*j + nx*ny * 8]));
	ftemp[i + nx*j + nx*ny * 7] = ftemp[i + nx*j + nx*ny * 5];
// ============================================================================ //



// ============================================================================ //
//	TOP-RIGHT CORNER (VELOCITY)
// ============================================================================ //
	i = nx - 1;
	j = ny - 1;

	ftemp[i + nx*j + nx*ny * 3] = ftemp[i + nx*j + nx*ny * 1];
	ftemp[i + nx*j + nx*ny * 4] = ftemp[i + nx*j + nx*ny * 2];
	ftemp[i + nx*j + nx*ny * 7] = ftemp[i + nx*j + nx*ny * 5];
	ftemp[i + nx*j + nx*ny * 6] = 0.5 * (rho[(i-1) + nx*(j-1)] - (ftemp[i + nx*j + nx*ny * 0] + ftemp[i + nx*j + nx*ny * 1] + ftemp[i + nx*j + nx*ny * 2]
		+ ftemp[i + nx*j + nx*ny * 3] + ftemp[i + nx*j + nx*ny * 4] + ftemp[i + nx*j + nx*ny * 5] + ftemp[i + nx*j + nx*ny * 7]));
	ftemp[i + nx*j + nx*ny * 8] = ftemp[i + nx*j + nx*ny * 6];
// ============================================================================ //
}

void LBM_CPU::Collision() {

	//Calculation of Macroscopic var 
	for (i = 0; i < nx; i++) {
		for (j = 0; j < ny; j++) {
			Ux[i + nx*j] = 0.0;
			Uy[i + nx*j] = 0.0;
			rho[i + nx*j] = 0.0;

			for (k = 0; k < a; k++) {
				rho[i + nx*j] += ftemp[i + nx*j + nx*ny*k];
				Ux[i + nx*j] += ftemp[i + nx*j + nx*ny*k] * ex[k];
				Uy[i + nx*j] += ftemp[i + nx*j + nx*ny*k] * ey[k];
			}
			Ux[i + nx*j] /= rho[i + nx*j];
			Uy[i + nx*j] /= rho[i + nx*j];
		}
	}


	for (i = 0; i < nx; i++) {
		for (j = 0; j < ny; j++) {
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
	}
}

void LBM_CPU::Error() {

	for (i = 0; i < nx; i++) {
		for (j = 0; j < ny; j++) {
			Ux[i + nx*j] = 0.0;
			Uy[i + nx*j] = 0.0;
			rho[i + nx*j] = 0.0;

			for (k = 0; k < a; k++) {
				rho[i + nx*j] += f[i + nx*j + nx*ny*k];
				Ux[i + nx*j] += f[i + nx*j + nx*ny*k] * ex[k];
				Uy[i + nx*j] += f[i + nx*j + nx*ny*k] * ey[k];
			}
			Ux[i + nx*j] /= rho[i + nx*j];
			Uy[i + nx*j] /= rho[i + nx*j];
			U[i + nx*j] = sqrt(pow(Ux[i + nx*j], 2) + pow(Uy[i + nx*j], 2));
		}
	}

	for (i = 0; i < nx; i++) {
		for (j = 0; j < ny; j++) {
			UxN[i + nx*j] = 0.0;
			UyN[i + nx*j] = 0.0;
			rhoN[i + nx*j] = 0.0;

			for (k = 0; k < a; k++) {
				fN[i + nx*j + nx*ny*k] = ftemp[i + nx*j + nx*ny*k] - (ftemp[i + nx*j + nx*ny*k] - feq[i + nx*j + nx*ny*k]) / tau;

				rhoN[i + nx*j] += fN[i + nx*j + nx*ny*k];
				UxN[i + nx*j] += fN[i + nx*j + nx*ny*k] * ex[k];
				UyN[i + nx*j] += fN[i + nx*j + nx*ny*k] * ey[k];
			}
			UxN[i + nx*j] /= rhoN[i + nx*j];
			UyN[i + nx*j] /= rhoN[i + nx*j];
			UN[i + nx*j] = sqrt(pow(UxN[i + nx*j], 2) + pow(UyN[i + nx*j], 2));
		}
	}
}

void LBM_CPU::Update() {

	for (i = 0; i < nx; i++) {
		for (j = 0; j < ny; j++) {
			Ux[i + nx*j] = 0.0;
			Uy[i + nx*j] = 0.0;
			rho[i + nx*j] = 0.0;

			for (k = 0; k < a; k++) {
				f[i + nx*j + nx*ny*k] = fN[i + nx*j + nx*ny*k];

				rho[i + nx*j] += f[i + nx*j + nx*ny*k];
				Ux[i + nx*j] += f[i + nx*j + nx*ny*k] * ex[k];
				Uy[i + nx*j] += f[i + nx*j + nx*ny*k] * ey[k];
			}
			Ux[i + nx*j] /= rho[i + nx*j];
			Uy[i + nx*j] /= rho[i + nx*j];
			U[i + nx*j] = sqrt(pow(Ux[i + nx*j], 2) + pow(Uy[i + nx*j], 2));
		}
	}

}

void LBM_CPU::Print() {

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
		W[i + nx*j] = (Uy[(i + 1) + nx*j] - 0.0) / (del_y) - (Ux[i + nx*(j + 1)] - Ux[i + nx*(j - 1)]) / (2.0*del_y);
	}
	
	//RIGHT BOUNDARY
	i = nx - 1;
	for (j = 1; j < ny - 1; j++) {
		W[i + nx*j] = (0.0 - Uy[(i - 1) + nx*j]) / (del_y) - (Ux[i + nx*(j + 1)] - Ux[i + nx*(j - 1)]) / (2.0*del_y);
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
	W[i + nx*j] = (Uy[(i + 1) + nx*j] - 0.0) / (del_y) - (0.0 - Ux[i + nx*(j - 1)]) / (del_y);

	//BOTTOM-LEFT CONNER
	i = 0;
	j = 0;
	W[i + nx*j] = (Uy[(i + 1) + nx*j] - 0.0) / (del_y) - (Ux[i + nx*(j + 1)] - 0.0) / (del_y);

	//TOP-RIGHT CONNER
	i = nx - 1;
	j = ny - 1;
	W[i + nx*j] = (0.0 - Uy[(i - 1) + nx*j]) / (del_y) - (0.0 - Ux[i + nx*(j - 1)]) / (del_y);

	//BOTTOM-RIGHT CONNER
	i = nx - 1;
	j = 0;
	W[i + nx*j] = (0.0 - Uy[(i - 1) + nx*j]) / (del_y) - (Ux[i + nx*(j + 1)] - 0.0) / (del_y);
// ============================================================================ //



// ============================================================================ //
//  NORMALIZATION 
// ============================================================================ //
	for (j = 0; j < ny; j++) {
		for (i = 0; i < nx; i++) {
			Ux_p[i + nx*j] = Ux[i + nx*j] / Ux0;
			Uy_p[i + nx*j] = Uy[i + nx*j] / Ux0;
			U_p[i + nx*j] = U[i + nx*j] / Ux0;
			W[i + nx*j] = W[i + nx*j] / Ux0;
		}
	}
// ============================================================================ //






	fout_CPU << endl;
	fout_CPU << "variables = X Y Ux Uy U rho W" << endl;
	fout_CPU << "zone i=" << nx << " j=" << ny << endl;
	for (j = 0; j < ny; j++) {
		for (i = 0; i < nx; i++) {
			fout_CPU << i << "\t" << j << "\t" << Ux_p[i + nx*j] << "\t" << Uy_p[i + nx*j] << "\t"
				<< U_p[i + nx*j] << "\t" << rho[i + nx*j] << "\t" << W[i + nx*j] << endl;
		}
	}

	fout_CPU_Ux << "variables = X Y Ux " << endl;
	i = nx / 2;
	for (j = 0; j < ny; j++) {
		fout_CPU_Ux << i << "\t" << j << "\t" << Ux_p[i + nx*j] << endl;
	}

	fout_CPU_Uy << "variables = X Y Uy " << endl;
	j = ny / 2;
	for (i = 0; i < nx; i++) {
		fout_CPU_Uy << i << "\t" << j << "\t" << Uy_p[i + nx*j] << endl;
	}

}

LBM_CPU::~LBM_CPU()
{
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

