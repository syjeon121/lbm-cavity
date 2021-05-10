#include "CPU.h"
#include <ctime>
void CPU() {
	LBM_CPU lbm;
	clock_t t0, t1;
	Error_Lp error_Lp;
	float eps = 0.0001;
	float e = 1.0;
	
	int n = 1;
	int print_step = 100;

	cout << endl << "2D Cavity flow for LBM using CPU" << endl << endl;
	cout << "// ================== Physical property =================== //" << endl;
	cout << "Physical length in X direction = " << lbm.Lx << endl;
	cout << "Physical length in Y direction = " << lbm.Ly << endl;
	cout << "Physical velocity at Top boundary = " << lbm.Ux0_p << endl;
	cout << "Physical kinematic viscosity = " << lbm.nu << endl;
	cout << "Reynolds number = " << lbm.Ux0_p*lbm.Lx / lbm.nu << endl;
	cout << "// ======================================================== //" << endl << endl;

	cout << "// ===================== LBM property ===================== //" << endl;
	cout << "(LBM) The number of Node in X direction = " << lbm.nx << endl;
	cout << "(LBM) The number of Node in Y direction = " << lbm.ny << endl;
	cout << "(LBM) The velocity at Top boundary = " << lbm.Ux0 << endl;
	cout << "(LBM) kinematic viscosity = " << lbm.nu << endl;
	cout << "Relaxation time = " << lbm.tau << endl;
	cout << "Reynolds number = " << lbm.Ux0*lbm.nx / lbm.nu << endl;
	cout << "// ======================================================== //" << endl << endl;

	cout << "eps = " << eps << endl;
	t0 = clock();
	while (e > eps) {

		lbm.Streaming();
		lbm.BC_bounceback();
		lbm.BC_vel();
		lbm.Collision();

		lbm.Error();
		e = error_Lp.Lp(2, lbm.U, lbm.UN, lbm.nx*lbm.ny);
		if (n%print_step == 0) cout << "Error check : [" << e << "]" << endl;

		lbm.Update();
		n++;
	}
	t1 = clock();

	lbm.Print();
	
	cout << "Computation time : " << double(t1 - t0) / CLOCKS_PER_SEC << "[s]" << endl;
}