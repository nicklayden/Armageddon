/*
	
	Spherically Symmetric Perfect Fluid Spacetime Simulator
	Nicholas Layden, 2024


	compilation:
		g++ SphericalCollapse.cpp -o sspf -std=c++20 -O2

	run:
		./sspf
	
	The numerical algorithm requires the boost header libraries to be installed, 
	no other external libraries are required :)



*/

// Standard Packages
#include <iostream>
#include <vector>
#include <math.h>
#include <cstdint>
#include <time.h>
#include <fstream>
#include <utility>
#include <algorithm>
#include <boost/math/tools/roots.hpp>
#include <boost/math/interpolators/cardinal_cubic_b_spline.hpp>
#include <boost/math/interpolators/barycentric_rational.hpp>

// Definitions
template<class T>
T InitialDensity(T r);

template<class T>
T GridVelocity(T A, T B, T dt, T dr);

template<class T>
T FriedmannConstraint(T M, T R, T Ai, T Ai1 ,T Api, T Api1, T Rpi, T Rpi1, T Rdi, T dt, T U) ;

template<class T>
T Mdot(T M, T R, T Rp, T p);

template<class T>
T Rdot(T M, T R, T Rp, T B0, T rho);

template<class T>
std::vector<T> gradient(const std::vector<T>& data, T dr);

void save_to_file(const std::vector<std::vector<double>>& data, const std::string& filename);

template<class T>
std::vector<T> integrate_A(std::vector<T>& Pp, std::vector<T>& rho, std::vector<T>& p, T dr);

template<class T>
std::vector<T> integrate_M(std::vector<T>& R, std::vector<T>& Rp, std::vector<T> density, T dr, T kappa);

template<class T>
T Mprime(T R, T Rp, T density, T kappa);

template<class T>
T Aprime(T A, T Pp, T rho, T p);

template<class T>
T InitializeRdot(T A, T B, T R, T M);

template<class T>
T ComputeMdot(T R, T Rdot, T p, T kappa);

template<class T>
T B_0(T r);

template<class T>
T ComputeB(T Rpi, T Rpi1, T Rdi, T Rdi1, T B0, T Ai,T Ai1, T Api, T Api1, T dt);

template<class T>
T spincoef_rho(T A, T Rp, T B, T Rd);


int main() {

	// Variables and Containers
	std::vector<double> r_grid;
	std::vector<double> density;
	std::vector<double> density_pred;
	std::vector<double> pressure;
	std::vector<double> pressure_pred;
	std::vector<double> pressure_prime;
	std::vector<double> pressure_prime_pred;
	std::vector<double> mass;
	std::vector<double> mass_pred;
	std::vector<double> mass_prime;
	std::vector<double> mass_prime_pred;
	std::vector<double> mass_dot;
	std::vector<double> mass_dot_pred;
	std::vector<double> R_metric;
	std::vector<double> R_metric_pred;
	std::vector<double> R_metric_prime;
	std::vector<double> R_metric_prime_pred;
	std::vector<double> R_metric_dot;
	std::vector<double> R_metric_dot_pred;
	std::vector<double> A_metric;
	std::vector<double> A_metric_pred;
	std::vector<double> A_metric_prime;
	std::vector<double> A_metric_prime_pred;
	std::vector<double> B_metric;
	std::vector<double> B_metric_pred;
	std::vector<double> rho_time_slice;
	std::vector<double> central_density;

	// Complete Containers
	std::vector<std::vector<double> > R_metric_full;
	std::vector<std::vector<double> > density_full;
	std::vector<std::vector<double> > pressure_full;
	std::vector<std::vector<double> > A_full;
	std::vector<std::vector<double> > A_metric_prime_full;
	std::vector<std::vector<double> > mass_full;
	std::vector<std::vector<double> > mass_prime_full;
	std::vector<std::vector<double> > mass_dot_full;
	std::vector<std::vector<double> > r_full;
	std::vector<std::vector<double> > R_metric_dot_full;
	std::vector<std::vector<double> > R_metric_prime_full;
	std::vector<std::vector<double> > spincoef_rho_full;
	std::vector<std::vector<double> > central_density_full;

	int N_space;
	int N_temporal;

	std::vector<int> bisection_errors;
	std::uintmax_t Max_iterations = 1000; // maximum iterations for root finding method.

	double r;
	double dr;
	double dt;
	double r_s;
	double omega,kappa,mass_total;
	double sound_velocity;

	// Temporary variables for currying, used in bisection loop
	double Mi,Ri,Rpi,Api,Api1,Rpi1,Rdi,Ai,Ai1;


	// Parameters and constants
	N_space = 200;
	N_temporal = 91;
	r_s = sqrt(1.6);
	dr = r_s/N_space;
	dt = 0.002;
	omega = 1.0/3.0;
	sound_velocity = sqrt(omega);
	kappa = 8.0*M_PI;
	mass_total = 0.221557 * kappa; // from integrating rho(r) = e^(-r^2). from r=0,4.

	// Initial Specified Data
	for (int i = 0; i < N_space; ++i) {
		r = (i+1) * dr;
		r_grid.push_back(r);
		density.push_back(InitialDensity(r));
		pressure.push_back(omega*InitialDensity(r));
		R_metric.push_back(r);
		R_metric_prime.push_back(1.0);
		B_metric.push_back(B_0(r));
	}

	// Initial Derived Data
	pressure_prime = gradient(pressure,dr);
	mass = integrate_M(R_metric,R_metric_prime,density,dr,kappa);
	mass_prime = gradient(mass,dr);
	A_metric = integrate_A(pressure_prime,density,pressure,dr);
	A_metric_prime = gradient(A_metric,dr);

	central_density.push_back(density[0]); // Initial Central Density
	R_metric_dot.push_back(0.0); // Boundary data on Rdot
	mass_dot.push_back(0.0); // Boundary data on Mdot
	for (int i = 1; i < N_space; ++i)
	{
		// R_metric_dot.push_back(InitializeRdot(A_metric[i],B_metric[i],R_metric[i],mass[i]));
		R_metric_dot.push_back(0.0);
		// mass_dot.push_back(ComputeMdot(R_metric[i],R_metric_dot[i],pressure[i],kappa));
		mass_dot.push_back(0.0);
	}

	// Reserve memory for the 'prediction' vectors at each timestep

	mass_dot_pred.reserve(N_space);
	mass_prime_pred.reserve(N_space);
	R_metric_dot_pred.reserve(N_space);
	R_metric_prime_pred.reserve(N_space);
	pressure_pred.reserve(N_space);
	density_pred.reserve(N_space);
	A_metric_pred.reserve(N_space);
	A_metric_prime_pred.reserve(N_space);

	pressure_prime_pred = pressure_prime;
	density_pred = density;
	pressure_pred = pressure;
	mass_pred = mass;
	mass_prime_pred = mass_prime;
	R_metric_pred = R_metric;
	R_metric_prime_pred = R_metric_prime;
	A_metric_pred = A_metric;
	A_metric_prime_pred = A_metric_prime;




	// Append initial data to storage containers
	mass_full.push_back(mass);
	mass_prime_full.push_back(mass_prime);
	mass_dot_full.push_back(mass_dot);
	density_full.push_back(density);
	r_full.push_back(r_grid);
	A_full.push_back(A_metric);
	A_metric_prime_full.push_back(A_metric_prime);
	R_metric_dot_full.push_back(R_metric_dot);
	R_metric_prime_full.push_back(R_metric_prime);
	R_metric_full.push_back(R_metric);

	// Compute and store the spin coefficient rho on the initial time slice.
	for (int i = 0; i < N_space; ++i) {
		rho_time_slice.push_back(spincoef_rho(A_metric[i],R_metric_prime[i],B_metric[i],R_metric_dot[i]));
	}
	spincoef_rho_full.push_back(rho_time_slice);

	/*****************************************************************
	 * 	
	 * 	NUMERICAL SIMULATION
	 * 
	 *****************************************************************/

	for (int j = 0; j < N_temporal; ++j) {
		// Updating equations
		

		/****************************************************************
		 * 
		 *                       PREDICTOR STEP
		 *
		 *
		 ****************************************************************/
		// Step 1: Update R_i and M_i
		// std::cout << "Timestep: " << j << "Got to step 1" << std::endl;
		// Euler Step R 
		for (int i = 0; i < N_space; ++i) {
			// R_metric[i] += dr*R_metric_dot[i];
			R_metric_pred[i] += dt*R_metric_dot[i];
			// mass[i] += dr*mass_dot[i];
			mass_pred[i] += dt*mass_dot[i];
		} // Updating R and M
		// std::cout << mass_pred.size() << " mass size" << std::endl;
		// std::cout << R_metric_pred.size() << " mass size" << std::endl;
		// Step 1.5: Compute dR_i, dM_i
		// std::cout << "Timestep: " << j << "Got to step 1.5" << std::endl;
		R_metric_prime_pred = gradient(R_metric_pred,dr);
		// std::cout << "Timestep: " << j << "Got to step 1.55" << std::endl;
		mass_prime_pred = gradient(mass_pred,dr);

		// Step 2: Update rho_i to rho_i+1
		// std::cout << "Timestep: " << j << "Got to step 2" << std::endl;
		for (int i = 0; i < N_space; ++i) {
			density_pred[i] = 2*mass_prime_pred[i]/(kappa* R_metric_pred[i]*R_metric_pred[i] * R_metric_prime_pred[i]);
		}

		// Step 3: Compute p_i+1, and p'_i+1
		// std::cout << "Timestep: " << j << "Got to step 3" << std::endl;
		for (int i = 0; i < N_space; ++i) {
			pressure_pred[i] = omega * density_pred[i];
		}
		pressure_prime_pred = gradient(pressure,dr);

		// Step 4: Solve A'/A equation to determine A_i+1
		A_metric_pred = integrate_A(pressure_prime_pred, density_pred, pressure_pred, dr);
		A_metric_prime_pred = gradient(A_metric_pred,dr);

		// Step 5: Solve Friedmann Constraint via Bisection for Rdot_i+1
		// NEW TRICK! -- Use a lambda expression to "curry" the function
		// that we pass to a bisection method!
		// If the bisection method fails (usually near r=0), then we construct an
		// interpolator, and 'fill in' the problematic data near the origin.
		R_metric_dot_pred[0] = 0.0; // Boundary condition for Rdot.

		for (int i = 1; i < N_space; ++i)
		{
			Mi = mass_full[j][i]; 
			Ri = R_metric_full[j][i];
			Ai = A_full[j][i];
			Ai1= A_metric_pred[i];
			Api1 = A_metric_prime_pred[i];
			Api = A_full[j][i];
			Rpi = R_metric_prime_full[j][i];
			Rpi1= R_metric_prime_pred[i];
			Rdi = R_metric_dot_full[j][i];

			try {
				std::pair<double, double> U = boost::math::tools::bracket_and_solve_root(
				[Mi,Ri,Ai,Ai1,Api,Api1,Rpi,Rpi1,Rdi,dt](double x) {
					return FriedmannConstraint(Mi,Ri,Ai,Ai1,Api,Api1,Rpi,Rpi1,Rdi,dt,x);
				},// End lambda expression
				-1.0, // initial guess
				2.0, // factor to reduce grid by
				true, // function is rising
				boost::math::tools::eps_tolerance<double>(),Max_iterations); // termination conditions

				R_metric_dot_pred[i] = cbrt((pow((U.first),3.) + pow(U.second,3.))/2.); // cubic average of bracketed roots
			} catch(...) {
				// Because of the numerical issue near r=0, the bisection method might fail,
				// Thus an exception needs to be caught.
				// Consider using interpolation to fill in the problematic values, since
				// Rdot=0 at r=0.
				// std::cout << "Issue with bisection on iteration: " << i  << " " << j<< std::endl;
				bisection_errors.push_back(i);
			}
		}	

		// Use an interpolation method to fix the issue near r=0, which causes artifacts in Rdot,
		// where it should smoothly converge to zero at r=0.
		if (j%15 == 0)
		{
			std::cout << "Timestep: " << j << "\r" << std::flush;
		}
		if (bisection_errors.size() > 0)
		{
			std::cout << "Bisection Errors: " << bisection_errors.size() << " " << j << std::endl;
			if (bisection_errors.size()*2 < N_space)
			{   
				// We'll use 2x the number of error points to create the interpolant.
				// Experiment with different values.
				bisection_errors.push_back(2*bisection_errors.size());
			}
			std::vector<double> x; // containers for the interpolant grid (x,y)
			std::vector<double> y;
			x.push_back(0.0); // For Rdot, the origin r=0 always has Rdot=0
			y.push_back(0.0);
			int offset = bisection_errors[bisection_errors.size()-1];
			if (2*offset < N_space) {

				// x and y values where the interpolant is defined
				for (int i = 0; i < 2*offset; ++i) {
					x.push_back(r_grid[i + offset]);
					y.push_back(R_metric_dot_pred[i + offset]);
				}
				// Construct the interpolating spline (cubic spline is pretty bad!)
				// boost::math::interpolators::cardinal_cubic_b_spline<double> spline(y.begin(),y.end(),0,dr);
				boost::math::interpolators::barycentric_rational<double> interpolant(std::move(x),std::move(y),4);

				for (int i = 1; i < bisection_errors[bisection_errors.size()-1]; ++i) {
					R_metric_dot_pred[i] = interpolant(r_grid[i]);
				}
				// release the data from the interpolant (destroying the interpolant)
				interpolant.return_x();
				interpolant.return_y();
				x.clear();
				y.clear();

			} // End if 

		} // End interpolation for Rdot

		// Step 6: Compute Mdot_i+1 from Rdot_i+1
		mass_dot_pred[j] = 0.0;
		for (int i = 1; i < N_space; ++i) {
			mass_dot_pred[i] = ComputeMdot(R_metric_pred[i],R_metric_dot_pred[i],pressure_pred[i],kappa);
		}

		// END PREDICTOR STEP FOR M_i and R_i


		/****************************************************************
		 * 
		 *                       CORRECTOR STEP
		 *
		 *
		 ****************************************************************/
		// Step 1: Update R_i and M_i
		// Trapezoid Correction Step R and M 
		for (int i = 0; i < N_space; ++i) {
			R_metric[i] += (dt/2.0) * (R_metric_dot[i] + R_metric_dot_pred[i]);
			mass[i] += (dt/2.0) * (mass_dot[i] + mass_dot_pred[i]);
		} // Updating R and M

		// Step 1.5: Compute dR_i, dM_i
		R_metric_prime = gradient(R_metric,dr);
		mass_prime = gradient(mass,dr);

		// Step 2: Update rho_i to rho_i+1
		for (int i = 0; i < N_space; ++i) {
			density[i] = 2*mass_prime[i]/(kappa* R_metric[i]*R_metric[i] * R_metric_prime[i]);
		}

		// Step 3: Compute p_i+1, and p'_i+1
		for (int i = 0; i < N_space; ++i) {
			pressure[i] = omega * density[i];
		}
		pressure_prime = gradient(pressure,dr);

		// Step 4: Solve A'/A equation to determine A_i+1
		A_metric = integrate_A(pressure_prime, density, pressure, dr);
		A_metric_prime = gradient(A_metric,dr);

		// Step 5: Solve Friedmann Constraint via Bisection for Rdot_i+1
		// NEW TRICK! -- Use a lambda expression to "curry" the function
		// that we pass to a bisection method!
		// If the bisection method fails (usually near r=0), then we construct an
		// interpolator, and 'fill in' the problematic data near the origin.
		R_metric_dot[0] = 0.0; // Boundary condition for Rdot.

		for (int i = 1; i < N_space; ++i)
		{
			Mi = mass_full[j][i]; 
			Ri = R_metric_full[j][i];
			Ai = A_full[j][i];
			Ai1= A_metric[i];
			Api1 = A_metric_prime[i];
			Api = A_full[j][i];
			Rpi = R_metric_prime_full[j][i];
			Rpi1= R_metric_prime[i];
			Rdi = R_metric_dot_full[j][i];

			try {
				std::pair<double, double> U = boost::math::tools::bracket_and_solve_root(
				[Mi,Ri,Ai,Ai1,Api,Api1,Rpi,Rpi1,Rdi,dt](double x) {
					return FriedmannConstraint(Mi,Ri,Ai,Ai1,Api,Api1,Rpi,Rpi1,Rdi,dt,x);
				},// End lambda expression
				-1.0, // initial guess
				2.0, // factor to reduce grid by
				true, // function is rising
				boost::math::tools::eps_tolerance<double>(),Max_iterations); // termination conditions
				R_metric_dot[i] = cbrt((pow((U.first),3.) + pow(U.second,3.))/2.); // average of bracketed roots
			} catch(...) {
				// Because of the numerical issue near r=0, the bisection method might fail,
				// Thus an exception needs to be caught.
				// Consider using interpolation to fill in the problematic values, since
				// Rdot=0 at r=0.
				// std::cout << "Issue with bisection on iteration: " << i  << " " << j<< std::endl;
				bisection_errors.push_back(i);
			}
		}	

		// Use an interpolation method to fix the issue near r=0, which causes artifacts in Rdot,
		// where it should smoothly converge to zero at r=0.
		if (j%15 == 0)
		{
			std::cout << "Timestep: " << j << "\r" << std::flush;
		}
		if (bisection_errors.size() > 0)
		{
			std::cout << "Bisection Errors: " << bisection_errors.size() << " " << j << std::endl;
			if (bisection_errors.size()*2 < N_space)
			{   
				// We'll use 2x the number of error points to create the interpolant.
				// Experiment with different values.
				bisection_errors.push_back(2*bisection_errors.size());
			}
			std::vector<double> x; // containers for the interpolant grid (x,y)
			std::vector<double> y;
			x.push_back(0.0); // For Rdot, the origin r=0 always has Rdot=0
			y.push_back(0.0);
			int offset = bisection_errors[bisection_errors.size()-1];
			if (2*offset < N_space) {

				// x and y values where the interpolant is defined
				for (int i = 0; i < 2*offset; ++i) {
					x.push_back(r_grid[i + offset]);
					y.push_back(R_metric_dot[i + offset]);
				}
				// Construct the interpolating spline (cubic spline is pretty bad!)
				// boost::math::interpolators::cardinal_cubic_b_spline<double> spline(y.begin(),y.end(),0,dr);
				boost::math::interpolators::barycentric_rational<double> interpolant(std::move(x),std::move(y),4);

				for (int i = 1; i < bisection_errors[bisection_errors.size()-1]; ++i) {
					R_metric_dot[i] = interpolant(r_grid[i]);
				}
				// release the data from the interpolant (destroying the interpolant)
				interpolant.return_x();
				interpolant.return_y();
				x.clear();
				y.clear();

			} // End if 

		} // End interpolation for Rdot

		// Step 6: Compute Mdot_i+1 from Rdot_i+1
		mass_dot[0] = 0.0;
		for (int i = 1; i < N_space; ++i) {
			mass_dot[i] = ComputeMdot(R_metric[i],R_metric_dot[i],pressure[i],kappa);
		}

		// compute and store the spin coefficient rho at the current timestep.
		for (int i = 0; i < N_space; ++i) {
			rho_time_slice[i] = spincoef_rho(A_metric[i],R_metric_prime[i],B_metric[i],R_metric_dot[i]);
		}
		// Compute Grid Velocity
		double minGV = 1e5;
		for (int i = 0; i < N_space; ++i)
		{
			double B = ComputeB(
				R_metric_prime[i],
				R_metric_prime_full[j][i],
				R_metric_dot_full[j][i],
				R_metric_dot[i],
				B_0(i*dr),
				A_full[j][i],
				A_metric[i],
				A_metric_prime_full[j][i],
				A_metric_prime[i],
				dt
				);
			double gridV= GridVelocity(A_metric[i],B,dt,dr);
			minGV = std::min(minGV,gridV);
		}
		// std::cout << "Smallest Grid velocity: " << minGV << std::endl;
		if ( j >1 && sound_velocity > minGV)
		{	
			// std::cout << "Sound Velocity: " << sound_velocity << std::endl;
			// std::cout << "Grid Velocity : " << minGV << std::endl;
			// std::cout << "Sound velocity exceeds grid velocity, evolution unstable..." << std::endl;
			// std::cout << "Stopping evolution at timestep: " << j << std::endl;
		}

		if (j>1) {
			// Update the central density using first order finite differences
			double curr_cent_dens = central_density[j-2];
			std::cout<< "Central density: " << curr_cent_dens << std::endl;
			central_density.push_back(curr_cent_dens - (dt/dr)*(1. + omega)*curr_cent_dens* ( 4.*R_metric_dot[1] - 3.*R_metric_dot[2] + (4./3.)*R_metric_dot[3]- (1./4.)* R_metric_dot[4] )  );			
		}









		// Step 7: Store data from current timestep
		R_metric_full.push_back(R_metric);
		R_metric_prime_full.push_back(R_metric_prime);
		R_metric_dot_full.push_back(R_metric_dot);
		mass_full.push_back(mass);
		mass_prime_full.push_back(mass_prime);
		mass_dot_full.push_back(mass_dot);
		A_full.push_back(A_metric);
		A_metric_prime_full.push_back(A_metric_prime);
		pressure_full.push_back(pressure);
		density_full.push_back(density);
		spincoef_rho_full.push_back(rho_time_slice);
		

	} // Time loop

	std::cout << "Sound velocity: " << sound_velocity << std::endl;
	std::cout << "Saving data to file." << std::endl;
	central_density_full.push_back(central_density);


	// Save Data to File
	save_to_file(A_full,"Results/A_metric.txt");
	save_to_file(A_metric_prime_full,"Results/A_metric_prime.txt");
	save_to_file(density_full,"Results/Density.txt");
	save_to_file(r_full,"Results/r_grid.txt");
	save_to_file(mass_full,"Results/Mass.txt");
	save_to_file(mass_prime_full,"Results/Mass_prime.txt");
	save_to_file(R_metric_dot_full,"Results/R_dot.txt");
	save_to_file(R_metric_prime_full,"Results/R_prime.txt");
	save_to_file(R_metric_full,"Results/R_metric.txt");
	save_to_file(spincoef_rho_full,"Results/rho.txt");
	save_to_file(mass_dot_full,"Results/mass_dot.txt");
	save_to_file(central_density_full,"Results/CentralDensity.txt");

} // End Main

/**************************
 *
 * Function Definitions
 *
 *
 **************************/
template<class T>
T InitialDensity(T r) {
	// return (1.0/(16.0*M_PI))*exp(-4.0*r*r);
	return exp(-4.0*r*r);
}

template<class T>
std::vector<T> gradient(const std::vector<T>& data, T dr) {
    std::vector<T> derivative(data.size());
    // Central difference formula for interior points
    for (size_t i = 1; i < data.size() - 1; ++i) {
        derivative[i] = (data[i + 1] - data[i - 1]) / (2 * dr);
    }
    // For the endpoints, use forward and backward difference
    derivative[0] = (data[1] - data[0]) / dr;
    derivative[data.size() - 1] = (data[data.size() - 1] - data[data.size() - 2]) / dr;
    return derivative;
}

template<class T>
T Mprime(T R, T Rp, T density, T kappa) {
	return kappa*density*Rp*R*R/2.0;
}

void save_to_file(const std::vector<std::vector<double>>& data, const std::string& filename) {
    std::ofstream file(filename);
    if (file.is_open()) {
    	// Add header information to file


    	// Add data to file
        for (const auto& row : data) {
            for (const auto& element : row) {
                file << element << ' ';
            }
            file << '\n';
        }
        file.close();
    } else {
        std::cout << "Unable to open file";
    }
}

template<class T>
std::vector<T> integrate_M(std::vector<T>& R, std::vector<T>& Rp, std::vector<T> density, T dr, T kappa) {
	// Integrates the mass from 0 to r, with initial data M(0)=0.
	std::vector<T> mass;
	T k1,k2,k3,k4; // Runge-Kutta slopes
	T R_half;


	mass.push_back(0.0); // boundary (initial) condition
	
	for (int i = 1; i < R.size(); ++i) {
		// Need R', rho at half intervals.
		// Do simple arithmetic mean
		// From Baumgarte: Cubic average for R,Rdot is better.
		k1 = Mprime(R[i-1],Rp[i-1],density[i-1],kappa);
		k2 = Mprime(R[i-1] + k1*dr/2, (Rp[i-1] + Rp[i])/2, (density[i-1] + density[i])/2, kappa);
		k3 = Mprime(R[i-1] + k2*dr/2, (Rp[i-1] + Rp[i])/2, (density[i-1] + density[i])/2, kappa);
		k4 = Mprime(R[i-1] + k3*dr, Rp[i-1] + dr, density[i-1] + dr, kappa);
		mass.push_back(mass[i-1] + (dr/6) * (k1 + 2*k2 + 2*k3 + k4));
	}
	return mass;

}

template<class T>
T Aprime(T A, T Pp, T rho, T p) {
	// The conservation equation A' = -A p'/(rho + p)
	return -A*Pp/(rho + p);
}

template<class T>
std::vector<T> integrate_A(std::vector<T>& Pp, std::vector<T>& rho, std::vector<T>& p, T dr) {
	// Integrate the conservation equation A' = -A p'/(rho + p)
	std::vector<T> A(Pp.size());
	T k1,k2,k3,k4; // Runge-Kutta Slopes
	A[Pp.size()-1] = 1.0;

	for (int i = Pp.size()-2; i >= 0; i--)
	{
	
		// Need P', rho, P, at half intervals.
		// Do simple arithmetic mean
		k1 = Aprime(A[i+1],Pp[i+1],rho[i+1],p[i+1]);
		k2 = Aprime(A[i+1]+k1*dr/2,(Pp[i+1] + Pp[i])/2,(rho[i+1]+rho[i])/2,(p[i+1]+p[i])/2);
		k3 = Aprime(A[i+1]+k2*dr/2,(Pp[i+1] + Pp[i])/2,(rho[i+1]+rho[i])/2,(p[i+1]+p[i])/2);
		k4 = Aprime(A[i]+k3*dr,Pp[i],rho[i],p[i]);
		A[i] = A[i+1] - (dr/6)*(k1 + 2*k2 + 2*k3 + k4);
	}
	return A;
}

template<class T>
T FriedmannConstraint(T M, T R, T Ai, T Ai1 ,T Api, T Api1, T Rpi, T Rpi1, T Rdi, T dt, T U) {
	// Estimating the integral with a simple trapezoidal step.
	// Numerically root find to determine x
	T a,b;
	a = 2*M/R;
	b = -1 + exp( (dt/2.) *( (Api/Rpi)/Ai * Rdi + (Api1/Rpi1)/Ai1 * U ));
	return a + b - U*U;
}

template<class T>
T InitializeRdot(T A, T B ,T R, T M) {
	// Equation for Rdot on initial data
	return -A*sqrt(2*M/R - 1 + 1/B);
}

template<class T>
T B_0(T r) { 
	// Function of integration from the field equation for B.
	// Is equivalent to the LTB "/(1 + E)" function.
	return 1.0;
}

template<class T>
T ComputeMdot(T R, T Rdot, T p, T kappa) {
	// Field equation for Mdot
	return -(kappa/2) * R*R * Rdot * p;
}

template<class T>
T spincoef_rho(T A, T Rp, T B, T Rd) {
	// Numerator of spin coefficient rho in invariant frame.
	// Denominator is sqrt(2)ABR
	return -A*Rp + B*Rd;
}

template<class T>
T ComputeB(T Rpi, T Rpi1, T Rdi, T Rdi1, T B0, T Ai,T Ai1, T Api, T Api1, T dt) {
	return Rpi * B0 * exp((dt/2) * ( (Api/Ai)/Rpi * Rdi + (Api1/Ai1)/Rpi1 * Rdi1) );
}

template<class T>
T GridVelocity(T A, T B, T dt, T dr) {
	// This function computes the velocity of the coordinate grid
	// For use in satisfying the Courant condition:
	// v_s < v_grid = B dr/ (A dt)
	return B*dr/(A*dt);
}


















