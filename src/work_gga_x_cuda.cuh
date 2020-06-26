/************************************************************************
This file is to be included in GGA exchange functionals. As often these
functionals are written as a function of x = |grad n|/n^(4/3), this
routine performs the necessary conversions between a functional of x
and of rho.
************************************************************************/

// CUDA runtime
#include <cuda_runtime.h>
// helper functions and utilities to work with CUDA
#include "helper/helper_cuda.h"

#include "util.h"
#include "work_utils.h"

#ifndef XC_DIMENSIONS
#define XC_DIMENSIONS 3
#endif

//////////////////////////////////////////////////
// Kernel original
static void __global__
#ifdef XC_KINETIC_FUNCTIONAL
kernel_work_gga_k
#else
kernel_work_gga_x
#endif
(const xc_func_type_cuda *p,
	int np, const double *rho, const double *sigma,
	double *zk, double *vrho, double *vsigma,
	double *v2rho2, double *v2rhosigma, double *v2sigma2,
	double *v3rho3, double *v3rho2sigma, double *v3rhosigma2, double *v3sigma3)
{
	xc_gga_work_x_t r;

	//double sfact, sfact2, x_factor_c, alpha, beta, dens;
	double sfact, x_factor_c, alpha, beta, dens;
	
	int is, is2;

	// constants for the evaluation of the different terms 
	double c_zk[1];
	double c_vrho[3], c_vsigma[2];
	double c_v2rho2[3], c_v2rhosigma[4], c_v2sigma2[2];
	//double c_v3rho3[4], c_v3rho2sigma[3], c_v3rhosigma2[3], c_v3sigma3[3];

	// variables used inside the is loop
	double gdm, ds, rhoLDA;

	// alpha is the power of rho in the corresponding LDA
	//beta  is the power of rho in the expression for x 
	beta = 1.0 + 1.0 / XC_DIMENSIONS; // exponent of the density in expression for x

#ifndef XC_KINETIC_FUNCTIONAL
	alpha = beta;

#  if XC_DIMENSIONS == 2
	x_factor_c = -X_FACTOR_2D_C;
#  else // three dimensions
	x_factor_c = -X_FACTOR_C;
#  endif

#else

#  if XC_DIMENSIONS == 2
#  else // three dimensions
	alpha = 5.0 / 3.0;
	x_factor_c = K_FACTOR_C;
#  endif

#endif

	sfact = (p->nspin == XC_POLARIZED) ? 1.0 : 2.0;
	//sfact2 = sfact*sfact;
	
	// Initialize several constants
	r.order = -1;
	if (zk != NULL) {
		r.order = 0;
		c_zk[0] = sfact*x_factor_c;
	}
	if (vrho != NULL) {
		r.order = 1;
		c_vrho[0] = x_factor_c*alpha;
		c_vrho[1] = -x_factor_c*beta;
		c_vrho[2] = x_factor_c;
		c_vsigma[0] = sfact*x_factor_c;
		c_vsigma[1] = sfact*x_factor_c;
	}
	if (v2rho2 != NULL) {
		r.order = 2;
		c_v2rho2[0] = (x_factor_c / sfact) * (alpha - 1.0)*alpha;
		c_v2rho2[1] = (x_factor_c / sfact) * beta*(beta - 2.0*alpha + 1.0);
		c_v2rho2[2] = (x_factor_c / sfact) * beta*beta;
		c_v2rhosigma[0] = x_factor_c * (alpha - beta) / 2.0;
		c_v2rhosigma[1] = -x_factor_c * beta / 2.0;
		c_v2rhosigma[2] = x_factor_c * alpha;
		c_v2rhosigma[3] = -x_factor_c * beta;
		c_v2sigma2[0] = x_factor_c*sfact / 4.0;
		c_v2sigma2[1] = x_factor_c*sfact;
	}
	/*
	if (v3rho3 != NULL) {
		r.order = 3;
		c_v3rho3[0] = (x_factor_c / sfact2) * (alpha - 2.0)*(alpha - 1.0)*alpha;
		c_v3rho3[1] = -(x_factor_c / sfact2) * (3.0*alpha*alpha - 3.0*alpha*(2.0 + beta) + (1.0 + beta)*(2.0 + beta))*beta;
		c_v3rho3[2] = -(x_factor_c / sfact2) * 3.0*(1.0 - alpha + beta)*beta*beta;
		c_v3rho3[3] = -(x_factor_c / sfact2) * beta*beta*beta;
		c_v3rho2sigma[0] = (x_factor_c / sfact) * (alpha - beta - 1.0)*(alpha - beta) / 2.0;
		c_v3rho2sigma[1] = (x_factor_c / sfact) * (1.0 - 2.0*alpha + 3.0*beta)*beta / 2.0;
		c_v3rho2sigma[2] = (x_factor_c / sfact) * beta*beta / 2.0;
		c_v3rhosigma2[0] = -x_factor_c * (alpha - beta) / 4.0;
		c_v3rhosigma2[1] = x_factor_c * (alpha - beta) / 4.0;
		c_v3rhosigma2[2] = -x_factor_c * beta / 4.0;
		c_v3sigma3[0] = x_factor_c*sfact * 3.0 / 8.0;
		c_v3sigma3[1] = -x_factor_c*sfact * 3.0 / 8.0;
		c_v3sigma3[2] = x_factor_c*sfact / 8.0;
	}
	*/
	if (r.order < 0) return;

    	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("blockIdx.x: %i, blockDim.x: %i, threadIdx.x: %i \n", blockIdx.x, blockDim.x, threadIdx.x);
	//printf("idx: %i \n", idx);
	
	// the loop over the points starts
	if (idx < np) {
		// increment pointers
		rho += p->n_rho * idx;
		sigma += p->n_sigma * idx;

		if (zk != NULL)
			zk += p->n_zk * idx;

		if (vrho != NULL) {
			vrho += p->n_vrho * idx;
			vsigma += p->n_vsigma * idx;
		}

		if (v2rho2 != NULL) {
			v2rho2 += p->n_v2rho2 * idx;
			v2rhosigma += p->n_v2rhosigma * idx;
			v2sigma2 += p->n_v2sigma2 * idx;
		}

		if (v3rho3 != NULL) {
			v3rho3 += p->n_v3rho3 * idx;
			v3rho2sigma += p->n_v3rho2sigma * idx;
			v3rhosigma2 += p->n_v3rhosigma2 * idx;
			v3sigma3 += p->n_v3sigma3 * idx;
		}

		dens = (p->nspin == XC_UNPOLARIZED) ? rho[0] : rho[0] + rho[1];
		if (dens < p->dens_threshold) {
			return;
		}

		// Unroll this loop
		for (is = 0; is < p->nspin; is++) {
			is2 = 2 * is;

			if (rho[is] < p->dens_threshold) continue;

			gdm = max(sqrt(sigma[is2]) / sfact, p->dens_threshold);
			ds = rho[is] / sfact;
			rhoLDA = pow(ds, alpha);
			r.x = gdm / pow(ds, beta);

			// TODO: revisar esto tambien!
			//func(p, &r);

			if (r.order > 0) r.dfdx *= r.x;
			if (r.order > 1) r.d2fdx2 *= r.x*r.x;
			if (r.order > 2) r.d3fdx3 *= r.x*r.x*r.x;

			if (zk != NULL && (p->info->flags & XC_FLAGS_HAVE_EXC))
				*zk += rhoLDA*
				c_zk[0] * r.f;

			if (vrho != NULL && (p->info->flags & XC_FLAGS_HAVE_VXC)) {
				vrho[is] += (rhoLDA / ds)*
					(c_vrho[0] * r.f + c_vrho[1] * r.dfdx);

				if (gdm > p->dens_threshold)
					vsigma[is2] = rhoLDA*
					(c_vsigma[0] * r.dfdx / (2.0*sigma[is2]));
			}

			if (v2rho2 != NULL && (p->info->flags & XC_FLAGS_HAVE_FXC)) {
				v2rho2[is2] = rhoLDA / (ds*ds) * (c_v2rho2[0] * r.f + c_v2rho2[1] * r.dfdx + c_v2rho2[2] * r.d2fdx2);

				if (gdm > p->dens_threshold) {
					v2rhosigma[is * 5] = (rhoLDA / ds) *
						((c_v2rhosigma[0] * r.dfdx + c_v2rhosigma[1] * r.d2fdx2) / sigma[is2]);
					v2sigma2[is * 5] = rhoLDA*
						(c_v2sigma2[0] * (r.d2fdx2 - r.dfdx) / (sigma[is2] * sigma[is2]));
				}
			}

			/*
			if (v3rho3 != NULL && (p->info->flags & XC_FLAGS_HAVE_KXC)) {
				v3rho3[is * 3] = rhoLDA / (ds*ds*ds) *
					(c_v3rho3[0] * r.f + c_v3rho3[1] * r.dfdx + c_v3rho3[2] * r.d2fdx2 + c_v3rho3[3] * r.d3fdx3);

				if (gdm > p->dens_threshold) {
					v3rho2sigma[is * 8] = rhoLDA / (ds*ds) *
						(c_v3rho2sigma[0] * r.dfdx + c_v3rho2sigma[1] * r.d2fdx2 + c_v3rho2sigma[2] * r.d3fdx3) / sigma[is2];

					v3rhosigma2[is * 11] = (rhoLDA / ds) *
						(c_v3rhosigma2[0] * r.dfdx + c_v3rhosigma2[1] * r.d2fdx2 + c_v3rhosigma2[2] * r.d3fdx3) / (sigma[is2] * sigma[is2]);

					v3sigma3[is * 9] = rhoLDA*
						(c_v3sigma3[0] * r.dfdx + c_v3sigma3[1] * r.d2fdx2 + c_v3sigma3[2] * r.d3fdx3) / (sigma[is2] * sigma[is2] * sigma[is2]);
				}
			}
			*/
		}

		if (zk != NULL && (p->info->flags & XC_FLAGS_HAVE_EXC)) {
			*zk /= dens; // we want energy per particle
		}

	} //endif (idx < np)
}


//////////////////////////////////////////////////
// Kernel modificado POLARIZED

static void __global__
#ifdef XC_KINETIC_FUNCTIONAL
kernel_work_gga_k_polarized
#else
kernel_work_gga_x_polarized
#endif
(const xc_func_type_cuda *p,
	int np, const double *rho, const double *sigma,
	double *zk, double *vrho, double *vsigma,
	double *v2rho2, double *v2rhosigma, double *v2sigma2)
{
	xc_gga_work_x_t r;

	int flags = p->info->flags;
	int nspin = p->nspin;
	double dens_threshold = p->dens_threshold;
	//double sfact, sfact2, x_factor_c, alpha, beta, dens;
	double sfact, x_factor_c, alpha, beta, dens;
	

	// constants for the evaluation of the different terms 
	double c_zk[1];
	double c_vrho[3], c_vsigma[2];
	double c_v2rho2[3], c_v2rhosigma[4], c_v2sigma2[2];

	// variables used inside the is loop
	double gdm, ds, rhoLDA;

	// alpha is the power of rho in the corresponding LDA
	//beta  is the power of rho in the expression for x 
	beta = 1.0 + 1.0 / XC_DIMENSIONS; // exponent of the density in expression for x

#ifndef XC_KINETIC_FUNCTIONAL
	alpha = beta;

#  if XC_DIMENSIONS == 2
	x_factor_c = -X_FACTOR_2D_C;
#  else // three dimensions
	x_factor_c = -X_FACTOR_C;
#  endif

#else

#  if XC_DIMENSIONS == 2
#  else // three dimensions
	alpha = 5.0 / 3.0;
	x_factor_c = K_FACTOR_C;
#  endif

#endif

	sfact = (nspin == XC_POLARIZED) ? 1.0 : 2.0;
	//sfact2 = sfact*sfact;
	
	// Initialize several constants
	r.order = 2;
	c_zk[0] = sfact * x_factor_c;

	c_vrho[0] = x_factor_c*alpha;
	c_vrho[1] = -x_factor_c*beta;
	c_vrho[2] = x_factor_c;
	c_vsigma[0] = sfact*x_factor_c;
	c_vsigma[1] = sfact*x_factor_c;

	c_v2rho2[0] = (x_factor_c / sfact) * (alpha - 1.0)*alpha;
	c_v2rho2[1] = (x_factor_c / sfact) * beta*(beta - 2.0*alpha + 1.0);
	c_v2rho2[2] = (x_factor_c / sfact) * beta*beta;
	c_v2rhosigma[0] = x_factor_c * (alpha - beta) / 2.0;
	c_v2rhosigma[1] = -x_factor_c * beta / 2.0;
	c_v2rhosigma[2] = x_factor_c * alpha;
	c_v2rhosigma[3] = -x_factor_c * beta;
	c_v2sigma2[0] = x_factor_c*sfact / 4.0;
	c_v2sigma2[1] = x_factor_c*sfact;

    	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	// the loop over the points starts
	if (idx < np) {
		// increment pointers
		rho += p->n_rho * idx;
		sigma += p->n_sigma * idx;
		zk += p->n_zk * idx;
		vrho += p->n_vrho * idx;
		vsigma += p->n_vsigma * idx;
		v2rho2 += p->n_v2rho2 * idx;
		v2rhosigma += p->n_v2rhosigma * idx;
		v2sigma2 += p->n_v2sigma2 * idx;

		dens = (nspin == XC_UNPOLARIZED) ? rho[0] : rho[0] + rho[1];
		if (dens < dens_threshold) {
			return;
		}

		int is, is2;
		for (is = 0; is < nspin; is++) {
			is2 = 2 * is;

			if (rho[is] < dens_threshold) continue;

			gdm = max ( sqrt(sigma[is2]) / sfact, dens_threshold);
			ds = rho[is] / sfact;
			rhoLDA = pow(ds, alpha);
			r.x = gdm / pow(ds, beta);

			//func(p, &r);

			r.dfdx *= r.x;
			r.d2fdx2 *= r.x*r.x;

			if (flags & XC_FLAGS_HAVE_EXC) {
			    *zk += rhoLDA * c_zk[0] * r.f;
			}

			if (flags & XC_FLAGS_HAVE_VXC) {
			    vrho[is] += (rhoLDA / ds) * (c_vrho[0] * r.f + c_vrho[1] * r.dfdx);
			    if (gdm > dens_threshold) {
				vsigma[is2] = rhoLDA * (c_vsigma[0] * r.dfdx / (2.0*sigma[is2]));
			    }
			}

			if (flags & XC_FLAGS_HAVE_FXC) {
			    v2rho2[is2] = rhoLDA / (ds*ds) * (c_v2rho2[0] * r.f + c_v2rho2[1] * r.dfdx + c_v2rho2[2] * r.d2fdx2);

			    if (gdm > dens_threshold) {
				v2rhosigma[is * 5] = (rhoLDA / ds) * ((c_v2rhosigma[0] * r.dfdx + c_v2rhosigma[1] * r.d2fdx2) / sigma[is2]);
				v2sigma2[is * 5] = rhoLDA * (c_v2sigma2[0] * (r.d2fdx2 - r.dfdx) / (sigma[is2] * sigma[is2]));
			    }
			}
		}

		if (flags & XC_FLAGS_HAVE_EXC) {
		    *zk /= dens; // we want energy per particle
		}

	} //endif (idx < np)
}

///////////////////////////////////////////////////
// Kernel modificado UNPOLARIZED 3rd DERIVATIVE
static void __global__
#ifdef XC_KINETIC_FUNCTIONAL
kernel_work_gga_k_unpolarized_3rd
#else
kernel_work_gga_x_unpolarized_3rd
#endif
(const xc_func_type_cuda *p,
        int np, const double *rho, const double *sigma,
        double *zk, double *vrho, double *vsigma,
        double *v2rho2, double *v2rhosigma, double *v2sigma2,
        double *v3rho3, double *v3rho2sigma, double* v3rhosigma2, double *v3sigma3)
{
        xc_gga_work_x_t r;
        double dens_threshold = p->dens_threshold;

        // constants for the evaluation of the different terms 
        __shared__ double sfact, sfact2, x_factor_c, alpha, beta;
        __shared__ double c_zk[1];
        __shared__ double c_vrho[3], c_vsigma[2];
        __shared__ double c_v2rho2[3], c_v2rhosigma[4], c_v2sigma2[2];
        __shared__ double c_v3rho3[4], c_v3rho2sigma[3], c_v3rhosigma2[3], c_v3sigma3[3];


        // alpha is the power of rho in the corresponding LDA
        //beta  is the power of rho in the expression for x 
        beta = 1.0 + 1.0 / XC_DIMENSIONS; // exponent of the density in expression for x

#ifndef XC_KINETIC_FUNCTIONAL
        alpha = beta;

#  if XC_DIMENSIONS == 2
        x_factor_c = -X_FACTOR_2D_C;
#  else // three dimensions
        x_factor_c = -X_FACTOR_C;
#  endif

#else

#  if XC_DIMENSIONS == 2
#  else // three dimensions
        alpha = 5.0 / 3.0;
        x_factor_c = K_FACTOR_C;
#  endif

#endif

        sfact = 2.0;
        sfact2 = 4.0;
        // Initialize several constants
        r.order = 3;
        c_zk[0] = sfact * x_factor_c;

        c_vrho[0] = x_factor_c*alpha;
        c_vrho[1] = -x_factor_c*beta;
        c_vrho[2] = x_factor_c;
        c_vsigma[0] = sfact*x_factor_c;
        c_vsigma[1] = sfact*x_factor_c;

        c_v2rho2[0] = (x_factor_c / sfact) * (alpha - 1.0)*alpha;
        c_v2rho2[1] = (x_factor_c / sfact) * beta*(beta - 2.0*alpha + 1.0);
        c_v2rho2[2] = (x_factor_c / sfact) * beta*beta;
        c_v2rhosigma[0] = x_factor_c * (alpha - beta) / 2.0;
        c_v2rhosigma[1] = -x_factor_c * beta / 2.0;
        c_v2rhosigma[2] = x_factor_c * alpha;
        c_v2rhosigma[3] = -x_factor_c * beta;
        c_v2sigma2[0] = x_factor_c*sfact / 4.0;
        c_v2sigma2[1] = x_factor_c*sfact;

        c_v3rho3[0] =  (x_factor_c/sfact2) * (alpha - 2.0)*(alpha - 1.0)*alpha;
        c_v3rho3[1] = -(x_factor_c/sfact2) * (3.0*alpha*alpha - 3.0*alpha*(2.0 + beta) + (1.0 + beta)*(2.0 + beta))*beta;
        c_v3rho3[2] = -(x_factor_c/sfact2) * 3.0*(1.0 - alpha + beta)*beta*beta;
        c_v3rho3[3] = -(x_factor_c/sfact2) * beta*beta*beta;
        c_v3rho2sigma[0] = (x_factor_c/sfact) * (alpha - beta - 1.0)*(alpha - beta)/2.0;
        c_v3rho2sigma[1] = (x_factor_c/sfact) * (1.0 - 2.0*alpha + 3.0*beta)*beta/2.0;
        c_v3rho2sigma[2] = (x_factor_c/sfact) * beta*beta/2.0;
        c_v3rhosigma2[0] = -x_factor_c * (alpha - beta)/4.0;
        c_v3rhosigma2[1] =  x_factor_c * (alpha - beta)/4.0;
        c_v3rhosigma2[2] = -x_factor_c * beta/4.0;
        c_v3sigma3[0] =  x_factor_c*sfact * 3.0/8.0;
        c_v3sigma3[1] = -x_factor_c*sfact * 3.0/8.0;
        c_v3sigma3[2] =  x_factor_c*sfact /8.0;

        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        // the loop over the points starts
        if (idx < np) {
                // increment pointers
                rho += p->n_rho * idx;
                sigma += p->n_sigma * idx;
                zk += p->n_zk * idx;
                vrho += p->n_vrho * idx;
                vsigma += p->n_vsigma * idx;
                v2rho2 += p->n_v2rho2 * idx;
                v2rhosigma += p->n_v2rhosigma * idx;
                v2sigma2 += p->n_v2sigma2 * idx;
                v3rho3 += p->n_v3rho3 * idx;
                v3rho2sigma += p->n_v3rho2sigma * idx;
                v3rhosigma2 += p->n_v3rhosigma2 * idx;
                v3sigma3 += p->n_v3sigma3 * idx;

                if (rho[0] < dens_threshold) {
                    return;
                }

                // variables used inside the is loop
                double gdm, ds, rhoLDA;
                gdm = max ( sqrt(sigma[0]) / sfact, dens_threshold);
                ds = rho[0] / sfact;
                rhoLDA = pow(ds, alpha);
                r.x = gdm / pow(ds, beta);

                func(p, &r);

                r.dfdx *= r.x;
                r.d2fdx2 *= r.x*r.x;
                r.d3fdx3 *= r.x*r.x*r.x;

                *zk += rhoLDA * c_zk[0] * r.f;

                vrho[0] += (rhoLDA / ds) * (c_vrho[0] * r.f + c_vrho[1] * r.dfdx);
                v2rho2[0] = rhoLDA / (ds*ds) * (c_v2rho2[0] * r.f + c_v2rho2[1] * r.dfdx + c_v2rho2[2] * r.d2fdx2);
                v3rho3[0] = rhoLDA/(ds*ds*ds) * (c_v3rho3[0]*r.f + c_v3rho3[1]*r.dfdx + c_v3rho3[2]*r.d2fdx2 + c_v3rho3[3]*r.d3fdx3);

                if (gdm > dens_threshold) {
                    vsigma[0] = rhoLDA * (c_vsigma[0] * r.dfdx / (2.0*sigma[0]));
                    v2rhosigma[0] = (rhoLDA / ds) * ((c_v2rhosigma[0] * r.dfdx + c_v2rhosigma[1] * r.d2fdx2) / sigma[0]);
                    v2sigma2[0] = rhoLDA * (c_v2sigma2[0] * (r.d2fdx2 - r.dfdx) / (sigma[0] * sigma[0]));
                    v3rho2sigma[0] =rhoLDA/(ds*ds)*(c_v3rho2sigma[0]*r.dfdx + c_v3rho2sigma[1]*r.d2fdx2 + c_v3rho2sigma[2]*r.d3fdx3)/sigma[0];
                    v3rhosigma2[0] =(rhoLDA/ds)*(c_v3rhosigma2[0]*r.dfdx+c_v3rhosigma2[1]*r.d2fdx2+c_v3rhosigma2[2]*r.d3fdx3)/(sigma[0]*sigma[0]);
                    v3sigma3[0] =rhoLDA*(c_v3sigma3[0]*r.dfdx + c_v3sigma3[1]*r.d2fdx2 + c_v3sigma3[2]*r.d3fdx3)/(sigma[0]*sigma[0]*sigma[0]);
                }

                *zk /= rho[0]; // we want energy per particle

        } //endif (idx < np)

}


//////////////////////////////////////////////////
// Kernel modificado UNPOLARIZED
static void __global__
#ifdef XC_KINETIC_FUNCTIONAL
kernel_work_gga_k_unpolarized
#else
kernel_work_gga_x_unpolarized
#endif
(const xc_func_type_cuda *p,
	int np, const double *rho, const double *sigma,
	double *zk, double *vrho, double *vsigma,
	double *v2rho2, double *v2rhosigma, double *v2sigma2)
{
	xc_gga_work_x_t r;
	double dens_threshold = p->dens_threshold;

	// constants for the evaluation of the different terms 
	__shared__ double sfact, x_factor_c, alpha, beta;
	__shared__ double c_zk[1];
	__shared__ double c_vrho[3], c_vsigma[2];
	__shared__ double c_v2rho2[3], c_v2rhosigma[4], c_v2sigma2[2];

	// alpha is the power of rho in the corresponding LDA
	//beta  is the power of rho in the expression for x 
	beta = 1.0 + 1.0 / XC_DIMENSIONS; // exponent of the density in expression for x

#ifndef XC_KINETIC_FUNCTIONAL
	alpha = beta;

#  if XC_DIMENSIONS == 2
	x_factor_c = -X_FACTOR_2D_C;
#  else // three dimensions
	x_factor_c = -X_FACTOR_C;
#  endif

#else

#  if XC_DIMENSIONS == 2
#  else // three dimensions
	alpha = 5.0 / 3.0;
	x_factor_c = K_FACTOR_C;
#  endif

#endif

	sfact = 2.0;
	// Initialize several constants
	r.order = 2;
	c_zk[0] = sfact * x_factor_c;

	c_vrho[0] = x_factor_c*alpha;
	c_vrho[1] = -x_factor_c*beta;
	c_vrho[2] = x_factor_c;
	c_vsigma[0] = sfact*x_factor_c;
	c_vsigma[1] = sfact*x_factor_c;

	c_v2rho2[0] = (x_factor_c / sfact) * (alpha - 1.0)*alpha;
	c_v2rho2[1] = (x_factor_c / sfact) * beta*(beta - 2.0*alpha + 1.0);
	c_v2rho2[2] = (x_factor_c / sfact) * beta*beta;
	c_v2rhosigma[0] = x_factor_c * (alpha - beta) / 2.0;
	c_v2rhosigma[1] = -x_factor_c * beta / 2.0;
	c_v2rhosigma[2] = x_factor_c * alpha;
	c_v2rhosigma[3] = -x_factor_c * beta;
	c_v2sigma2[0] = x_factor_c*sfact / 4.0;
	c_v2sigma2[1] = x_factor_c*sfact;

    	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// the loop over the points starts
	if (idx < np) {
		// increment pointers
		rho += p->n_rho * idx;
		sigma += p->n_sigma * idx;
		zk += p->n_zk * idx;
		vrho += p->n_vrho * idx;
		vsigma += p->n_vsigma * idx;
		v2rho2 += p->n_v2rho2 * idx;
		v2rhosigma += p->n_v2rhosigma * idx;
		v2sigma2 += p->n_v2sigma2 * idx;

		if (rho[0] < dens_threshold) {
		    return;
		}

		// variables used inside the is loop
		double gdm, ds, rhoLDA;
		gdm = max ( sqrt(sigma[0]) / sfact, dens_threshold);
		ds = rho[0] / sfact;
		rhoLDA = pow(ds, alpha);
		r.x = gdm / pow(ds, beta);

		func(p, &r);

		r.dfdx *= r.x;
		r.d2fdx2 *= r.x*r.x;

		*zk += rhoLDA * c_zk[0] * r.f;

		vrho[0] += (rhoLDA / ds) * (c_vrho[0] * r.f + c_vrho[1] * r.dfdx);
		v2rho2[0] = rhoLDA / (ds*ds) * (c_v2rho2[0] * r.f + c_v2rho2[1] * r.dfdx + c_v2rho2[2] * r.d2fdx2);

		if (gdm > dens_threshold) {
		    vsigma[0] = rhoLDA * (c_vsigma[0] * r.dfdx / (2.0*sigma[0]));
		    v2rhosigma[0] = (rhoLDA / ds) * ((c_v2rhosigma[0] * r.dfdx + c_v2rhosigma[1] * r.d2fdx2) / sigma[0]);
		    v2sigma2[0] = rhoLDA * (c_v2sigma2[0] * (r.d2fdx2 - r.dfdx) / (sigma[0] * sigma[0]));
		}

		*zk /= rho[0]; // we want energy per particle

	} //endif (idx < np)
}

/////////////////////////////////////////////////////////
// Kernel modificado UNPOLARIZED OPTIMIZED FOR GGA_X_PBE
static void __global__
#ifdef XC_KINETIC_FUNCTIONAL
kernel_work_gga_k_unpolarized_optimized_for_gga
#else
kernel_work_gga_x_unpolarized_optimized_for_gga
#endif
(const xc_func_type_cuda *p,
	int np, const double *rho, const double *sigma,
	double *zk, double *vrho, double *vsigma,
	double *v2rho2, double *v2rhosigma, double *v2sigma2)
{
	//xc_gga_work_x_t r;
	double dens_threshold = p->dens_threshold;

	// constants for the evaluation of the different terms 
	__shared__ double sfact, x_factor_c, alpha, beta;
	__shared__ double c_zk[1];
	__shared__ double c_vrho[3], c_vsigma[2];
	__shared__ double c_v2rho2[3], c_v2rhosigma[4], c_v2sigma2[2];

	// alpha is the power of rho in the corresponding LDA
	//beta  is the power of rho in the expression for x 
	beta = 1.0 + 1.0 / XC_DIMENSIONS; // exponent of the density in expression for x

#ifndef XC_KINETIC_FUNCTIONAL
	alpha = beta;

#  if XC_DIMENSIONS == 2
	x_factor_c = -X_FACTOR_2D_C;
#  else // three dimensions
	x_factor_c = -X_FACTOR_C;
#  endif

#else

#  if XC_DIMENSIONS == 2
#  else // three dimensions
	alpha = 5.0 / 3.0;
	x_factor_c = K_FACTOR_C;
#  endif

#endif

	sfact = 2.0;
	// Initialize several constants
	//r.order = 2;
	c_zk[0] = sfact * x_factor_c;

	c_vrho[0] = x_factor_c*alpha;
	c_vrho[1] = -x_factor_c*beta;
	c_vrho[2] = x_factor_c;
	c_vsigma[0] = sfact*x_factor_c;
	c_vsigma[1] = sfact*x_factor_c;

	c_v2rho2[0] = (x_factor_c / sfact) * (alpha - 1.0)*alpha;
	c_v2rho2[1] = (x_factor_c / sfact) * beta*(beta - 2.0*alpha + 1.0);
	c_v2rho2[2] = (x_factor_c / sfact) * beta*beta;
	c_v2rhosigma[0] = x_factor_c * (alpha - beta) / 2.0;
	c_v2rhosigma[1] = -x_factor_c * beta / 2.0;
	c_v2rhosigma[2] = x_factor_c * alpha;
	c_v2rhosigma[3] = -x_factor_c * beta;
	c_v2sigma2[0] = x_factor_c*sfact / 4.0;
	c_v2sigma2[1] = x_factor_c*sfact;

    	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// the loop over the points starts
	if (idx < np) {
		xc_gga_work_x_t r;
		r.order = 2;

		// increment pointers
		rho += p->n_rho * idx;
		sigma += p->n_sigma * idx;
		zk += p->n_zk * idx;
		vrho += p->n_vrho * idx;
		vsigma += p->n_vsigma * idx;
		v2rho2 += p->n_v2rho2 * idx;
		v2rhosigma += p->n_v2rhosigma * idx;
		v2sigma2 += p->n_v2sigma2 * idx;

		if (rho[0] < dens_threshold) {
		    return;
		}

		// variables used inside the is loop
		double gdm, ds, rhoLDA;
		gdm = max ( sqrt(sigma[0]) / sfact, dens_threshold);
		ds = rho[0] / sfact;
		rhoLDA = pow(ds, alpha);
		r.x = gdm / pow(ds, beta);

		func(p, &r);

		r.dfdx *= r.x;
		r.d2fdx2 *= r.x*r.x;

		*zk += rhoLDA * c_zk[0] * r.f;

		vrho[0] += (rhoLDA / ds) * (c_vrho[0] * r.f + c_vrho[1] * r.dfdx);
		v2rho2[0] = rhoLDA / (ds*ds) * (c_v2rho2[0] * r.f + c_v2rho2[1] * r.dfdx + c_v2rho2[2] * r.d2fdx2);

		if (gdm > dens_threshold) {
		    vsigma[0] = rhoLDA * (c_vsigma[0] * r.dfdx / (2.0*sigma[0]));
		    v2rhosigma[0] = (rhoLDA / ds) * ((c_v2rhosigma[0] * r.dfdx + c_v2rhosigma[1] * r.d2fdx2) / sigma[0]);
		    v2sigma2[0] = rhoLDA * (c_v2sigma2[0] * (r.d2fdx2 - r.dfdx) / (sigma[0] * sigma[0]));
		}

		*zk /= rho[0]; // we want energy per particle

	} //endif (idx < np)
}


extern "C" static void
#ifdef XC_KINETIC_FUNCTIONAL
work_gga_k
#else
work_gga_x_cuda
#endif
(const xc_func_type_cuda *p, int np, const double *rho, const double *sigma,
	double *zk, double *vrho, double *vsigma,
	double *v2rho2, double *v2rhosigma, double *v2sigma2,
	double *v3rho3, double *v3rho2sigma, double *v3rhosigma2, double *v3sigma3)
{
	//printf("libxcGPU work_gga_x() pero en cuda-> Esto se ejecuta en CPU \n");
	//print_work_gga_x_input (np, rho, sigma, zk, vrho, vsigma, v2rho2, v2rhosigma, v2sigma2,
	//    v3rho3, v3rho2sigma, v3rhosigma2, v3sigma3);

	// Setting cache memory preferences
	//checkCudaErrors (cudaFuncSetCacheConfig (kernel_work_gga_x_unpolarized_optimized_for_gga, cudaFuncCachePreferShared));
	//checkCudaErrors (cudaFuncSetCacheConfig (kernel_work_gga_x_unpolarized_optimized_for_gga, cudaFuncCachePreferL1));
	//checkCudaErrors (cudaFuncSetCacheConfig (kernel_work_gga_x_unpolarized_optimized_for_gga, cudaFuncCachePreferEqual));
	//checkCudaErrors(cudaDeviceSetSharedMemConfig (cudaSharedMemBankSizeEightByte));

	// Variables for the device
	//xc_func_info_type_cuda* infoTypeCUDA;
	xc_func_type_cuda* pCUDA;

	// Minimial information for gga_x_pbe
	//xc_func_type_cuda_gga* pGGACUDA;
	//xc_func_type_cuda_gga pGGA;

	// Set the minimun type of information for
	// the functional in cuda.
	//pGGA.nspin = p->nspin;
	//pGGA.n_rho = p->n_rho;
	//pGGA.n_sigma = p->n_sigma;
	//pGGA.n_zk = p->n_zk;
	//pGGA.n_vrho = p->n_vrho;
	//pGGA.n_vsigma = p->n_vsigma;
	//pGGA.n_v2rho2 = p->n_v2rho2;
	//pGGA.n_v2sigma2 = p->n_v2sigma2;
	//pGGA.n_v2rhosigma = p->n_v2rhosigma;
	//pGGA.params = p->params;
	//pGGA.dens_threshold = p->dens_threshold;
	//pGGA.params_gpu = p->params_gpu;
	//pGGA.size_of_params_gpu = p->size_of_params_gpu;


	int xc_func_type_cuda_size = sizeof(struct xc_func_type_cuda);
	//int info_type_size = sizeof(xc_func_info_type_cuda);
	//int xc_func_type_cuda_gga_size = sizeof(xc_func_type_gga);

	// Copy info_type to CUDA
	//checkCudaErrors(cudaMalloc((void **)&infoTypeCUDA, info_type_size));
	//checkCudaErrors(cudaMemcpy(infoTypeCUDA, (xc_func_info_type_cuda*)(p->info), info_type_size, cudaMemcpyHostToDevice));

	// Copy p->params_gpu to CUDA
	checkCudaErrors(cudaMalloc((void **)&pCUDA, xc_func_type_cuda_size));
	checkCudaErrors(cudaMemcpy(pCUDA, p, xc_func_type_cuda_size, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(&(pCUDA->params_gpu), &(p->params_gpu), sizeof(p->params_gpu), cudaMemcpyHostToDevice));

	// Copy data to pGGACUDA
	//checkCudaErrors(cudaMalloc((void **)&pGGACUDA, xc_func_type_cuda_gga_size));
	//checkCudaErrors(cudaMemcpy(pGGACUDA, &pGGA, xc_func_type_cuda_gga_size, cudaMemcpyHostToDevice));
	//checkCudaErrors(cudaMemcpy(&(pGGACUDA->params_gpu), &(pGGA.params_gpu), sizeof(pGGA.params_gpu), cudaMemcpyHostToDevice));

	// Deep copy of p->info.
	//checkCudaErrors(cudaMemcpy(&(pCUDA->info), &infoTypeCUDA, sizeof(infoTypeCUDA), cudaMemcpyHostToDevice));

	// Launch the CUDA Kernel
	//int threadsPerBlock = 256;
	//int threadsPerBlock = MAX_THREAD_COUNT;
	//int blocksPerGrid =(np + threadsPerBlock - 1) / threadsPerBlock;

	int blockSize;
	int minGridSize;
	int gridSize;
	size_t dynamicSMemUsage = 0;

	if (p->nspin == XC_POLARIZED) {
            checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(
                            &minGridSize,
                            &blockSize,
                            (void*)kernel_work_gga_x_polarized,
                            dynamicSMemUsage,
                            np));

	} else {
            checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(
                            &minGridSize,
                            &blockSize,
                            (void*)kernel_work_gga_x_unpolarized,
                            dynamicSMemUsage,
                            np));
	}

	// Round up
        //
        gridSize = (np + blockSize - 1) / blockSize;

	//printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
	if (p->nspin == XC_POLARIZED) {
	    kernel_work_gga_x_polarized <<<gridSize, blockSize, dynamicSMemUsage>>> (pCUDA,
		np, rho, sigma,
		zk, vrho, vsigma,
		v2rho2, v2rhosigma, v2sigma2);
	} else {
            //printf("libxc cuda kernel_work_gga_x_unpo\n");
            if (v3rho3 == NULL) {
	        kernel_work_gga_x_unpolarized <<<gridSize, blockSize, dynamicSMemUsage>>> (pCUDA,
		np, rho, sigma,
		zk, vrho, vsigma,
		v2rho2, v2rhosigma, v2sigma2);
            } else {
	        kernel_work_gga_x_unpolarized_3rd <<<gridSize, blockSize, dynamicSMemUsage>>> (pCUDA,
		np, rho, sigma,
		zk, vrho, vsigma,
		v2rho2, v2rhosigma, v2sigma2,
		v3rho3, v3rho2sigma, v3rhosigma2, v3sigma3);
            }
/*
	    kernel_work_gga_x_unpolarized_optimized_for_gga <<<gridSize, blockSize, dynamicSMemUsage>>> (pGGACUDA,
		np, rho, sigma,
		zk, vrho, vsigma,
		v2rho2, v2rhosigma, v2sigma2);
*/
	}
	// Check for errors.
	cudaError_t lastError = cudaGetLastError();

	if (lastError != cudaSuccess) {
	    fprintf (stderr, "CUDA error %d \n", lastError);
	}

	checkCudaErrors(cudaFree(pCUDA));
	//checkCudaErrors(cudaFree(pGGACUDA));
	//checkCudaErrors(cudaFree(infoTypeCUDA));
}
