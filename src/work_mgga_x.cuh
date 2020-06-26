/*
 Copyright (C) 2006-2008 M.A.L. Marques

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

/************************************************************************
  This file is to be included in meta GGA exchange functionals. As often these
  functionals are written as a function of s = |grad n|/n^(4/3) and tau, this
  routine performs the necessary conversions between a functional of s and tau
  and of rho.
************************************************************************/


/* WARNING Kinetic energy functionals are still not working */
// CUDA runtime
#include <cuda_runtime.h>
// helper functions and utilities to work with CUDA
#include "helper/helper_cuda.h"

#include "util.h"
#include "work_utils.h"


#ifndef XC_DIMENSIONS
#  define XC_DIMENSIONS 3
#endif


static void __global__ kernel_work_mgga_x (const xc_func_type_cuda *p, int np,
 const double *rho, const double *sigma, const double *lapl, const double *tau,
 double *zk, double *vrho, double *vsigma, double *vlapl, double *vtau,
 double *v2rho2, double *v2sigma2, double *v2lapl2, double *v2tau2,
 double *v2rhosigma, double *v2rholapl, double *v2rhotau, 
 double *v2sigmalapl, double *v2sigmatau, double *v2lapltau)
{
  xc_mgga_work_x_t r;
  double sfact, sfact2, x_factor_c, beta, dens;
  double min_grad = p->dens_threshold;
  int is, ip, idx;

  /* WARNING: derivatives are _not_ OK for 2 dimensions */
#if XC_DIMENSIONS == 2
  const double cnst_rs = 1.0/M_SQRTPI;
  x_factor_c = X_FACTOR_2D_C;
#else /* three dimensions */
  const double cnst_rs = RS_FACTOR;
  x_factor_c = X_FACTOR_C;
#endif

  r.order = -1;
  if(zk     != NULL) r.order = 0;
  if(vrho   != NULL) r.order = 1;
  if(v2rho2 != NULL) r.order = 2;
  if(r.order < 0) return;

  sfact = (p->nspin == XC_POLARIZED) ? 1.0 : 2.0;
  sfact2 = sfact*sfact;

  idx = blockIdx.x*blockDim.x + threadIdx.x;

  if (idx < np) {

    /* increment pointers */
    rho   += p->n_rho * idx;
    sigma += p->n_sigma * idx;
    tau   += p->n_tau * idx;
    lapl  += p->n_lapl * idx;

    if(zk != NULL) {
      zk += p->n_zk * idx;
    }

    if(vrho != NULL) {
      vrho   += p->n_vrho * idx;
      vsigma += p->n_vsigma * idx;
      vtau   += p->n_vtau * idx;
      vlapl  += p->n_vlapl * idx;
    }

    if(v2rho2 != NULL){
      v2rho2      += p->n_v2rho2 * idx;
      v2sigma2    += p->n_v2sigma2 * idx;
      v2tau2      += p->n_v2tau2 * idx;
      v2lapl2     += p->n_v2lapl2 * idx;
      v2rhosigma  += p->n_v2rhosigma * idx;
      v2rhotau    += p->n_v2rhotau * idx;
      v2rholapl   += p->n_v2rholapl * idx;
      v2sigmatau  += p->n_v2sigmatau * idx;
      v2sigmalapl += p->n_v2sigmalapl * idx;
      v2lapltau   += p->n_v2lapltau * idx;
    }

    //xc_rho2dzeta(p->nspin, rho, &dens, &r.zeta);
    if(p->nspin == XC_UNPOLARIZED) {
	dens = max(rho[0], 0.0);
	r.zeta = 0.0;
    } else {
	dens = rho[0] + rho[1];
	if(dens > 0.0){
	    r.zeta = (rho[0] - rho[1])/(dens);
	    r.zeta = min(r.zeta,  1.0);
	    r.zeta = max(r.zeta, -1.0);
	} else {
	    dens = 0.0;
	    r.zeta = 0.0;
	}
    }

    if(dens < p->dens_threshold){
	return;
    }

    r.rs = cnst_rs*pow(dens, -1.0/XC_DIMENSIONS);

    for(is=0; is<p->nspin; is++){
      double lrho, rho1D, rho2pD_D, lsigma, gdm, lnr2, ltau;
      int js = (is == 0) ? 0 : 2;
      int ls = (is == 0) ? 0 : 3;
      int ks = (is == 0) ? 0 : 5;

      if (rho[is] < p->dens_threshold) continue;

      lsigma= max(sigma[js]/sfact2, min_grad*min_grad);
      gdm   = sqrt(lsigma);
      lrho  = rho[is]/sfact;
      rho1D = pow(lrho, 1.0/XC_DIMENSIONS);
      rho2pD_D = lrho*rho1D*rho1D;
      r.x   = gdm/(lrho*rho1D);

      ltau  = tau[is]/sfact;
      r.t   = ltau/rho2pD_D;  /* tau/rho^((2+D)/D) */

      if(p->info->flags & XC_FLAGS_NEEDS_LAPLACIAN){
        lnr2  = lapl[is]/sfact; /* this can be negative */
        r.u   = lnr2/rho2pD_D;  /* lapl/rho^((2+D)/D) */
      }

      func(p, &r);

      if(zk != NULL && (p->info->flags & XC_FLAGS_HAVE_EXC)){
	*zk += -sfact*x_factor_c*(lrho*rho1D)*r.f;
      }

      if(vrho != NULL && (p->info->flags & XC_FLAGS_HAVE_VXC)){
	vrho[is]  = -x_factor_c*rho1D*(-r.rs*r.dfdrs + 4.0/3.0*(r.f - r.dfdx*r.x) - 5.0/3.0*(r.dfdt*r.t + r.dfdu*r.u));

	vtau[is]  = -x_factor_c*r.dfdt/rho1D;

        if(p->info->flags & XC_FLAGS_NEEDS_LAPLACIAN)
          vlapl[is] = -x_factor_c*r.dfdu/rho1D;

	if(gdm>min_grad)
	  vsigma[js] = -x_factor_c*(rho1D*lrho)*r.dfdx*r.x/(2.0*sfact*lsigma);
      }

      /* WARNING: terms with rs not implemented yet */
      if(v2rho2 != NULL && (p->info->flags & XC_FLAGS_HAVE_FXC)){
	v2rho2[js]    = -x_factor_c/(9.0*sfact*rho1D*rho1D)*
	  (4.0*r.f - 4.0*r.x*r.dfdx + 4.0*4.0*r.x*r.x*r.d2fdx2 + 5.0*5.0*r.t*r.t*r.d2fdt2 + 5.0*5.0*r.u*r.u*r.d2fdu2 +
	   2.0*5.0*(4.0*r.x*r.t*r.d2fdxt + 4.0*r.x*r.u*r.d2fdxu + 5.0*r.t*r.u*r.d2fdtu));

	v2tau2[js]    = -x_factor_c*r.d2fdt2/(sfact*rho1D*rho2pD_D);

        if(p->info->flags & XC_FLAGS_NEEDS_LAPLACIAN){
          v2lapl2[js]   = -x_factor_c*r.d2fdu2/(sfact*rho1D*rho2pD_D);

          v2rholapl[ls] = -x_factor_c*rho1D/(3.0*sfact*rho2pD_D)*
            (4.0*r.dfdu - 4.0*r.x*r.d2fdxu - 5.0*r.t*r.d2fdtu - 5.0*(r.dfdu + r.u*r.d2fdu2));

          v2lapltau[ls] = -x_factor_c*r.d2fdtu/(rho1D*rho2pD_D);
        }

	v2rhotau[ls]  = -x_factor_c*rho1D/(3.0*sfact*rho2pD_D)*
	  (4.0*r.dfdt - 4.0*r.x*r.d2fdxt - 5.0*r.u*r.d2fdtu - 5.0*(r.dfdt + r.t*r.d2fdt2));

	if(gdm > min_grad){
	  v2sigma2[ks]    =  -x_factor_c*(rho1D*lrho)/(4.0*sfact2*sfact*lsigma*lsigma)*
	    (r.d2fdx2*r.x*r.x - r.dfdx*r.x);

	  v2rhosigma[ks]  = -x_factor_c*rho1D*r.x/(3.0*2.0*sfact2*lsigma)*
	    (-4.0*r.x*r.d2fdx2 - 5.0*r.t*r.d2fdxt - 5.0*r.u*r.d2fdxu);

          if(p->info->flags & XC_FLAGS_NEEDS_LAPLACIAN)
            v2sigmalapl[ks] = -x_factor_c*r.x/(2.0*sfact2*lsigma*rho1D)*r.d2fdxu;

          v2sigmatau[ks]  = -x_factor_c*r.x/(2.0*sfact2*lsigma*rho1D)*r.d2fdxt;

	}
      }
    }

    if(zk != NULL) {
      *zk /= dens; /* we want energy per particle */
    }

  }
}

extern "C" static void
#ifdef XC_KINETIC_FUNCTIONAL
work_mgga_k
#else
work_mgga_x
#endif
(const xc_func_type_cuda *p, int np,
 const double *rho, const double *sigma, const double *lapl, const double *tau,
 double *zk, double *vrho, double *vsigma, double *vlapl, double *vtau,
 double *v2rho2, double *v2sigma2, double *v2lapl2, double *v2tau2,
 double *v2rhosigma, double *v2rholapl, double *v2rhotau, 
 double *v2sigmalapl, double *v2sigmatau, double *v2lapltau)
{
    printf("work_mgga_x()\n");

    // CUDA variables for the kernel.
    xc_func_info_type_cuda* infoTypeCUDA;
    xc_func_type_cuda* pCUDA;

    // Struct sizes
    int xc_func_type_cuda_size = sizeof(xc_func_type_cuda);
    int info_type_size = sizeof(xc_func_info_type_cuda);

    // Alloc CUDA memory for infoTypeCUDA
    checkCudaErrors(cudaMalloc((void **)&infoTypeCUDA, info_type_size));
    // Copy infoTypeCUDA to the device
    checkCudaErrors(cudaMemcpy(infoTypeCUDA, (xc_func_info_type_cuda*)(p->info), info_type_size, cudaMemcpyHostToDevice));

    // Alloc CUDA memory.
    checkCudaErrors(cudaMalloc((void **)&pCUDA, xc_func_type_cuda_size));
    // Copy the first parameter to the device.
    checkCudaErrors(cudaMemcpy(pCUDA, p, xc_func_type_cuda_size, cudaMemcpyHostToDevice));
    // Now the make a "deep copy" of p->params.
    checkCudaErrors(cudaMemcpy(&(pCUDA->params_gpu), &(p->params_gpu), sizeof(p->params_gpu), cudaMemcpyHostToDevice));

    // Deep copy of infoType
    checkCudaErrors(cudaMemcpy(&(pCUDA->info), &infoTypeCUDA, sizeof(infoTypeCUDA), cudaMemcpyHostToDevice));

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = MAX_THREAD_COUNT;
    int blocksPerGrid = (np + threadsPerBlock - 1) / threadsPerBlock;
    //printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    kernel_work_mgga_x <<<blocksPerGrid, threadsPerBlock>>> (
	pCUDA, np,
	rho, sigma, lapl, tau,
	zk, vrho, vsigma, vlapl, vtau,
	v2rho2, v2sigma2, v2lapl2, v2tau2,
	v2rhosigma, v2rholapl, v2rhotau, 
	v2sigmalapl, v2sigmatau, v2lapltau);

    cudaError_t lastError = cudaGetLastError();

    if (lastError != cudaSuccess) {
	fprintf (stderr, "CUDA error %d \n", lastError);
    }

    // Free CUDA memory.
    checkCudaErrors(cudaFree(pCUDA));
    checkCudaErrors(cudaFree(infoTypeCUDA));

}