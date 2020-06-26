/*
 Copyright (C) 2006-2007 M.A.L. Marques

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

/**
 * @file work_lda.cuh
 * @brief This file is to be included in LDA functionals. As often these
 *        functionals are written as a function of rs and zeta, this
 *        routine performs the necessary conversions between this and a functional
 *        of rho.
 */

// CUDA runtime
#include <cuda_runtime.h>
// helper functions and utilities to work with CUDA
#include "helper/helper_cuda.h"

#include "util.h"
#include "work_utils.h"

#ifndef XC_DIMENSIONS
#define XC_DIMENSIONS 3
#endif

static void __global__ kernel_work_lda (const xc_func_type_cuda *p, int np, const double *rho, 
	 double *zk, double *vrho, double *v2rho2, double *v3rho3)
{
  xc_lda_work_t r;
  int is, idx;
  double dens, drs, d2rs, d3rs;

  /* Wigner radius */
# if   XC_DIMENSIONS == 1
  const double cnst_rs = 0.5;
# elif XC_DIMENSIONS == 2
  const double cnst_rs = 1.0/M_SQRTPI;
# else /* three dimensions */
  const double cnst_rs = RS_FACTOR;
# endif

  /* Initialize memory */
  //memset(&r, 0, sizeof(r));

  r.order = -1;
  if(zk     != NULL) r.order = 0;
  if(vrho   != NULL) r.order = 1;
  if(v2rho2 != NULL) r.order = 2;
  if(v3rho3 != NULL) r.order = 3;
  if(r.order < 0) return;

  idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx < np) {

    // Increment pointers
    rho += p->n_rho;
    if(zk != NULL){
      zk += p->n_zk * idx;
    }
    if(vrho != NULL){
      vrho += p->n_vrho * idx;
    }
    if(v2rho2 != NULL){
      v2rho2 += p->n_v2rho2 * idx;
    }
    if(v3rho3 != NULL){
      v3rho3 += p->n_v3rho3 * idx;
    }

    // TODO: remplazar xc_rho2zeta.
    //xc_rho2dzeta(p->nspin, rho, &dens, &r.z);
    if(p->nspin == XC_UNPOLARIZED){
	dens = max(rho[0], 0.0);
	r.z = 0.0;
    } else {
	dens = rho[0] + rho[1];
        if(dens > 0.0){
	    r.z = (rho[0] - rho[1])/(dens);
	    r.z = min(r.z,  1.0);
	    r.z = max(r.z, -1.0);
	}else{
	    dens = 0.0;
	    r.z = 0.0;
	}
    }

    if(dens < p->dens_threshold){
	return;
    }

    r.rs = cnst_rs*pow(dens, -1.0/XC_DIMENSIONS);

    // The functional algorithm goes here.
    func(p, &r);

    if(zk != NULL && (p->info->flags & XC_FLAGS_HAVE_EXC)) {
      *zk = r.f;
    }

    if(r.order < 1) {
	return;
    }

    drs = -r.rs/(XC_DIMENSIONS*dens);

    if(vrho != NULL && (p->info->flags & XC_FLAGS_HAVE_VXC)){
      vrho[0] = r.f + dens*r.dfdrs*drs;

      if(p->nspin == XC_POLARIZED){
	vrho[1] = vrho[0] - (r.z + 1.0)*r.dfdz;
	vrho[0] = vrho[0] - (r.z - 1.0)*r.dfdz;
      }
    }

    if(r.order < 2){
	return;
    }

    d2rs = -drs*(1.0 + XC_DIMENSIONS)/(XC_DIMENSIONS*dens);

    if(v2rho2 != NULL && (p->info->flags & XC_FLAGS_HAVE_FXC)){
      v2rho2[0] = r.dfdrs*(2.0*drs + dens*d2rs) + dens*r.d2fdrs2*drs*drs;
      
      if(p->nspin == XC_POLARIZED){
	double sign[3][2] = {{-1.0, -1.0}, {-1.0, +1.0}, {+1.0, +1.0}};
	
	for(is=2; is>=0; is--){
	  v2rho2[is] = v2rho2[0] - r.d2fdrsz*(2.0*r.z + sign[is][0] + sign[is][1])*drs
	    + (r.z + sign[is][0])*(r.z + sign[is][1])*r.d2fdz2/dens;
	}
      }
    }

    if(r.order < 3) {
	return;
    }

    d3rs = -d2rs*(1.0 + 2.0*XC_DIMENSIONS)/(XC_DIMENSIONS*dens);
    
    if(v3rho3 != NULL && (p->info->flags & XC_FLAGS_HAVE_KXC)){
      v3rho3[0] = r.dfdrs*(3.0*d2rs + dens*d3rs) + 
	3.0*r.d2fdrs2*drs*(drs + dens*d2rs) + r.d3fdrs3*dens*drs*drs*drs;
      
      if(p->nspin == XC_POLARIZED){
	double sign[4][3] = {{-1.0, -1.0, -1.0}, {-1.0, -1.0, +1.0}, {-1.0, +1.0, +1.0}, {+1.0, +1.0, +1.0}};
	
	for(is=3; is>=0; is--){
	  double ff;
	  
	  v3rho3[is]  = v3rho3[0] - (2.0*r.z  + sign[is][0] + sign[is][1])*(d2rs*r.d2fdrsz + drs*drs*r.d3fdrs2z);
	  v3rho3[is] += (r.z + sign[is][0])*(r.z + sign[is][1])*(-r.d2fdz2/dens + r.d3fdrsz2*drs)/dens;
	  
	  ff  = r.d2fdrsz*(2.0*drs + dens*d2rs) + dens*r.d3fdrs2z*drs*drs;
	  ff += -2.0*r.d2fdrsz*drs - r.d3fdrsz2*(2.0*r.z + sign[is][0] + sign[is][1])*drs;
	  ff += (r.z + sign[is][0])*(r.z + sign[is][1])*r.d3fdz3/dens;
	  ff += (2.0*r.z  + sign[is][0] + sign[is][1])*r.d2fdz2/dens;
	  
	  v3rho3[is] += -ff*(r.z + sign[is][2])/dens;
	}
      }
    }

    }//end if idx<np
}

/**
 * @param[in,out] func_type: pointer to pspdata structure to be initialized
 */
static void 
work_lda(const xc_func_type_cuda *p, int np, const double *rho, 
	 double *zk, double *vrho, double *v2rho2, double *v3rho3)
{
    //printf("work_lda(...)\n");

    // CUDA variables for the kernel.
    xc_func_info_type_cuda* infoTypeCUDA;
    xc_func_type_cuda* pCUDA;

    // Struct sizes.
    int xc_func_type_cuda_size = sizeof(xc_func_type_cuda);
    int info_type_size = sizeof(xc_func_info_type_cuda);

    // Alloc CUDA memory for infoTypeCUDA.
    checkCudaErrors(cudaMalloc((void **)&infoTypeCUDA, info_type_size));
    // Copy infoTypeCUDA to the device.
    checkCudaErrors(cudaMemcpy(infoTypeCUDA, (xc_func_info_type_cuda*)(p->info), info_type_size, cudaMemcpyHostToDevice));

    // Alloc CUDA memory fir xc_func_type_cuda.
    checkCudaErrors(cudaMalloc((void **)&pCUDA, xc_func_type_cuda_size));

    // Copy the first parameter to the device.
    checkCudaErrors(cudaMemcpy(pCUDA, p, xc_func_type_cuda_size, cudaMemcpyHostToDevice));

    // Now the make a "deep copy" of p->params_gpu.
    checkCudaErrors(cudaMemcpy(&(pCUDA->params_gpu), &(p->params_gpu), sizeof(p->params_gpu), cudaMemcpyHostToDevice));

    // Deep copy of infoTypeCUDA.
    checkCudaErrors(cudaMemcpy(&(pCUDA->info), &infoTypeCUDA, sizeof(infoTypeCUDA), cudaMemcpyHostToDevice));

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = MAX_THREAD_COUNT;
    int blocksPerGrid = (np + threadsPerBlock - 1) / threadsPerBlock;

    kernel_work_lda <<<blocksPerGrid, threadsPerBlock>>> (pCUDA,
	np,
	rho,
	zk,
	vrho,
	v2rho2,
	v3rho3);

    cudaError_t lastError = cudaGetLastError();

    if (lastError != cudaSuccess) {
	fprintf (stderr, "CUDA error %d \n", lastError);
    }

    // Free CUDA memory.
    checkCudaErrors(cudaFree(pCUDA));
    checkCudaErrors(cudaFree(infoTypeCUDA));

}
