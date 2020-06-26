/*
 Copyright (C) 2008 M.A.L. Marques

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <cuda_runtime.h>
#include "helper/helper_cuda.h"
#include "util.h"

#define XC_MGGA_C_BC95         1240 /* Becke correlation 95 */

typedef struct{
  double css, copp;
} mgga_c_bc95_params;


extern "C" void xc_mgga_c_bc95_set_params(xc_func_type_cuda *p, double css, double copp)
{
  printf("xc_mgga_c_bc95_set_params(...)");
  mgga_c_bc95_params *params;

  assert(p != NULL && p->params != NULL);
  params = (mgga_c_bc95_params *) (p->params);

  params->css  = css;
  params->copp = copp;

  // Copy data to cuda memory.
  checkCudaErrors(cudaMemcpy(p->params_gpu, p->params, p->size_of_params_gpu, cudaMemcpyHostToDevice));
}

static void mgga_c_bc95_init(xc_func_type_cuda *p)
{
  printf("mmga_c_bc95_init(...) \n");

  assert(p!=NULL && p->params == NULL);
  p->params = malloc(sizeof(mgga_c_bc95_params));

  p->size_of_params_gpu = sizeof(mgga_c_bc95_params);

  // Alloc memory for params_gpy in the device memory.
  checkCudaErrors(cudaMalloc((void**)(&(p->params_gpu)), p->size_of_params_gpu));

  xc_mgga_c_bc95_set_params(p, 0.038, 0.0031);

}

static void mgga_c_bc95_end(xc_func_type_cuda *p)
{
  printf("mmga_c_bc95_end(...) \n");
  assert(p!=NULL && p->params_gpu != NULL);
  // Free device memory.
  checkCudaErrors(cudaFree(p->params_gpu));
}

#include "maple2c/mgga_c_bc95.cu"

#define func maple2c_func
#include "work_mgga_c.cuh"

extern "C" const xc_func_info_type_cuda xc_func_info_mgga_c_bc95 = {
  XC_MGGA_C_BC95,
  XC_CORRELATION,
  "Becke correlation 95",
  XC_FAMILY_MGGA,
  {&xc_ref_Becke1996_1040, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC,
  1e-23,
  0, NULL, NULL,
  mgga_c_bc95_init,
  mgga_c_bc95_end, NULL, NULL,
  work_mgga_c,
};

