/*
 Copyright (C) 2006-2007 M.A.L. Marques

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <cuda_runtime.h>
#include "helper/helper_cuda.h"
#include "util.h"

/************************************************************************
 Implements Tao, Perdew, Staroverov & Scuseria 
   meta-Generalized Gradient Approximation.

  Exchange part
************************************************************************/

#define XC_MGGA_X_TPSS          1202 /* Tao, Perdew, Staroverov & Scuseria exchange */
#define XC_MGGA_X_MODTPSS       1245 /* Modified Tao, Perdew, Staroverov & Scuseria exchange */
#define XC_MGGA_X_REVTPSS       1212 /* revised Tao, Perdew, Staroverov & Scuseria exchange */
#define XC_MGGA_X_BLOC          1244 /* functional with balanced localization */

typedef struct{
  double b, c, e, kappa, mu;
  double BLOC_a, BLOC_b;
} mgga_x_tpss_params;

extern "C" void
xc_mgga_x_tpss_set_params(xc_func_type_cuda *p, double b, double c, double e, double kappa, double mu, 
                           double BLOC_a, double BLOC_b)
{
  mgga_x_tpss_params *params;

  assert(p != NULL && p->params != NULL);
  params = (mgga_x_tpss_params *) (p->params);

  params->b      = b;
  params->c      = c;
  params->e      = e;
  params->kappa  = kappa;
  params->mu     = mu;
  params->BLOC_a = BLOC_a;
  params->BLOC_b = BLOC_b;
}

static void 
mgga_x_tpss_init(xc_func_type_cuda *p)
{
  printf("mgga_x_tpss_init(...)\n");
  mgga_x_tpss_params *params;

  assert(p!=NULL && p->params == NULL);
  p->params = malloc(sizeof(mgga_x_tpss_params));
  p->size_of_params_gpu = sizeof(mgga_x_tpss_params);

  params = (mgga_x_tpss_params *)p->params;

  switch(p->info->number){
  case XC_MGGA_X_TPSS:
    xc_mgga_x_tpss_set_params(p, 0.40, 1.59096, 1.537, 0.804, 0.21951, 2.0, 0.0);
    break;
  case XC_MGGA_X_MODTPSS:
    xc_mgga_x_tpss_set_params(p, 0.40, 1.38496, 1.37, 0.804, 0.252, 2.0, 0.0);
    break;
  case XC_MGGA_X_REVTPSS:
    xc_mgga_x_tpss_set_params(p, 0.40, 2.35203946, 2.16769874, 0.804, 0.14, 3.0, 0.0);
    break;
  case XC_MGGA_X_BLOC:
    xc_mgga_x_tpss_set_params(p, 0.40, 1.59096, 1.537, 0.804, 0.21951, 4.0, -3.3);
    break;
  default:
    fprintf(stderr, "Internal error in mgga_x_tpss\n");
    exit(1);
  }

  // Alloc memory for params on the device.
  checkCudaErrors(cudaMalloc((void**)(&(p->params_gpu)), p->size_of_params_gpu));
  // Copy params to the device.
  checkCudaErrors(cudaMemcpy(p->params_gpu, p->params, p->size_of_params_gpu, cudaMemcpyHostToDevice));
}

static void 
mgga_x_tpss_end(xc_func_type_cuda *p)
{
  printf("mgga_x_tpss_end(...)\n");
  mgga_x_tpss_params *params;

  assert(p!=NULL && p->params_gpu != NULL);
  checkCudaErrors(cudaFree(p->params_gpu));
}

#include "maple2c/mgga_x_tpss.cu"

#define func xc_mgga_x_tpss_enhance
#include "work_mgga_x.cuh"

extern "C" const xc_func_info_type_cuda xc_func_info_mgga_x_tpss = {
  XC_MGGA_X_TPSS,
  XC_EXCHANGE,
  "Tao, Perdew, Staroverov & Scuseria",
  XC_FAMILY_MGGA,
  {&xc_ref_Tao2003_146401, &xc_ref_Perdew2004_6898, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC,
  1e-23,
  0, NULL, NULL,
  mgga_x_tpss_init, mgga_x_tpss_end, 
  NULL, NULL, work_mgga_x,
};

extern "C" const xc_func_info_type_cuda xc_func_info_mgga_x_modtpss = {
  XC_MGGA_X_MODTPSS,
  XC_EXCHANGE,
  "Modified Tao, Perdew, Staroverov & Scuseria",
  XC_FAMILY_MGGA,
  {&xc_ref_Perdew2007_042506, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC,
  1e-23,
  0, NULL, NULL,
  mgga_x_tpss_init, mgga_x_tpss_end, 
  NULL, NULL, work_mgga_x,
};

extern "C" const xc_func_info_type_cuda xc_func_info_mgga_x_revtpss = {
  XC_MGGA_X_REVTPSS,
  XC_EXCHANGE,
  "revised Tao, Perdew, Staroverov & Scuseria",
  XC_FAMILY_MGGA,
  {&xc_ref_Perdew2009_026403, &xc_ref_Perdew2009_026403_err, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC,
  1e-23,
  0, NULL, NULL,
  mgga_x_tpss_init, mgga_x_tpss_end, 
  NULL, NULL, work_mgga_x,
};

extern "C" const xc_func_info_type_cuda xc_func_info_mgga_x_bloc = {
  XC_MGGA_X_BLOC,
  XC_EXCHANGE,
  "functional with balanced localization",
  XC_FAMILY_MGGA,
  {&xc_ref_Constantin2013_2256, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC,
  1e-23,
  0, NULL, NULL,
  mgga_x_tpss_init, mgga_x_tpss_end, 
  NULL, NULL, work_mgga_x,
};
