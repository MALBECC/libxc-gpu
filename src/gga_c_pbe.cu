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
 Implements Perdew, Burke & Ernzerhof Generalized Gradient Approximation
 correlation functional.

 I based this implementation on a routine from L.C. Balbas and J.M. Soler
************************************************************************/

#define XC_GGA_C_PBE_cuda          1130 /* Perdew, Burke & Ernzerhof correlation              */
#define XC_GGA_C_PBE_SOL      1133 /* Perdew, Burke & Ernzerhof correlation SOL          */
#define XC_GGA_C_XPBE         1136 /* xPBE reparametrization by Xu & Goddard             */
#define XC_GGA_C_PBE_JRGX     1138 /* JRGX reparametrization by Pedroza, Silva & Capelle */
#define XC_GGA_C_RGE2         1143 /* Regularized PBE                                    */
#define XC_GGA_C_APBE         1186 /* mu fixed from the semiclassical neutral atom       */
#define XC_GGA_C_SPBE         1089 /* PBE correlation to be used with the SSB exchange   */
#define XC_GGA_C_PBEINT       1062 /* PBE for hybrid interfaces                          */
#define XC_GGA_C_PBEFE        1258 /* PBE for formation energies                         */
#define XC_GGA_C_PBE_MOL      1272 /* Del Campo, Gazquez, Trickey and Vela (PBE-like)    */
#define XC_GGA_C_TM_PBE       1560  /* Thakkar and McCarthy reparametrization */

typedef struct{
  double beta, gamma, BB;
} gga_c_pbe_params;

static void gga_c_pbe_init(xc_func_type_cuda *p)
{
  //printf ("gga_c_pbe_init() \n");
  gga_c_pbe_params *params;

  assert(p!=NULL && p->params == NULL);
  p->params = malloc(sizeof(gga_c_pbe_params));
  p->size_of_params_gpu = sizeof(gga_c_pbe_params);
  params = (gga_c_pbe_params *) (p->params);

  /* most functionals have the same gamma and B */
  params->gamma = (1.0 - log(2.0))/(M_PI*M_PI);
  params->BB = 1.0; 

  switch(p->info->number){
  case XC_GGA_C_PBE_cuda:
    params->beta = 0.06672455060314922;
    break;
  case XC_GGA_C_PBE_SOL:
    params->beta = 0.046;
    break;
  case XC_GGA_C_XPBE:
    params->beta  = 0.089809;
    params->gamma = params->beta*params->beta/(2.0*0.197363);
    break;
  case XC_GGA_C_PBE_JRGX:
    params->beta = 3.0*10.0/(81.0*M_PI*M_PI);
    break;
  case XC_GGA_C_RGE2:
    params->beta = 0.053;
    break;
  case XC_GGA_C_APBE:
    params->beta = 3.0*0.260/(M_PI*M_PI);
    break;
  case XC_GGA_C_SPBE:
    params->beta = 0.06672455060314922;
    /* the sPBE functional contains one term less than the original PBE, so we set it to zero */
    params->BB = 0.0; 
    break;
  case XC_GGA_C_PBEINT:
    params->beta = 0.052;
    break;
  case XC_GGA_C_PBEFE:
    params->beta = 0.043;
    break;
  case XC_GGA_C_PBE_MOL:
    params->beta = 0.08384;
    break;
  case XC_GGA_C_TM_PBE:
    params->gamma = -0.0156;
    params->beta  = 3.38*params->gamma;
    break;
  default:
    fprintf(stderr, "Internal error in gga_c_pbe\n");
    exit(1);
  }

  // Copy the params to device.
  checkCudaErrors(cudaMalloc((void**)(&(p->params_gpu)), p->size_of_params_gpu));
  checkCudaErrors(cudaMemcpy(p->params_gpu, p->params, p->size_of_params_gpu, cudaMemcpyHostToDevice));
}

static void gga_c_pbe_end(xc_func_type_cuda *p)
{
    //printf ("gga_c_pbe_end() \n");
    if (p->params_gpu != NULL) {
	checkCudaErrors(cudaFree(p->params_gpu));
    }
}

extern "C" void 
xc_gga_c_pbe_set_params(xc_func_type_cuda *p, double beta)
{
  //printf ("xc_gga_c_pbe_set_params \n");
  gga_c_pbe_params *params;

  assert(p != NULL && p->params != NULL);
  params = (gga_c_pbe_params *) (p->params);

  params->beta = beta;
}

#include "maple2c/gga_c_pbe.cu"

#define func maple2c_func
#include "work_gga_c.cuh"

extern "C" const xc_func_info_type_cuda xc_func_info_gga_c_pbe_cuda = {
  XC_GGA_C_PBE_cuda,
  XC_CORRELATION,
  (char*)"Perdew, Burke & Ernzerhof",
  XC_FAMILY_GGA,
  {&xc_ref_Perdew1996_3865, &xc_ref_Perdew1996_3865_err, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC,
  1e-12,
  0, NULL, NULL,
  gga_c_pbe_init, gga_c_pbe_end, 
  NULL, work_gga_c, NULL
};

extern "C" const xc_func_info_type_cuda xc_func_info_gga_c_pbe_sol = {
  XC_GGA_C_PBE_SOL,
  XC_CORRELATION,
  (char*)"Perdew, Burke & Ernzerhof SOL",
  XC_FAMILY_GGA,
  {&xc_ref_Perdew2008_136406, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC,
  1e-12,
  0, NULL, NULL,
  gga_c_pbe_init, gga_c_pbe_end, 
  NULL, work_gga_c, NULL
};

extern "C" const xc_func_info_type_cuda xc_func_info_gga_c_xpbe = {
  XC_GGA_C_XPBE,
  XC_CORRELATION,
  (char*)"Extended PBE by Xu & Goddard III",
  XC_FAMILY_GGA,
  {&xc_ref_Xu2004_4068, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC,
  1e-12,
  0, NULL, NULL,
  gga_c_pbe_init, gga_c_pbe_end, 
  NULL, work_gga_c, NULL
};

extern "C" const xc_func_info_type_cuda xc_func_info_gga_c_pbe_jrgx = {
  XC_GGA_C_PBE_JRGX,
  XC_CORRELATION,
  (char*)"Reparametrized PBE by Pedroza, Silva & Capelle",
  XC_FAMILY_GGA,
  {&xc_ref_Pedroza2009_201106, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC,
  1e-12,
  0, NULL, NULL,
  gga_c_pbe_init, gga_c_pbe_end, 
  NULL, work_gga_c, NULL
};

extern "C" const xc_func_info_type_cuda xc_func_info_gga_c_rge2 = {
  XC_GGA_C_RGE2,
  XC_CORRELATION,
  (char*)"Regularized PBE",
  XC_FAMILY_GGA,
  {&xc_ref_Ruzsinszky2009_763, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC,
  1e-12,
  0, NULL, NULL,
  gga_c_pbe_init, gga_c_pbe_end, 
  NULL, work_gga_c, NULL
};

extern "C" const xc_func_info_type_cuda xc_func_info_gga_c_apbe = {
  XC_GGA_C_APBE,
  XC_CORRELATION,
  (char*)"mu fixed from the semiclassical neutral atom",
  XC_FAMILY_GGA,
  {&xc_ref_Constantin2011_186406, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC,
  1e-12,
  0, NULL, NULL,
  gga_c_pbe_init, gga_c_pbe_end, 
  NULL, work_gga_c, NULL
};

extern "C" const xc_func_info_type_cuda xc_func_info_gga_c_spbe = {
  XC_GGA_C_SPBE,
  XC_CORRELATION,
  (char*)"PBE correlation to be used with the SSB exchange",
  XC_FAMILY_GGA,
  {&xc_ref_Swart2009_094103, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC,
  1e-12,
  0, NULL, NULL,
  gga_c_pbe_init, gga_c_pbe_end, 
  NULL, work_gga_c, NULL
};

extern "C" const xc_func_info_type_cuda xc_func_info_gga_c_pbeint = {
  XC_GGA_C_PBEINT,
  XC_CORRELATION,
  (char*)"PBE for hybrid interfaces",
  XC_FAMILY_GGA,
  {&xc_ref_Fabiano2010_113104, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC,
  1e-12,
  0, NULL, NULL,
  gga_c_pbe_init, gga_c_pbe_end, 
  NULL, work_gga_c, NULL
};

extern "C" const xc_func_info_type_cuda xc_func_info_gga_c_pbefe = {
  XC_GGA_C_PBEFE,
  XC_CORRELATION,
  (char*)"PBE for formation energies",
  XC_FAMILY_GGA,
  {&xc_ref_Perez2015_3844, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC,
  1e-12,
  0, NULL, NULL,
  gga_c_pbe_init, gga_c_pbe_end, 
  NULL, work_gga_c, NULL
};

extern "C" const xc_func_info_type_cuda xc_func_info_gga_c_pbe_mol = {
  XC_GGA_C_PBE_MOL,
  XC_CORRELATION,
  (char*)"Reparametrized PBE by del Campo, Gazquez, Trickey & Vela",
  XC_FAMILY_GGA,
  {&xc_ref_delCampo2012_104108, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC,
  1e-12,
  0, NULL, NULL,
  gga_c_pbe_init, gga_c_pbe_end, 
  NULL, work_gga_c, NULL
};

extern "C" const xc_func_info_type_cuda xc_func_info_gga_c_tm_pbe = {
  XC_GGA_C_TM_PBE,
  XC_CORRELATION,
  (char*)"Thakkar and McCarthy reparametrization",
  XC_FAMILY_GGA,
  {&xc_ref_Thakkar2009_134109, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC,
  1e-12,
  0, NULL, NULL,
  gga_c_pbe_init, gga_c_pbe_end, 
  NULL, work_gga_c, NULL
};
