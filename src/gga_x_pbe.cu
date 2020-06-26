/*
 Copyright (C) 2006-2007 M.A.L. Marques

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

// CUDA runtime
#include <cuda_runtime.h>
#include "helper/helper_cuda.h"
#include "util.h"

#define XC_GGA_X_PBE_cuda          1101 /* Perdew, Burke & Ernzerhof exchange             */
#define XC_GGA_X_PBE_R        1102 /* Perdew, Burke & Ernzerhof exchange (revised)   */
#define XC_GGA_X_PBE_SOL      1116 /* Perdew, Burke & Ernzerhof exchange (solids)    */
#define XC_GGA_X_XPBE         1123 /* xPBE reparametrization by Xu & Goddard         */
#define XC_GGA_X_PBE_JSJR     1126 /* JSJR reparametrization by Pedroza, Silva & Capelle */
#define XC_GGA_X_PBEK1_VDW    1140 /* PBE reparametrization for vdW                  */
#define XC_GGA_X_APBE         1184 /* mu fixed from the semiclassical neutral atom   */
#define XC_GGA_X_PBE_TCA      1059 /* PBE revised by Tognetti et al                  */
#define XC_GGA_X_PBE_MOL      1049 /* Del Campo, Gazquez, Trickey and Vela (PBE-like) */
#define XC_GGA_X_LAMBDA_LO_N  1045 /* lambda_LO(N) version of PBE                    */
#define XC_GGA_X_LAMBDA_CH_N  1044 /* lambda_CH(N) version of PBE                    */
#define XC_GGA_X_LAMBDA_OC2_N 1040 /* lambda_OC2(N) version of PBE                   */
#define XC_GGA_X_BCGP         1038 /* Burke, Cancio, Gould, and Pittalis             */
#define XC_GGA_X_PBEFE        1265 /* PBE for formation energies                     */

#define XC_GGA_K_APBE         1185 /* mu fixed from the semiclassical neutral atom   */
#define XC_GGA_K_TW1          1187 /* Tran and Wesolowski set 1 (Table II)           */
#define XC_GGA_K_TW2          1188 /* Tran and Wesolowski set 2 (Table II)           */
#define XC_GGA_K_TW3          1189 /* Tran and Wesolowski set 3 (Table II)           */
#define XC_GGA_K_TW4          1190 /* Tran and Wesolowski set 4 (Table II)           */
#define XC_GGA_K_REVAPBE      1055 /* revised APBE                                   */


typedef struct{
  double kappa, mu;

  /* parameter used in the Odashima & Capelle versions */
  double lambda;
} gga_x_pbe_params;


extern "C" void 
xc_gga_x_pbe_set_params(xc_func_type_cuda *p, double kappa, double mu)
{
  gga_x_pbe_params *params;

  assert(p != NULL && p->params != NULL);
  params = (gga_x_pbe_params *) (p->params);

  params->kappa = kappa;
  params->mu    = mu;
}


static void 
gga_x_pbe_init(xc_func_type_cuda *p)
{
  //printf("gga_x_pbe_init()\n");
  gga_x_pbe_params *params;

  assert(p!=NULL && p->params == NULL);
  p->params = malloc(sizeof(gga_x_pbe_params));
  p->size_of_params_gpu = sizeof(gga_x_pbe_params);
  params = (gga_x_pbe_params *) (p->params);
 
  params->lambda = 0.0;

  switch(p->info->number){
  case XC_GGA_X_PBE_cuda:
    /* PBE: mu = beta*pi^2/3, beta = 0.06672455060314922 */
    xc_gga_x_pbe_set_params(p, 0.8040, 0.2195149727645171);
    break;
  case XC_GGA_X_PBE_R:
    xc_gga_x_pbe_set_params(p, 1.245, 0.2195149727645171);
    break;
  case XC_GGA_X_PBE_SOL:
    xc_gga_x_pbe_set_params(p, 0.804, MU_GE);
    break;
  case XC_GGA_X_XPBE:
    xc_gga_x_pbe_set_params(p, 0.91954, 0.23214);
    break;
  case XC_GGA_X_PBE_JSJR:
    xc_gga_x_pbe_set_params(p, 0.8040, 0.046*M_PI*M_PI/3.0);
    break;
  case XC_GGA_X_PBEK1_VDW:
    xc_gga_x_pbe_set_params(p, 1.0, 0.2195149727645171);
    break;
  case XC_GGA_X_APBE:
    xc_gga_x_pbe_set_params(p, 0.8040, 0.260);
    break;
  case XC_GGA_K_APBE:
    xc_gga_x_pbe_set_params(p, 0.8040, 0.23889);
    break;
  case XC_GGA_K_TW1:
    xc_gga_x_pbe_set_params(p, 0.8209, 0.2335);
    break;
  case XC_GGA_K_TW2:
    xc_gga_x_pbe_set_params(p, 0.6774, 0.2371);
    break;
  case XC_GGA_K_TW3:
    xc_gga_x_pbe_set_params(p, 0.8438, 0.2319);
    break;
  case XC_GGA_K_TW4:
    xc_gga_x_pbe_set_params(p, 0.8589, 0.2309);
    break;
  case XC_GGA_X_PBE_TCA:
    xc_gga_x_pbe_set_params(p, 1.227, 0.2195149727645171);
    break;
  case XC_GGA_K_REVAPBE:
    xc_gga_x_pbe_set_params(p, 1.245, 0.23889);
    break;
  case XC_GGA_X_PBE_MOL:
    xc_gga_x_pbe_set_params(p, 0.8040, 0.27583);
    break;
  case XC_GGA_X_LAMBDA_LO_N:
    xc_gga_x_pbe_set_params(p, -1.0, 0.2195149727645171);
    params->lambda = 2.273;
    break;
  case XC_GGA_X_LAMBDA_CH_N:
    xc_gga_x_pbe_set_params(p, -1.0, 0.2195149727645171);
    params->lambda = 2.215;
    break;
  case XC_GGA_X_LAMBDA_OC2_N:
    xc_gga_x_pbe_set_params(p, -1.0, 0.2195149727645171);
    params->lambda = 2.00;
    break;
  case XC_GGA_X_BCGP:
    xc_gga_x_pbe_set_params(p, 0.8040, 0.249);
    break;
  case XC_GGA_X_PBEFE:
    xc_gga_x_pbe_set_params(p, 0.437, 0.346);
    break;
  default:
    fprintf(stderr, "Internal error in gga_x_pbe\n");
    exit(1);
  }
  // Copy the params to device.
  checkCudaErrors(cudaMalloc((void**)(&(p->params_gpu)), p->size_of_params_gpu));
  checkCudaErrors(cudaMemcpy(p->params_gpu, p->params, p->size_of_params_gpu, cudaMemcpyHostToDevice));
}

static void gga_x_pbe_end(xc_func_type_cuda *p)
{
    //printf ("gga_x_pbe_end() \n");
    if (p->params_gpu != NULL) {
	checkCudaErrors(cudaFree(p->params_gpu));
    }
}

#include "maple2c/gga_x_pbe.cu"

#define func xc_gga_x_pbe_enhance
#include "work_gga_x_cuda.cuh"


extern "C" const xc_func_info_type_cuda xc_func_info_gga_x_pbe_cuda = {
  XC_GGA_X_PBE_cuda,
  XC_EXCHANGE,
  (char*)"Perdew, Burke & Ernzerhof",
  XC_FAMILY_GGA,
  {&xc_ref_Perdew1996_3865, &xc_ref_Perdew1996_3865_err, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC,
  1e-32,
  0, NULL, NULL,
  gga_x_pbe_init, gga_x_pbe_end, 
  NULL, work_gga_x_cuda, NULL
};

extern "C" const xc_func_info_type_cuda xc_func_info_gga_x_pbe_r = {
  XC_GGA_X_PBE_R,
  XC_EXCHANGE,
  (char*)"Revised PBE from Zhang & Yang",
  XC_FAMILY_GGA,
  {&xc_ref_Zhang1998_890, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC,
  1e-32,
  0, NULL, NULL,
  gga_x_pbe_init, gga_x_pbe_end, 
  NULL, work_gga_x_cuda, NULL
};

extern "C" const xc_func_info_type_cuda xc_func_info_gga_x_pbe_sol = {
  XC_GGA_X_PBE_SOL,
  XC_EXCHANGE,
  (char*)"Perdew, Burke & Ernzerhof SOL",
  XC_FAMILY_GGA,
  {&xc_ref_Perdew2008_136406, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC,
  1e-32,
  0, NULL, NULL,
  gga_x_pbe_init, gga_x_pbe_end, 
  NULL, work_gga_x_cuda, NULL
};

extern "C" const xc_func_info_type_cuda xc_func_info_gga_x_xpbe = {
  XC_GGA_X_XPBE,
  XC_EXCHANGE,
  (char*)"Extended PBE by Xu & Goddard III",
  XC_FAMILY_GGA,
  {&xc_ref_Xu2004_4068, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC,
  1e-32,
  0, NULL, NULL,
  gga_x_pbe_init, gga_x_pbe_end, 
  NULL, work_gga_x_cuda, NULL
};

extern "C" const xc_func_info_type_cuda xc_func_info_gga_x_pbe_jsjr = {
  XC_GGA_X_PBE_JSJR,
  XC_EXCHANGE,
  (char*)"Reparametrized PBE by Pedroza, Silva & Capelle",
  XC_FAMILY_GGA,
  {&xc_ref_Pedroza2009_201106, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC,
  1e-32,
  0, NULL, NULL,
  gga_x_pbe_init, gga_x_pbe_end, 
  NULL, work_gga_x_cuda, NULL
};

extern "C" const xc_func_info_type_cuda xc_func_info_gga_x_pbek1_vdw = {
  XC_GGA_X_PBEK1_VDW,
  XC_EXCHANGE,
  (char*)"Reparametrized PBE for vdW",
  XC_FAMILY_GGA,
  {&xc_ref_Klimes2010_022201, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC,
  1e-32,
  0, NULL, NULL,
  gga_x_pbe_init, gga_x_pbe_end, 
  NULL, work_gga_x_cuda, NULL
};

extern "C" const xc_func_info_type_cuda xc_func_info_gga_x_apbe = {
  XC_GGA_X_APBE,
  XC_EXCHANGE,
  (char*)"mu fixed from the semiclassical neutral atom",
  XC_FAMILY_GGA,
  {&xc_ref_Constantin2011_186406, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC,
  1e-32,
  0, NULL, NULL,
  gga_x_pbe_init, gga_x_pbe_end, 
  NULL, work_gga_x_cuda, NULL
};

extern "C" const xc_func_info_type_cuda xc_func_info_gga_x_pbe_tca = {
  XC_GGA_X_PBE_TCA,
  XC_EXCHANGE,
  (char*)"PBE revised by Tognetti et al",
  XC_FAMILY_GGA,
  {&xc_ref_Tognetti2008_536, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC,
  1e-32,
  0, NULL, NULL,
  gga_x_pbe_init, gga_x_pbe_end, 
  NULL, work_gga_x_cuda, NULL
};

static const func_params_type_cuda ext_params[] = {
  {1e23, (char*)"Number of electrons"},
};

static void 
set_ext_params(xc_func_type_cuda *p, const double *ext_params)
{
  const double lambda_1 = 1.48;

  gga_x_pbe_params *params;
  double lambda, ff;

  assert(p != NULL && p->params != NULL);
  params = (gga_x_pbe_params *) (p->params);

  ff = (ext_params == NULL) ? p->info->ext_params[0].value : ext_params[0];

  lambda = (1.0 - 1.0/ff)*params->lambda + lambda_1/ff;
  params->kappa = lambda/M_CBRT2 - 1.0;
}

extern "C" const xc_func_info_type_cuda xc_func_info_gga_x_lambda_lo_n = {
  XC_GGA_X_LAMBDA_LO_N,
  XC_EXCHANGE,
  (char*)"lambda_LO(N) version of PBE",
  XC_FAMILY_GGA,
  {&xc_ref_Odashima2009_798, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC,
  1e-32,
  1, ext_params, set_ext_params,
  gga_x_pbe_init, gga_x_pbe_end, 
  NULL, work_gga_x_cuda, NULL
};

extern "C" const xc_func_info_type_cuda xc_func_info_gga_x_lambda_ch_n = {
  XC_GGA_X_LAMBDA_CH_N,
  XC_EXCHANGE,
  (char*)"lambda_CH(N) version of PBE",
  XC_FAMILY_GGA,
  {&xc_ref_Odashima2009_798, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC,
  1e-32,
  1, ext_params, set_ext_params,
  gga_x_pbe_init, gga_x_pbe_end, 
  NULL, work_gga_x_cuda, NULL
};

extern "C" const xc_func_info_type_cuda xc_func_info_gga_x_lambda_oc2_n = {
  XC_GGA_X_LAMBDA_OC2_N,
  XC_EXCHANGE,
  (char*)"lambda_OC2(N) version of PBE",
  XC_FAMILY_GGA,
  {&xc_ref_Odashima2009_798, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC,
  1e-32,
  1, ext_params, set_ext_params,
  gga_x_pbe_init, gga_x_pbe_end, 
  NULL, work_gga_x_cuda, NULL
};

extern "C" const xc_func_info_type_cuda xc_func_info_gga_x_pbe_mol = {
  XC_GGA_X_PBE_MOL,
  XC_EXCHANGE,
  (char*)"Reparametrized PBE by del Campo, Gazquez, Trickey & Vela",
  XC_FAMILY_GGA,
  {&xc_ref_delCampo2012_104108, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC,
  1e-32,
  0, NULL, NULL,
  gga_x_pbe_init, gga_x_pbe_end, 
  NULL, work_gga_x_cuda, NULL
};

extern "C" const xc_func_info_type_cuda xc_func_info_gga_x_bcgp = {
  XC_GGA_X_BCGP,
  XC_EXCHANGE,
  (char*)"Burke, Cancio, Gould, and Pittalis",
  XC_FAMILY_GGA,
  {&xc_ref_Burke2014_4834, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC,
  1e-32,
  0, NULL, NULL,
  gga_x_pbe_init, gga_x_pbe_end, 
  NULL, work_gga_x_cuda, NULL
};

extern "C" const xc_func_info_type_cuda xc_func_info_gga_x_pbefe = {
  XC_GGA_X_PBEFE,
  XC_EXCHANGE,
  (char*)"PBE for formation energies",
  XC_FAMILY_GGA,
  {&xc_ref_Perez2015_3844, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC,
  1e-32,
  0, NULL, NULL,
  gga_x_pbe_init, gga_x_pbe_end, 
  NULL, work_gga_x_cuda, NULL
};

#define XC_KINETIC_FUNCTIONAL
#include "work_gga_x_cuda.cuh"

extern "C" const xc_func_info_type_cuda xc_func_info_gga_k_apbe = {
  XC_GGA_K_APBE,
  XC_KINETIC,
  (char*)"mu fixed from the semiclassical neutral atom",
  XC_FAMILY_GGA,
  {&xc_ref_Constantin2011_186406, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC,
  1e-32,
  0, NULL, NULL,
  gga_x_pbe_init, gga_x_pbe_end, 
  NULL, work_gga_k, NULL
};

extern "C" const xc_func_info_type_cuda xc_func_info_gga_k_revapbe = {
  XC_GGA_K_REVAPBE,
  XC_KINETIC,
  (char*)"revised APBE",
  XC_FAMILY_GGA,
  {&xc_ref_Constantin2011_186406, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC,
  1e-32,
  0, NULL, NULL,
  gga_x_pbe_init, gga_x_pbe_end, 
  NULL, work_gga_k, NULL
};

extern "C" const xc_func_info_type_cuda xc_func_info_gga_k_tw1 = {
  XC_GGA_K_TW1,
  XC_KINETIC,
  (char*)"Tran and Wesolowski set 1 (Table II)",
  XC_FAMILY_GGA,
  {&xc_ref_Tran2002_441, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC,
  1e-32,
  0, NULL, NULL,
  gga_x_pbe_init, gga_x_pbe_end, 
  NULL, work_gga_k, NULL
};

extern "C" const xc_func_info_type_cuda xc_func_info_gga_k_tw2 = {
  XC_GGA_K_TW2,
  XC_KINETIC,
  (char*)"Tran and Wesolowski set 2 (Table II)",
  XC_FAMILY_GGA,
  {&xc_ref_Tran2002_441, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC,
  1e-32,
  0, NULL, NULL,
  gga_x_pbe_init, gga_x_pbe_end, 
  NULL, work_gga_k, NULL
};

extern "C" const xc_func_info_type_cuda xc_func_info_gga_k_tw3 = {
  XC_GGA_K_TW3,
  XC_KINETIC,
  (char*)"Tran and Wesolowski set 3 (Table II)",
  XC_FAMILY_GGA,
  {&xc_ref_Tran2002_441, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC,
  1e-32,
  0, NULL, NULL,
  gga_x_pbe_init, gga_x_pbe_end, 
  NULL, work_gga_k, NULL
};

extern "C" const xc_func_info_type_cuda xc_func_info_gga_k_tw4 = {
  XC_GGA_K_TW4,
  XC_KINETIC,
  (char*)"Tran and Wesolowski set 4 (Table II)",
  XC_FAMILY_GGA,
  {&xc_ref_Tran2002_441, NULL, NULL, NULL, NULL},
  XC_FLAGS_3D | XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC,
  1e-32,
  0, NULL, NULL,
  gga_x_pbe_init, gga_x_pbe_end, 
  NULL, work_gga_k, NULL
};
