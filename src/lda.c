/*
 Copyright (C) 2006-2007 M.A.L. Marques

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/


#include "util.h"
#include "funcs_lda.c"

/* get the lda functional */
void 
xc_lda_cuda(const xc_func_type_cuda *func, int np, const double *rho, 
	double *zk, double *vrho, double *v2rho2, double *v3rho3)
{
  assert(func != NULL);

  /* sanity check */
  if(zk != NULL && !(func->info->flags & XC_FLAGS_HAVE_EXC)){
    fprintf(stderr, "Functional '%s' does not provide an implementation of Exc\n",
	    func->info->name);
    exit(1);
  }

  if(vrho != NULL && !(func->info->flags & XC_FLAGS_HAVE_VXC)){
    fprintf(stderr, "Functional '%s' does not provide an implementation of vxc\n",
	    func->info->name);
    exit(1);
  }

  if(v2rho2 != NULL && !(func->info->flags & XC_FLAGS_HAVE_FXC)){
    fprintf(stderr, "Functional '%s' does not provide an implementation of fxc\n",
	    func->info->name);
    exit(1);
  }

  if(v3rho3 != NULL && !(func->info->flags & XC_FLAGS_HAVE_KXC)){
    fprintf(stderr, "Functional '%s' does not provide an implementation of kxc\n",
	    func->info->name);
    exit(1);
  }

  /* initialize output */
  /* This part is ommited since now the parameters are in CUDA memory.
  if(zk != NULL)
    memset(zk,     0, np*sizeof(double)*func->n_zk);

  if(vrho != NULL)
    memset(vrho,   0, np*sizeof(double)*func->n_vrho);

  if(v2rho2 != NULL)
    memset(v2rho2, 0, np*sizeof(double)*func->n_v2rho2);

  if(v3rho3 != NULL)
    memset(v3rho3, 0, np*sizeof(double)*func->n_v3rho3);
  */

  assert(func->info!=NULL && func->info->lda!=NULL);

  /* call the LDA routines */
  func->info->lda(func, np, rho, zk, vrho, v2rho2, v3rho3);
}


/* specializations */
inline void 
xc_lda_exc_cuda(const xc_func_type_cuda *p, int np, const double *rho, double *zk)
{
  xc_lda_cuda(p, np, rho, zk, NULL, NULL, NULL);
}

inline void 
xc_lda_exc_vxc_cuda(const xc_func_type_cuda *p, int np, const double *rho, double *zk, double *vrho)
{
  xc_lda_cuda(p, np, rho, zk, vrho, NULL, NULL);
}

inline void 
xc_lda_vxc_cuda(const xc_func_type_cuda *p, int np, const double *rho, double *vrho)
{
  xc_lda_cuda(p, np, rho, NULL, vrho, NULL, NULL);
}

inline void 
xc_lda_fxc_cuda(const xc_func_type_cuda *p, int np, const double *rho, double *v2rho2)
{
  xc_lda_cuda(p, np, rho, NULL, NULL, v2rho2, NULL);
}

inline void 
xc_lda_kxc_cuda(const xc_func_type_cuda *p, int np, const double *rho, double *v3rho3)
{
  xc_lda_cuda(p, np, rho, NULL, NULL, NULL, v3rho3);
}


#define DELTA_RHO 1e-6

/* get the xc kernel through finite differences */
void 
xc_lda_fxc_fd(const xc_func_type_cuda *func, int np, const double *rho, double *v2rho2)
{
  int i, ip;

  assert(func != NULL);

  for(ip=0; ip<np; ip++){
    for(i=0; i<func->nspin; i++){
      double rho2[2], vc1[2], vc2[2];
      int j, js;
      
      j  = (i+1) % 2;
      js = (i==0) ? 0 : 2;
      
      rho2[i] = rho[i] + DELTA_RHO;
      rho2[j] = (func->nspin == XC_POLARIZED) ? rho[j] : 0.0;
      xc_lda_vxc_cuda(func, 1, rho2, vc1);
      
      if(rho[i]<2.0*DELTA_RHO){ /* we have to use a forward difference */
	xc_lda_vxc_cuda(func, 1, rho, vc2);
	
	v2rho2[js] = (vc1[i] - vc2[i])/(DELTA_RHO);
	if(func->nspin == XC_POLARIZED && i==0)
	  v2rho2[1] = (vc1[j] - vc2[j])/(DELTA_RHO);
	
      }else{                    /* centered difference (more precise)  */
	rho2[i] = rho[i] - DELTA_RHO;
	xc_lda_vxc_cuda(func, 1, rho2, vc2);
      
	v2rho2[js] = (vc1[i] - vc2[i])/(2.0*DELTA_RHO);
	if(func->nspin == XC_POLARIZED && i==0)
	  v2rho2[1] = (vc1[j] - vc2[j])/(2.0*DELTA_RHO);
      }
    }

    rho    += func->n_rho;
    v2rho2 += func->n_v2rho2;
  } /* for(ip) */
}


void
xc_lda_kxc_fd(const xc_func_type_cuda *func, int np, const double *rho, double *v3rho3)
{
  /* Kxc, this is a third order tensor with respect to the densities */
  int ip, i, j, n;

  assert(func != NULL);

  for(ip=0; ip<np; ip++){
    for(i=0; i<func->nspin; i++){
      double rho2[2], vc1[2], vc2[2], vc3[2];

      for(n=0; n<func->nspin; n++) rho2[n] = rho[n];
      xc_lda_vxc_cuda(func, 1, rho, vc2);

      rho2[i] += DELTA_RHO;
      xc_lda_vxc_cuda(func, 1, rho2, vc1);
	
      rho2[i] -= 2.0*DELTA_RHO;
      xc_lda_vxc_cuda(func, 1, rho2, vc3);    
    
      for(j=0; j<func->nspin; j++)
	v3rho3[i*func->nspin + j] = (vc1[j] - 2.0*vc2[j] + vc3[j])/(DELTA_RHO*DELTA_RHO);
    }
    
    rho    += func->n_rho;
    v3rho3 += func->n_v3rho3;
  } /* for(ip) */
}