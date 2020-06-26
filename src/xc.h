/*
 Copyright (C) 2006-2007 M.A.L. Marques

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#ifndef _XC_CUDA_H
#define _XC_CUDA_H

#ifdef __cplusplus
extern "C" {
#endif

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <xc_version.h>

#define XC_UNPOLARIZED          1
#define XC_POLARIZED            2

#define XC_NON_RELATIVISTIC     0
#define XC_RELATIVISTIC         1

#define XC_EXCHANGE             0
#define XC_CORRELATION          1
#define XC_EXCHANGE_CORRELATION 2
#define XC_KINETIC              3

#define XC_FAMILY_UNKNOWN      -1
#define XC_FAMILY_LDA           1
#define XC_FAMILY_GGA           2
#define XC_FAMILY_MGGA          4
#define XC_FAMILY_LCA           8
#define XC_FAMILY_OEP          16
#define XC_FAMILY_HYB_GGA      32
#define XC_FAMILY_HYB_MGGA     64

/* flags that can be used in info.flags. Don't reorder these since it
   will break the ABI of the library. */
#define XC_FLAGS_HAVE_EXC         (1 <<  0) /*     1 */
#define XC_FLAGS_HAVE_VXC         (1 <<  1) /*     2 */
#define XC_FLAGS_HAVE_FXC         (1 <<  2) /*     4 */
#define XC_FLAGS_HAVE_KXC         (1 <<  3) /*     8 */
#define XC_FLAGS_HAVE_LXC         (1 <<  4) /*    16 */
#define XC_FLAGS_1D               (1 <<  5) /*    32 */
#define XC_FLAGS_2D               (1 <<  6) /*    64 */
#define XC_FLAGS_3D               (1 <<  7) /*   128 */
#define XC_FLAGS_HYB_CAM          (1 <<  8) /*   256 */
#define XC_FLAGS_HYB_CAMY         (1 <<  9) /*   512 */
#define XC_FLAGS_VV10             (1 << 10) /*  1024 */
#define XC_FLAGS_HYB_LC           (1 << 11) /*  2048 */
#define XC_FLAGS_HYB_LCY          (1 << 12) /*  4096 */
#define XC_FLAGS_STABLE           (1 << 13) /*  8192 */
#define XC_FLAGS_DEVELOPMENT      (1 << 14) /* 16384 */
#define XC_FLAGS_NEEDS_LAPLACIAN  (1 << 15) /* 32768 */

#define XC_TAU_EXPLICIT         0
#define XC_TAU_EXPANSION        1

#define XC_MAX_REFERENCES       5

void xc_version(int *major, int *minor, int *micro);
const char *xc_version_string();

struct xc_func_type_cuda;
struct xc_func_type_cuda_gga;

typedef struct {
  char *ref, *doi, *bibtex;
} func_reference_type_cuda;

char const *xc_func_reference_get_ref_cuda(const func_reference_type_cuda *reference);
char const *xc_func_reference_get_doi_cuda(const func_reference_type_cuda *reference);
char const *xc_func_reference_get_bibtex_cuda(const func_reference_type_cuda *reference);

typedef struct {
  double value;
  char *description;
} func_params_type_cuda;

#ifdef __CUDACC__
typedef struct __align__(128) {
#else
typedef struct {
#endif
  int   number;   /* identifier number */
  int   kind;     /* XC_EXCHANGE, XC_CORRELATION, XC_EXCHANGE_CORRELATION, XC_KINETIC */

  char *name;     /* name of the functional, e.g. "PBE" */
  int   family;   /* type of the functional, e.g. XC_FAMILY_GGA */
  func_reference_type_cuda *refs[XC_MAX_REFERENCES];  /* index of the references */

  int   flags;    /* see above for a list of possible flags */

  double dens_threshold;

  /* this allows to have external parameters in the functional */
  int n_ext_params;
  const func_params_type_cuda *ext_params;
  void (*set_ext_params)(struct xc_func_type_cuda *p, const double *ext_params);

  void (*init)(struct xc_func_type_cuda *p);
  void (*end) (struct xc_func_type_cuda *p);
  void (*lda) (const struct xc_func_type_cuda *p, int np,
	       const double *rho,
	       double *zk, double *vrho, double *v2rho2, double *v3rho3);
  void (*gga) (const struct xc_func_type_cuda *p, int np,
	       const double *rho, const double *sigma,
	       double *zk, double *vrho, double *vsigma,
	       double *v2rho2, double *v2rhosigma, double *v2sigma2,
	       double *v3rho3, double *v3rho2sigma, double *v3rhosigma2, double *v3sigma3);
  void (*mgga)(const struct xc_func_type_cuda *p, int np,
	       const double *rho, const double *sigma, const double *lapl_rho, const double *tau,
	       double *zk, double *vrho, double *vsigma, double *vlapl_rho, double *vtau,
	       double *v2rho2, double *v2sigma2, double *v2lapl2, double *v2tau2,
	       double *v2rhosigma, double *v2rholapl, double *v2rhotau,
	       double *v2sigmalapl, double *v2sigmatau, double *v2lapltau);
} xc_func_info_type_cuda;

/* for API compability with older versions of libxc */
#define XC(func) xc_ ## func

int xc_func_info_get_number_cuda(const xc_func_info_type_cuda *info);
int xc_func_info_get_kind_cuda(const xc_func_info_type_cuda *info);
char const *xc_func_info_get_name_cuda(const xc_func_info_type_cuda *info);
int xc_func_info_get_family_cuda(const xc_func_info_type_cuda *info);
int xc_func_info_get_flags_cuda(const xc_func_info_type_cuda *info);
const func_reference_type_cuda *xc_func_info_get_references_cuda(const xc_func_info_type_cuda *info, int number);
int xc_func_info_get_n_ext_params_cuda(xc_func_info_type_cuda *info);
char const *xc_func_info_get_ext_params_description_cuda(xc_func_info_type_cuda *info, int number);
double xc_func_info_get_ext_params_default_value_cuda(xc_func_info_type_cuda *info, int number);

#ifdef __CUDACC__
struct __align__(256) xc_func_type_cuda {
#else
struct xc_func_type_cuda {
#endif
  const xc_func_info_type_cuda *info;       /* all the information concerning this functional */
  int nspin;                            /* XC_UNPOLARIZED or XC_POLARIZED  */

  int n_func_aux;                       /* how many auxiliary functions we need */
  struct xc_func_type_cuda **func_aux;      /* most GGAs are based on a LDA or other GGAs  */
  double *mix_coef;                      /* coefficients for the mixing */

  double cam_omega;                      /* range-separation parameter for range-separated hybrids */
  double cam_alpha;                      /* fraction of Hartree-Fock exchange for normal or range-separated hybrids */
  double cam_beta;                       /* fraction of short-range exchange for range-separated hybrids */

  double nlc_b;                          /* Non-local correlation, b parameter */
  double nlc_C;                          /* Non-local correlation, C parameter */

  int n_rho, n_sigma, n_tau, n_lapl;    /* spin dimensions of the arrays */
  int n_zk;

  int n_vrho, n_vsigma, n_vtau, n_vlapl;

  int n_v2rho2, n_v2sigma2, n_v2tau2, n_v2lapl2,
    n_v2rhosigma, n_v2rhotau, n_v2rholapl,
    n_v2sigmatau, n_v2sigmalapl, n_v2lapltau;

  int n_v3rho3, n_v3rho2sigma, n_v3rhosigma2, n_v3sigma3;

  void *params;                         /* this allows us to fix parameters in the functional */
  double dens_threshold;

  void *params_gpu;			/* pointer to gpu params in device */
  unsigned long size_of_params_gpu;
};

#ifdef __CUDACC__
struct __align__(128) xc_func_type_cuda_gga {
#else
struct xc_func_type_cuda_gga {
#endif
  int nspin; /* XC_UNPOLARIZED or XC_POLARIZED  */

  int n_rho, n_sigma; /* spin dimensions of the arrays */
  int n_zk;

  int n_vrho, n_vsigma;

  int n_v2rho2, n_v2sigma2, n_v2rhosigma;

  void *params; /* this allows us to fix parameters in the functional */
  double dens_threshold;

  void *params_gpu; /* pointer to gpu params in device */
  unsigned long size_of_params_gpu;
};


typedef struct xc_func_type_cuda xc_func_type_cuda;
typedef struct xc_func_type_cuda_gga xc_func_type_cuda_gga;

/* functionals */
int   xc_functional_get_number(const char *name);
char *xc_functional_get_name(int number);
int   xc_family_from_id_cuda(int id, int *family, int *number);
int   xc_number_of_functionals();
int   xc_maximum_name_length();
void xc_available_functional_numbers(int *list);
void xc_available_functional_names(char **list);

  xc_func_type_cuda *xc_func_alloc_cuda();
int   xc_func_init_cuda(xc_func_type_cuda *p, int functional, int nspin);
void  xc_func_end_cuda(xc_func_type_cuda *p);
void  xc_func_free_cuda(xc_func_type_cuda *p);
const xc_func_info_type_cuda *xc_func_get_info_cuda(const xc_func_type_cuda *p);
void xc_func_set_dens_threshold_cuda(xc_func_type_cuda *p, double dens_threshold);
void  xc_func_set_ext_params_cuda(xc_func_type_cuda *p, double *ext_params);

#include "xc_funcs.h"
#include "xc_funcs_removed.h"

void xc_lda_cuda        (const xc_func_type_cuda *p, int np, const double *rho, double *zk, double *vrho, double *v2rho2, double *v3rho3);
void xc_lda_exc_cuda    (const xc_func_type_cuda *p, int np, const double *rho, double *zk);
void xc_lda_exc_vxc_cuda(const xc_func_type_cuda *p, int np, const double *rho, double *zk, double *vrho);
void xc_lda_vxc_cuda    (const xc_func_type_cuda *p, int np, const double *rho, double *vrho);
void xc_lda_fxc_cuda    (const xc_func_type_cuda *p, int np, const double *rho, double *v2rho2);
void xc_lda_kxc_cuda    (const xc_func_type_cuda *p, int np, const double *rho, double *v3rho3);

/*
void xc_gga     (const xc_func_type_cuda *p, int np, const double *rho, const double *sigma,
		  double *zk, double *vrho, double *vsigma,
		  double *v2rho2, double *v2rhosigma, double *v2sigma2,
		  double *v3rho3, double *v3rho2sigma, double *v3rhosigma2, double *v3sigma3);
*/

void xc_gga_cuda  (const xc_func_type_cuda *p, int np, const double *rho, const double *sigma,
                  double *zk, double *vrho, double *vsigma,
                  double *v2rho2, double *v2rhosigma, double *v2sigma2,
                  double *v3rho3, double *v3rho2sigma, double *v3rhosigma2, double *v3sigma3);

void xc_gga_exc_cuda(const xc_func_type_cuda *p, int np, const double *rho, const double *sigma,
		 double *zk);
void xc_gga_exc_vxc_cuda(const xc_func_type_cuda *p, int np, const double *rho, const double *sigma,
		     double *zk, double *vrho, double *vsigma);
void xc_gga_vxc_cuda(const xc_func_type_cuda *p, int np, const double *rho, const double *sigma,
		 double *vrho, double *vsigma);
void xc_gga_fxc_cuda(const xc_func_type_cuda *p, int np, const double *rho, const double *sigma,
		 double *v2rho2, double *v2rhosigma, double *v2sigma2);
void xc_gga_kxc_cuda(const xc_func_type_cuda *p, int np, const double *rho, const double *sigma,
		 double *v3rho3, double *v3rho2sigma, double *v3rhosigma2, double *v3sigma3);

void xc_gga_lb_modified_cuda  (const xc_func_type_cuda *p, int np, const double *rho, const double *sigma,
			   double r, double *vrho);

double xc_gga_ak13_get_asymptotic (double homo);

double xc_hyb_exx_coef_cuda(const xc_func_type_cuda *p);
void  xc_hyb_cam_coef_cuda(const xc_func_type_cuda *p, double *omega, double *alpha, double *beta);
void  xc_nlc_coef_cuda(const xc_func_type_cuda *p, double *nlc_b, double *nlc_C);

/* the meta-GGAs */
void xc_mgga_cuda        (const xc_func_type_cuda *p, int np,
		      const double *rho, const double *sigma, const double *lapl, const double *tau,
		      double *zk, double *vrho, double *vsigma, double *vlapl, double *vtau,
		      double *v2rho2, double *v2sigma2, double *v2lapl2, double *v2tau2,
		      double *v2rhosigma, double *v2rholapl, double *v2rhotau,
		      double *v2sigmalapl, double *v2sigmatau, double *v2lapltau);
void xc_mgga_exc_cuda    (const xc_func_type_cuda *p, int np,
		      const double *rho, const double *sigma, const double *lapl, const double *tau,
		      double *zk);
void xc_mgga_exc_vxc_cuda(const xc_func_type_cuda *p, int np,
		      const double *rho, const double *sigma, const double *lapl, const double *tau,
		      double *zk, double *vrho, double *vsigma, double *vlapl, double *vtau);
void xc_mgga_vxc_cuda    (const xc_func_type_cuda *p, int np,
		      const double *rho, const double *sigma, const double *lapl, const double *tau,
		      double *vrho, double *vsigma, double *vlapl, double *vtau);
void xc_mgga_fxc_cuda    (const xc_func_type_cuda *p, int np,
		      const double *rho, const double *sigma, const double *lapl, const double *tau,
		      double *v2rho2, double *v2sigma2, double *v2lapl2, double *v2tau2,
		      double *v2rhosigma, double *v2rholapl, double *v2rhotau,
		      double *v2sigmalapl, double *v2sigmatau, double *v2lapltau);

/* Functionals that are defined as mixtures of others */
void xc_mix_func_cuda
  (const xc_func_type_cuda *func, int np,
   const double *rho, const double *sigma, const double *lapl, const double *tau,
   double *zk, double *vrho, double *vsigma, double *vlapl, double *vtau,
   double *v2rho2, double *v2sigma2, double *v2lapl2, double *v2tau2,
   double *v2rhosigma, double *v2rholapl, double *v2rhotau,
   double *v2sigmalapl, double *v2sigmatau, double *v2lapltau);


#ifdef __cplusplus
}
#endif

#endif
