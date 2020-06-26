#ifndef _WORK_UTILS_H
#define _WORK_UTILS_H

#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

//////////////////////////////////////
//// PRINT UTILS
__device__ inline void print_xc_gga_work_c_t_2 (xc_gga_work_c_t_2* r)
{
    printf ("xc_gga_work_c_t_2[ \n");
    if (r == NULL) {
	printf("empty");
    } else {

	printf("order: %e\n", r->order);
	printf("dens: %e \n", r->dens);
	printf("ds: %e %e \n", r->ds[0], r->ds[1]);
	printf("sigmat: %e %e \n", r->sigmat);

	printf("sigmas: %e %e %e \n", r->sigmas[0], r->sigmas[1], r->sigmas[2]);
	printf("rs: %e \n", r->rs);
	printf("z: %e \n", r->z);

	printf("xt: %e \n", r->xt);
	printf("xs: %e %e \n", r->xs[0], r->xs[1]);
	printf("f: %e \n", r->f);

	printf("dfdrs: %e \n", r->dfdrs);
	printf("dfdz: %e \n", r->dfdz);
	printf("dfdxt: %e \n", r->dfdxt);


	printf("dfdxs: %e %e \n", r->dfdxs[0], r->dfdxs[1]);
	printf("d2fdrs2: %e \n", r->d2fdrs2);
	printf("d2fdrsz: %e \n", r->d2fdrsz);

	printf("d2fdrsxt: %e \n", r->d2fdrsxt);
	printf("d2fdrsxs: %e %e \n", r->d2fdrsxs[0], r->d2fdrsxs[1]);
	printf("d2fdz2: %e \n", r->d2fdz2);

	printf("d2fdzxt: %e \n", r->d2fdzxt);
	printf("d2fdzxs: %e %e \n", r->d2fdzxs[0], r->d2fdzxs[1]);
	printf("d2fdxt2: %e \n", r->d2fdxt2);

	printf("d2fdxtxs: %e %e \n", r->d2fdxtxs[0], r->d2fdxtxs[1]);
	printf("d2fdxs2: %e %e %e \n", r->d2fdxs2[0], r->d2fdxs2[1], r->d2fdxs2[2]);

    }
    printf("]\n");

}

__device__ inline void print_xc_func_type_cuda (const xc_func_type_cuda* p)
{
    printf ("n_rho:%e n_sigma:%e n_zk:%e n_vrho:%e n_vsigma:%e n_v2rho2:%e n_v2rhosigma:%e n_v2sigma2:%e \n",
	p->n_rho, p->n_sigma, p->n_zk, p->n_vrho, p->n_vsigma, p->n_v2rho2, p->n_v2rhosigma, p->n_v2sigma2);

}

__device__ inline void print_xc_gga_work_c_t_2_line (xc_gga_work_c_t_2* r)
{
    //printf ("xc_gga_work_c_t_2[");
    if (r == NULL) {
	printf("empty");
    } else {
	printf ("order:%e \
	    dens:%e \
	    ds_0:%e ds_1: %e \
	    sigmat:%e \
	    sigmas_0:%e sigmas_1: %e sigmas_2:%e \
	    rs:%e \
	    z:%e \
	    xt:%e \
	    xs_0:%e xs_1: %e \
	    f:%e \
	    dfdrs:%e \
	    dfdz:%e \
	    dfdxt:%e \
	    dfdxs_0:%e dfdxs_1:%e \
	    d2fdrs2:%e \
	    d2fdrsz:%e \
	    d2fdrsxt:%e \
	    d2fdrsxs_0:%e d2fdrsxs_1:%e \
	    d2fdz2:%e \
	    d2fdzxt:%e \
	    d2fdzxs_0:%e d2fdzxs_1:%e \
	    d2fdxt2:%e \
	    d2fdxtxs_0:%e d2fdxtxs_1:%e \
	    d2fdxs2_0:%e d2fdxs2_1:%e d2fdxs2_2:%e \n",
	    r->order, 
	    r->dens,
	    r->ds[0], r->ds[1],
	    r->sigmat, 
	    r->sigmas[0], r->sigmas[1], r->sigmas[2], 
	    r->rs, 
	    r->z, 
	    r->xt, 
	    r->xs[0], r->xs[1], 
	    r->f, 
	    r->dfdrs, 
	    r->dfdz, 
	    r->dfdxt, 
	    r->dfdxs[0], r->dfdxs[1], 
	    r->d2fdrs2, 
	    r->d2fdrsz, 
	    r->d2fdrsxt, 
	    r->d2fdrsxs[0], r->d2fdrsxs[1], 
	    r->d2fdz2, 
	    r->d2fdzxt, 
	    r->d2fdzxs[0], r->d2fdzxs[1], 
	    r->d2fdxt2, 
	    r->d2fdxtxs[0], r->d2fdxtxs[1], 
	    r->d2fdxs2[0], r->d2fdxs2[1], r->d2fdxs2[2]);

    }
    //printf("]\n");

}



static void print_array (double* data, int size) 
{
    printf ("[");
    if (data == NULL) {
	printf("empty");
    } else {
	for (int i=0; i<size; i++) {
	    printf("%e,", data[i]);
	}
    }
    printf("]\n");
}

static void copy_to_host_and_print (int np, const double *rho, const double *sigma,
	double *zk, double *vrho, double *vsigma,
	double *v2rho2, double *v2rhosigma, double *v2sigma2)
{
    // Copy the data to the CPU before print
    // Alloc memory in the host for the gpu data
    int size = np * sizeof(double);

    double* rho_cpu = (double*)malloc(size);
    double* sigma_cpu = (double*)malloc(size);
    double* zk_cpu = (double*)malloc(size);
    double* vrho_cpu = (double*)malloc(size);
    double* vsigma_cpu = (double*)malloc(size);
    double* v2rho2_cpu = (double*)malloc(size);
    double* v2rhosigma_cpu = (double*)malloc(size);
    double* v2sigma2_cpu = (double*)malloc(size);

    // Copy data from device to host.
    if (rho != NULL)
        cudaMemcpy(rho_cpu, rho, size, cudaMemcpyDeviceToHost);
    if (sigma != NULL)
	cudaMemcpy(sigma_cpu, sigma, size, cudaMemcpyDeviceToHost);
    if (zk != NULL)
        cudaMemcpy(zk_cpu, zk, size, cudaMemcpyDeviceToHost);
    if (vrho != NULL)
        cudaMemcpy(vrho_cpu, vrho, size, cudaMemcpyDeviceToHost);
    if (vsigma != NULL)
        cudaMemcpy(vsigma_cpu, vsigma, size, cudaMemcpyDeviceToHost);
    if (v2rho2 != NULL)
        cudaMemcpy(v2rho2_cpu, v2rho2, size, cudaMemcpyDeviceToHost);
    if (v2rhosigma != NULL)
        cudaMemcpy(v2rhosigma_cpu, v2rhosigma, size, cudaMemcpyDeviceToHost);
    if (v2sigma2 != NULL)
        cudaMemcpy(v2sigma2_cpu, v2sigma2, size, cudaMemcpyDeviceToHost);

    printf("number_of_points:%i\n", np);
    printf("rho:"); print_array(rho_cpu, np);
    printf("sigma:"); print_array(sigma_cpu, np);
    printf("zk:"); print_array(zk_cpu, np);
    printf("vrho:"); print_array(vrho_cpu, np);
    printf("vsigma:"); print_array(vsigma_cpu, np);
    printf("v2rho2:"); print_array(v2rho2_cpu, np);
    printf("v2rhosigma:"); print_array(v2rhosigma_cpu, np);
    printf("v2sigma2:"); print_array(v2sigma2_cpu, np);
    printf("====================\n");

    // Free memory.
    free(rho_cpu);
    free(sigma_cpu);
    free(zk_cpu);
    free(vrho_cpu);
    free(vsigma_cpu);
    free(v2rho2_cpu);
    free(v2rhosigma_cpu);
    free(v2sigma2_cpu);

}

static void print_work_gga_x_input (int np, const double *rho, const double *sigma,
	double *zk, double *vrho, double *vsigma,
	double *v2rho2, double *v2rhosigma, double *v2sigma2,
	double *v3rho3, double *v3rho2sigma, double *v3rhosigma2, double *v3sigma3)
{
    printf("====================\n");
    printf("= work_gga X input =\n");
    printf("====================\n");

    copy_to_host_and_print(np, rho, sigma, zk, vrho, vsigma, v2rho2, v2rhosigma, v2sigma2);
}

static void print_work_gga_c_input (int np, const double *rho, const double *sigma,
	double *zk, double *vrho, double *vsigma,
	double *v2rho2, double *v2rhosigma, double *v2sigma2,
	double *v3rho3, double *v3rho2sigma, double *v3rhosigma2, double *v3sigma3)
{
    printf("====================\n");
    printf("= work_gga C input =\n");
    printf("====================\n");

    copy_to_host_and_print(np, rho, sigma, zk, vrho, vsigma, v2rho2, v2rhosigma, v2sigma2);
}

#endif
