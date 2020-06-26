#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <xc.h>
#include <cuda_runtime.h>
#include "../../../src/helper/helper_cuda.h"
#include "../../../src/util.h"


/**
 * @brief test_libxc_usage_01: Simple libxc test.
 * The following test calculates the xc energy
 * for a given functional for several values of the density
 *
 * The functionals are divided in families (LDA, GGA, etc.).
 * Given a functional identifier xc.func_id,
 * the functional is initialized by xc_func_init,
 * and evaluated by xc_XXX_exc, which returns
 * the energy per unit volume exc.
 * Finally the function xc_func_end cleans up.
 */
void test_libxc_usage_01 (int functional_id)
{
    printf ("Test: test_libxc_usage_01 \n");
    xc_func_type func;
/*
input:
  p: structure obtained from calling xc_func_init
  np: number of points
  rho[]: the density
  sigma[]: contracted gradients of the density
  lapl_rho[]: the laplacian of the density
  tau[]: the kinetic energy density
output:
  exc[]: energy density per unit particle
  vrho[]: first partial derivative of the energy per unit volume in terms of the density
  vsigma[]: first partial derivative of the energy per unit volume in terms of sigma
  vlapl_rho[]: first partial derivative of the energy per unit volume in terms of the laplacian of the density
  vtau[]: first partial derivative of the energy per unit volume in terms of the kinetic energy density
*/

    double rho[5] = {0.1, 0.2, 0.3, 0.4, 0.5};
    double sigma[5] = {0.2, 0.3, 0.4, 0.5, 0.6};
    double lapl_rho[5] = {0.2, 0.3, 0.4, 0.5, 0.6};
    double tau[5] = {0.2, 0.3, 0.4, 0.5, 0.6};
    double exc[5];
    double zk[5];
    double vrho[5];
    double vsigma[5];
    double vlapl_rho[5];
    double vtau[5];

    // Ahora reservamos para los arrays
    unsigned int mem_size_for_cuda_arrays = sizeof(double) * 5;
    double *rhoCUDA = NULL;
    cudaMalloc((void **)&rhoCUDA, mem_size_for_cuda_arrays);
    cudaMemcpy(rhoCUDA, &rho, mem_size_for_cuda_arrays, cudaMemcpyHostToDevice);

    double *sigmaCUDA = NULL;
    cudaMalloc((void **)&sigmaCUDA, mem_size_for_cuda_arrays);
    cudaMemcpy(sigmaCUDA, &sigma, mem_size_for_cuda_arrays, cudaMemcpyHostToDevice);

    double *lapl_rhoCUDA = NULL;
    cudaMalloc((void **)&lapl_rhoCUDA, mem_size_for_cuda_arrays);
    cudaMemcpy(lapl_rhoCUDA, &lapl_rho, mem_size_for_cuda_arrays, cudaMemcpyHostToDevice);

    double *tauCUDA = NULL;
    cudaMalloc((void **)&tauCUDA, mem_size_for_cuda_arrays);
    cudaMemcpy(tauCUDA, &tau, mem_size_for_cuda_arrays, cudaMemcpyHostToDevice);

    // Los siguientes pueden ser opcionales
    // asi que preguntamos si son != de NULL
    double *zkCUDA = NULL;
    if (zk != NULL) {
	cudaMalloc((void **)&zkCUDA, mem_size_for_cuda_arrays);
	cudaMemset(zkCUDA, 0, mem_size_for_cuda_arrays);
    }

    double *excCUDA = NULL;
    if (exc != NULL) {
	cudaMalloc((void **)&excCUDA, mem_size_for_cuda_arrays);
	cudaMemset(excCUDA, 0, mem_size_for_cuda_arrays);
    }

    double *vrhoCUDA = NULL;
    if (vrho != NULL) {
	cudaMalloc((void **)&vrhoCUDA, mem_size_for_cuda_arrays);
	cudaMemset(vrhoCUDA, 0, mem_size_for_cuda_arrays);
    }

    double *vsigmaCUDA = NULL;
    if (vsigma != NULL) {
	cudaMalloc((void **)&vsigmaCUDA, mem_size_for_cuda_arrays);
	cudaMemset(vsigmaCUDA, 0, mem_size_for_cuda_arrays);
    }

    double *vlapl_rhoCUDA = NULL;
    if (vlapl_rho != NULL) {
	cudaMalloc((void **)&vlapl_rhoCUDA, mem_size_for_cuda_arrays);
	cudaMemset(vlapl_rhoCUDA, 0, mem_size_for_cuda_arrays);
    }

    double* vtauCUDA = NULL;
    if (vtau != NULL) {
	cudaMalloc((void **)&vtauCUDA, mem_size_for_cuda_arrays);
	cudaMemset(vtauCUDA, 0, mem_size_for_cuda_arrays);
    }

    int func_id = functional_id;

    /*
     * int xc_func_init(xc_func_type *p, int functional, int nspin);
     * input:
     *  functional: which functional do we want?
     *  nspin: either XC_UNPOLARIZED or XC_POLARIZED
     * output:
     *  p: structure that holds our functional
     * returns: 0 (OK) or -1 (ERROR)
     * ref: https://gitlab.com/libxc/libxc/wikis/manual-4.0
    */

    if(xc_func_init(&func, func_id, XC_UNPOLARIZED) != 0){
        fprintf(stderr, "Functional '%d' not found\n", func_id);
        return;
    }

    printf("The functional '%s' is ", func.info->name);
    switch (func.info->kind) {
      case (XC_EXCHANGE):
	printf("an exchange functional");
        break;
      case (XC_CORRELATION):
        printf("a correlation functional");
        break;
      case (XC_EXCHANGE_CORRELATION):
	printf("an exchange-correlation functional");
        break;
      case (XC_KINETIC):
	printf("a kinetic energy functional");
        break;
      default:
        printf("of unknown kind");
	break;
    }
    
    printf(", it belongs to the '%s'", func.info->name);
    switch (func.info->family) {
	case (XC_FAMILY_LDA):
	    printf("LDA");
        break;
	case (XC_FAMILY_GGA):
	    printf("GGA");
        break;
        case (XC_FAMILY_HYB_GGA):
	    printf("Hybrid GGA");
        break;
	case (XC_FAMILY_MGGA):
	    printf("MGGA");
        break;
	case (XC_FAMILY_HYB_MGGA):
    	    printf("Hybrid MGGA");
	break;
	default:
	    printf("unknown");
	break;
    }
    printf("' family and is defined in the reference(s):\n");

    for(int ii = 0; func.info->refs[ii] != NULL; ii++){
	printf("[%d] %s\n", ii+1, func.info->refs[ii]->ref);
    }

    switch(func.info->family)
    {
    case XC_FAMILY_LDA:
        // Set extra parameters.
	//xc_func_set_ext_params(&func, ext_params);
	//xc_lda(&func, 5, rhoCUDA, zkCUDA, vrhoCUDA, v2rho2CUDA, v3rho3CUDA);
        break;
    case XC_FAMILY_GGA:
    case XC_FAMILY_HYB_GGA:
	//xc_gga(&func, 5, rhoCUDA, sigmaCUDA, excCUDA, vrhoCUDA, vsigmaCUDA, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
        break;
    case XC_FAMILY_MGGA:
	xc_mgga(&func, 5, rhoCUDA, sigmaCUDA, lapl_rhoCUDA,
	    tauCUDA, zkCUDA, vrhoCUDA, vsigmaCUDA, vlapl_rhoCUDA, vtauCUDA,
	    NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);

	break;
    }

    // Copiamos de vuelta lo que dejo en los arrays.
    //cudaMemcpy(&exc, excCUDA, mem_size_for_cuda_arrays, cudaMemcpyDeviceToHost);
    cudaMemcpy(&rho, rhoCUDA, mem_size_for_cuda_arrays, cudaMemcpyDeviceToHost);
    cudaMemcpy(&vrho, vrhoCUDA, mem_size_for_cuda_arrays, cudaMemcpyDeviceToHost);
    cudaMemcpy(&vsigma, vsigmaCUDA, mem_size_for_cuda_arrays, cudaMemcpyDeviceToHost);
    cudaMemcpy(&zk, zkCUDA, mem_size_for_cuda_arrays, cudaMemcpyDeviceToHost);
    cudaMemcpy(&vlapl_rho, vlapl_rhoCUDA, mem_size_for_cuda_arrays, cudaMemcpyDeviceToHost);
    cudaMemcpy(&vtau, vtauCUDA, mem_size_for_cuda_arrays, cudaMemcpyDeviceToHost);

    for(int i=0; i<5; i+=1){
        printf("%lf %lf %lf %lf %lf \n", rho[i], zk[i], vrho[i], vsigma[i], vlapl_rho[i]);
    }

    xc_func_end(&func);

    // Free cuda memory
    cudaFree(rhoCUDA);
    cudaFree(sigmaCUDA);
    cudaFree(lapl_rhoCUDA);
    cudaFree(tauCUDA);
    if (excCUDA != NULL) {
        cudaFree(excCUDA);
    }
    if (zkCUDA != NULL) {
	cudaFree(zkCUDA);
    }
    if (vrho != NULL) {
	cudaFree(vrhoCUDA);
    }
    if (vsigma != NULL) {
	cudaFree(vsigmaCUDA);
    }
    if (vlapl_rho != NULL) {
	cudaFree(vlapl_rhoCUDA);
    }
    if (vtau != NULL) {
	cudaFree(vtauCUDA);
    }

    /*
    The output of the above example should
    look something like this:

    Libxc version: 4.0.0
    0.100000 -0.342809
    0.200000 -0.431912
    0.300000 -0.494416
    0.400000 -0.544175
    0.500000 -0.586194
    */
}


int main(void)
{
    printf ("Test Libxc GPU - MGGA Exchange - BEGIN \n");
    try {
        int functionals[] = {1202,1212,1244,1245};
	int functionalCount = 4;
        for (int i=0; i<functionalCount; i++) {
	    test_libxc_usage_01(functionals[i]);
        }
    } catch (int e) {
        printf("An exception occurred: %u \n", e);
        exit (EXIT_FAILURE);
    }
    printf("Test Libxc GPU - MGGA Exchange - END\n");

    return 0;
}

