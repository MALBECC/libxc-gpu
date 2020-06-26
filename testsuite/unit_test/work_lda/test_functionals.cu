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
    double rho[5] = {0.1, 0.2, 0.3, 0.4, 0.5};
    double sigma[5] = {0.2, 0.3, 0.4, 0.5, 0.6};
    double vrho [5];
    double v2rho2 [5];
    double v3rho3 [5];
    double vsigma [5];
    double exc[5];
    double zk[5];

    // Ahora reservamos para los arrays
    unsigned int mem_size_for_cuda_arrays = sizeof(double) * 5;

    double *rhoCUDA = NULL;
    cudaMalloc((void **)&rhoCUDA, mem_size_for_cuda_arrays);
    cudaMemcpy(rhoCUDA, &rho, mem_size_for_cuda_arrays, cudaMemcpyHostToDevice);

    double *sigmaCUDA = NULL;
    cudaMalloc((void **)&sigmaCUDA, mem_size_for_cuda_arrays);
    cudaMemcpy(sigmaCUDA, &sigma, mem_size_for_cuda_arrays, cudaMemcpyHostToDevice);

    // Los siguientes pueden ser opcionales
    // asi que preguntamos si son != de NULL
    double *zkCUDA = NULL;
    if (zk != NULL) {
	cudaMalloc((void **)&zkCUDA, mem_size_for_cuda_arrays);
	//cudaMemcpy(zkCUDA, zk, mem_size_for_cuda_arrays, cudaMemcpyHostToDevice);
	cudaMemset(zkCUDA, 0, mem_size_for_cuda_arrays);
    }

    double *excCUDA = NULL;
    if (exc != NULL) {
	cudaMalloc((void **)&excCUDA, mem_size_for_cuda_arrays);
	//cudaMemcpy(excCUDA, exc, mem_size_for_cuda_arrays, cudaMemcpyHostToDevice);
	cudaMemset(excCUDA, 0, mem_size_for_cuda_arrays);
    }

    double *vrhoCUDA = NULL;
    if (vrho != NULL) {
	cudaMalloc((void **)&vrhoCUDA, mem_size_for_cuda_arrays);
	//cudaMemcpy(vrhoCUDA, vrho, mem_size_for_cuda_arrays, cudaMemcpyHostToDevice);
	cudaMemset(vrhoCUDA, 0, mem_size_for_cuda_arrays);
    }

    double *v2rho2CUDA = NULL;
    if (v2rho2 != NULL) {
	cudaMalloc((void **)&v2rho2CUDA, mem_size_for_cuda_arrays);
	//cudaMemcpy(v2rho2CUDA, v2rho2, mem_size_for_cuda_arrays, cudaMemcpyHostToDevice);
	cudaMemset(v2rho2CUDA, 0, mem_size_for_cuda_arrays);
    }

    double *v3rho3CUDA = NULL;
    if (v3rho3 != NULL) {
	cudaMalloc((void **)&v3rho3CUDA, mem_size_for_cuda_arrays);
	//cudaMemcpy(v3rho3CUDA, v3rho3, mem_size_for_cuda_arrays, cudaMemcpyHostToDevice);
	cudaMemset(v3rho3CUDA, 0, mem_size_for_cuda_arrays);
    }

    double *vsigmaCUDA = NULL;
    if (vsigma != NULL) {
	cudaMalloc((void **)&vsigmaCUDA, mem_size_for_cuda_arrays);
	//cudaMemcpy(vsigmaCUDA, vsigma, mem_size_for_cuda_arrays, cudaMemcpyHostToDevice);
	cudaMemset(vsigmaCUDA, 0, mem_size_for_cuda_arrays);
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

    double ext_params[2];

    switch (func_id) {
    case 1016:
	ext_params[0]=2;
	ext_params[1]=2;
	break;
    case 1018:
	ext_params[0]=1;
	ext_params[1]=1;
	break;
    }

    switch(func.info->family)
    {
    case XC_FAMILY_LDA:
        // Set extra parameters.
	xc_func_set_ext_params(&func, ext_params);
	xc_lda(&func, 5, rhoCUDA, zkCUDA, vrhoCUDA, v2rho2CUDA, v3rho3CUDA);
        break;
    case XC_FAMILY_GGA:
    case XC_FAMILY_HYB_GGA:
	//xc_gga(&func, 5, rhoCUDA, sigmaCUDA, excCUDA, vrhoCUDA, vsigmaCUDA, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
        break;
    }

    // Copiamos de vuelta lo que dejo en los arrays.
    cudaMemcpy(&exc, excCUDA, mem_size_for_cuda_arrays, cudaMemcpyDeviceToHost);
    cudaMemcpy(&zk, zkCUDA, mem_size_for_cuda_arrays, cudaMemcpyDeviceToHost);
    cudaMemcpy(&rho, rhoCUDA, mem_size_for_cuda_arrays, cudaMemcpyDeviceToHost);
    cudaMemcpy(&vrho, vrhoCUDA, mem_size_for_cuda_arrays, cudaMemcpyDeviceToHost);
    cudaMemcpy(&v2rho2, v2rho2CUDA, mem_size_for_cuda_arrays, cudaMemcpyDeviceToHost);
    cudaMemcpy(&v3rho3, v3rho3CUDA, mem_size_for_cuda_arrays, cudaMemcpyDeviceToHost);
    cudaMemcpy(&vsigma, vsigmaCUDA, mem_size_for_cuda_arrays, cudaMemcpyDeviceToHost);

    for(int i=0; i<5; i+=1){
        printf("%lf %lf %lf %lf %lf \n", rho[i], zk[i], vrho[i], v2rho2[i], v3rho3[i]);
    }

    xc_func_end(&func);

    // Free cuda memory
    cudaFree(rhoCUDA);
    cudaFree(sigmaCUDA);
    if (excCUDA != NULL) {
        cudaFree(excCUDA);
    }
    if (zkCUDA != NULL) {
	cudaFree(zkCUDA);
    }
    if (vrho != NULL) {
	cudaFree(vrhoCUDA);
    }
    if (v2rho2 != NULL) {
	cudaFree(v2rho2CUDA);
    }
    if (v3rho3 != NULL) {
	cudaFree(v3rho3CUDA);
    }
    if (vsigma != NULL) {
	cudaFree(vsigmaCUDA);
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
    printf ("Test Libxc GPU - LDA - BEGIN \n");

    try {
        int functionals[] = {1016,1018}; // Add more functionals id's here to test them.
	int functionalCount = 2; // update this count if you add more functionals.
        for (int i=0; i<functionalCount; i++) {
	    test_libxc_usage_01(functionals[i]);
        }
    } catch (int e) {
        printf("An exception occurred: %u \n", e);
        exit (EXIT_FAILURE);
    }
    printf("Test Libxc GPU - LDA - END\n");

    return 0;
}

