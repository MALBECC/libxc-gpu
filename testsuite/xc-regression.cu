/*
 Copyright (C) 2006-2007 M.A.L. Marques
 Copyright (C) 2014 Susi Lehtola

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <xc.h>
#include <cuda_runtime.h>

/* Buffer size (line length) for file reads */
#define BUFSIZE 1024

void print_xc_func_info_type_cuda (const xc_func_info_type_cuda* infoType) {
    printf ("xc_func_info_type_cuda {");
    if (infoType == NULL) {
	printf ("NULL");
    } else {
	printf ("number: %i,", infoType->number);
	printf ("kind: %i,", infoType->kind);
	printf ("name: %s,", infoType->name);
	printf ("family: %i,", infoType->family);
	//printf ("func_references_types: %i,", infoType->refs); // Es un puntero a otro struct.
	printf ("flags: %i,", infoType->flags);
	printf ("dens_threshold: %f,", infoType->dens_threshold);
	printf ("n_ext_params: %i,", infoType->n_ext_params);
	//printf ("func_params_type: %i,", infoType->func_params_type); // Otro struct (este es const)
	//printf ("set_ext_params: %p,", infoType->set_ext_params); // puntero a funcion
	printf ("init: %p,", infoType->init);
	printf ("end: %p,", infoType->end);
	printf ("lda: %p,", infoType->lda);
	printf ("gga: %p,", infoType->gga);
	printf ("mgga: %p,", infoType->mgga);
    }
    printf ("}\n");
}

/*
void print_xc_func_type_pbe_params (gga_c_pbe_params* params) 
{
    printf ("xc_func_type_pbe_params {\n");
    if (params != NULL) {
	printf ("beta:%d,",params->beta);
        printf ("gamma:%d,",params->gamma);
        printf ("BB:%d,",params->BB);
    } else {
	printf ("NULL");
    }
    printf ("}\n");
}
*/

void print_xc_func_type (xc_func_type_cuda* p) {
    printf ("xc_func_type_cuda[ \n");
    if (p == NULL) {
	printf("{}");
    } else {
	printf ("nspin: %i,", p->nspin);
	//print_xc_func_type_pbe_params (p.params);
        printf ("params:%p,",p->params);
        //printf ("paramsGPU:%p,",p->paramsGPU);
	//p.info = &infoType;
	print_xc_func_info_type_cuda (p->info);
        printf ("n_lapl:%i,",p->n_lapl);
        printf ("n_rho:%i,",p->n_rho);
        printf ("sigma:%i,",p->n_sigma);
        printf ("n_tau:%i,",p->n_tau);
        printf ("n_v2lapl2:%i,",p->n_v2lapl2);
        printf ("n_v2lapltau:%i,",p->n_v2lapltau);
        printf ("n_v2rho2:%i,",p->n_v2rho2);
        printf ("n_v2rholapl:%i,",p->n_v2rholapl);
        printf ("n_v2rhosigma:%i,",p->n_v2rhosigma);
        printf ("n_v2rhotau:%i,",p->n_v2rhotau);
        printf ("n_v2sigma2:%i,",p->n_v2sigma2);
        printf ("n_v2sigmalapl:%i,",p->n_v2sigmalapl);
        printf ("n_v2sigmatau:%i,",p->n_v2sigmatau);
        printf ("n_v2tau2:%i,",p->n_v2tau2);
        printf ("n_v3rho2sigma:%i,",p->n_v3rho2sigma);
        printf ("n_v3rho3:%i,",p->n_v3rho3);
        printf ("n_v3rhosigma2:%i,",p->n_v3rhosigma2);
        printf ("n_v3sigma3:%i,",p->n_v3sigma3);
        printf ("n_vlapl:%i,",p->n_vlapl);
        printf ("n_vrho:%i,",p->n_vrho);
        printf ("n_vsigma:%i,",p->n_vsigma);
        printf ("n_vtau:%i,",p->n_vtau);
	printf ("n_zk:%i,",p->n_zk);
    }
    printf ("]\n");
}

void print_param_list (double* list, int np)
{
    printf ("[");
    if (np <= 0 || list == NULL) {
	printf("NULL");
    } else {
	for (int i=0; i<np; i++) {
	    printf ("%f,", list[i]);
	}
    }
    printf ("]");
    printf ("\n");
}

void print_work_gga_c_params(xc_func_type_cuda* p, int np, 
    double* rho, double* sigma, double* zk,
    double* vrho, double* vsigma,
    double* v2rho2, double* v2rhosigma, double* v2sigma2,
    double* v3rho3, double* v3rho2sigma, double* v3rhosigma2, double* v3sigma3) 
{
    printf("** ******************  ** \n");
    printf("** work_gga_c_params() ** \n");
    print_xc_func_type (p);
    printf((char*)"rho: \t"); print_param_list (rho, np);
    printf((char*)"sigma: \t"); print_param_list (sigma, np);
    printf((char*)"zk: \t"); print_param_list (zk, np);
    printf((char*)"vrho: \t"); print_param_list (vrho, np);
    printf((char*)"vsigma: \t"); print_param_list (vsigma, np);
    printf((char*)"v2rho2: \t"); print_param_list (v2rho2, np);
    printf((char*)"v2rhosigma: \t"); print_param_list (v2rhosigma, np);
    printf((char*)"v2sigma2: \t"); print_param_list (v2sigma2, np);
    printf((char*)"v3rho3: \t"); print_param_list (v3rho3, np);
    printf((char*)"v3rho2sigma: \t"); print_param_list (v3rho2sigma, np);
    printf((char*)"v3rhosigma2: \t"); print_param_list (v3rhosigma2, np);
    printf((char*)"v3sigma3: \t"); print_param_list (v3sigma3, np);
    printf("** ******************  ** \n");
}

typedef struct {
  /* Amount of data points */
  int n;

  /* Input: density, gradient, laplacian and kinetic energy density */
  double *rho;
  double *sigma;
  double *lapl;
  double *tau;

  /* Output: energy density */
  double *zk;

  /* .. and potentials for density, gradient, laplacian and tau */
  double *vrho;
  double *vsigma;
  double *vlapl;
  double *vtau;

  /* ... and second derivatives */
  double *v2rho2;
  double *v2tau2;
  double *v2lapl2;
  double *v2rhotau;
  double *v2rholapl;
  double *v2lapltau;
  double *v2sigma2;
  double *v2rhosigma;
  double *v2sigmatau;
  double *v2sigmalapl;

  /* ... and third derivatives */
  double *v3rho3;
  double *v3rho2sigma;
  double *v3rhosigma2;
  double *v3sigma3;


  // The same things but now for CUDA
  /* Input: density, gradient, laplacian and kinetic energy density */
  double *rhoCUDA;
  double *sigmaCUDA;
  double *laplCUDA;
  double *tauCUDA;

  /* Output: energy density */
  double *zkCUDA;

  /* .. and potentials for density, gradient, laplacian and tau */
  double *vrhoCUDA;
  double *vsigmaCUDA;
  double *vlaplCUDA;
  double *vtauCUDA;

  /* ... and second derivatives */
  double *v2rho2CUDA;
  double *v2tau2CUDA;
  double *v2lapl2CUDA;
  double *v2rhotauCUDA;
  double *v2rholaplCUDA;
  double *v2lapltauCUDA;
  double *v2sigma2CUDA;
  double *v2rhosigmaCUDA;
  double *v2sigmatauCUDA;
  double *v2sigmalaplCUDA;

  /* ... and third derivatives */
  double *v3rho3CUDA;
  double *v3rho2sigmaCUDA;
  double *v3rhosigma2CUDA;
  double *v3sigma3CUDA;


} values_t;

void copy_arrays_to_cuda (values_t *data)
{
	// Ahora reservamos para los arrays
	unsigned int mem_size_for_cuda_arrays = sizeof(double) * data->n;

	//double *rhoCUDA = NULL;
	cudaMalloc((void **)&(data->rhoCUDA), mem_size_for_cuda_arrays);
	cudaMemcpy(data->rhoCUDA, data->rho, mem_size_for_cuda_arrays, cudaMemcpyHostToDevice);

	//double *sigmaCUDA = NULL;
	cudaMalloc((void **)&(data->sigmaCUDA), mem_size_for_cuda_arrays);
	cudaMemcpy(data->sigmaCUDA, data->sigma, mem_size_for_cuda_arrays, cudaMemcpyHostToDevice);

	// Los siguientes pueden ser opcionales
	// asi que preguntamos si son != de NULL
	//double *zkCUDA = NULL;
	if (data->zk != NULL) {
		cudaMalloc((void **)&(data->zkCUDA), mem_size_for_cuda_arrays);
		cudaMemcpy(data->zkCUDA, data->zk, mem_size_for_cuda_arrays, cudaMemcpyHostToDevice);
	}

	//double *vrhoCUDA = NULL;
	if (data->vrho != NULL) {
		cudaMalloc((void **)&(data->vrhoCUDA), mem_size_for_cuda_arrays);
		cudaMemcpy(data->vrhoCUDA, data->vrho, mem_size_for_cuda_arrays, cudaMemcpyHostToDevice);
	}
	
	//double *vsigmaCUDA = NULL;
	if (data->vsigma != NULL) {
		cudaMalloc((void **)&(data->vsigmaCUDA), mem_size_for_cuda_arrays);
		cudaMemcpy(data->vsigmaCUDA, data->vsigma, mem_size_for_cuda_arrays, cudaMemcpyHostToDevice);
	}

	//double *v2rho2CUDA = NULL;
	if (data->v2rho2 != NULL) {
		cudaMalloc((void **)&(data->v2rho2CUDA), mem_size_for_cuda_arrays);
		cudaMemcpy(data->v2rho2CUDA, data->v2rho2, mem_size_for_cuda_arrays, cudaMemcpyHostToDevice);
	}

	//double *v2rhosigmaCUDA = NULL;
	if (data->v2rhosigma != NULL) {
		cudaMalloc((void **)&(data->v2rhosigmaCUDA), mem_size_for_cuda_arrays);
		cudaMemcpy(data->v2rhosigmaCUDA, data->v2rhosigma, mem_size_for_cuda_arrays, cudaMemcpyHostToDevice);
	}

	//double *v2sigma2CUDA = NULL;
	if (data->v2sigma2 != NULL) {
		cudaMalloc((void **)&(data->v2sigma2CUDA), mem_size_for_cuda_arrays);
		cudaMemcpy(data->v2sigma2CUDA, data->v2rhosigma, mem_size_for_cuda_arrays, cudaMemcpyHostToDevice);
	}

	//double *v3rho3CUDA = NULL;
	if (data->v3rho3 != NULL) {
		cudaMalloc((void **)&(data->v3rho3CUDA), mem_size_for_cuda_arrays);
		cudaMemcpy(data->v3rho3CUDA, data->v3rho3, mem_size_for_cuda_arrays, cudaMemcpyHostToDevice);
	}

	//double *v3rho2sigmaCUDA = NULL;
	if (data->v3rho2sigma != NULL) {
		cudaMalloc((void **)&(data->v3rho2sigmaCUDA), mem_size_for_cuda_arrays);
		cudaMemcpy(data->v3rho2sigmaCUDA, data->v3rho2sigma, mem_size_for_cuda_arrays, cudaMemcpyHostToDevice);
	}

	//double *v3rhosigma2CUDA = NULL;
	if (data->v3rhosigma2 != NULL) {
		cudaMalloc((void **)&(data->v3rhosigma2CUDA), mem_size_for_cuda_arrays);
		cudaMemcpy(data->v3rhosigma2CUDA, data->v3rhosigma2, mem_size_for_cuda_arrays, cudaMemcpyHostToDevice);
	}

	//double *v3sigma3CUDA;
	if (data->v3sigma3 != NULL) {
		cudaMalloc((void **)&(data->v3sigma3CUDA), mem_size_for_cuda_arrays);
		cudaMemcpy(data->v3sigma3CUDA, data->v3sigma3, mem_size_for_cuda_arrays, cudaMemcpyHostToDevice);
	}
}


void allocate_memory(values_t *data, int nspin, int order)
{
  data->zk = NULL;
  data->vrho = NULL;
  data->vsigma = NULL;
  data->vlapl = NULL;
  data->vtau = NULL;
  data->v2rho2 = NULL;
  data->v2tau2 = NULL;
  data->v2lapl2 = NULL;
  data->v2rhotau = NULL;
  data->v2rholapl = NULL;
  data->v2lapltau = NULL;
  data->v2sigma2 = NULL;
  data->v2rhosigma = NULL;
  data->v2sigmatau = NULL;
  data->v2sigmalapl = NULL;
  data->v3rho3 = NULL;
  data->v3rho2sigma = NULL;
  data->v3rhosigma2 = NULL;
  data->v3sigma3 = NULL;

  // The same as above but for CUDA
  data->zkCUDA = NULL;
  data->vrhoCUDA = NULL;
  data->vsigmaCUDA = NULL;
  data->vlaplCUDA = NULL;
  data->vtauCUDA = NULL;
  data->v2rho2CUDA = NULL;
  data->v2tau2CUDA = NULL;
  data->v2lapl2CUDA = NULL;
  data->v2rhotauCUDA = NULL;
  data->v2rholaplCUDA = NULL;
  data->v2lapltauCUDA = NULL;
  data->v2sigma2CUDA = NULL;
  data->v2rhosigmaCUDA = NULL;
  data->v2sigmatauCUDA = NULL;
  data->v2sigmalaplCUDA = NULL;
  data->v3rho3CUDA = NULL;
  data->v3rho2sigmaCUDA = NULL;
  data->v3rhosigma2CUDA = NULL;
  data->v3sigma3CUDA = NULL;

  switch(nspin) {
    case (XC_UNPOLARIZED):
      data->rho = (double*)calloc(data->n, sizeof(double));
      data->sigma = (double*)calloc(data->n, sizeof(double));
      data->lapl = (double*)calloc(data->n, sizeof(double));
      data->tau = (double*)calloc(data->n, sizeof(double));
      switch (order) {
        case (0):
          data->zk = (double*)calloc(data->n, sizeof(double));
          break;
        case (1):
          data->vrho = (double*)calloc(data->n, sizeof(double));
          data->vsigma = (double*)calloc(data->n, sizeof(double));
          data->vlapl = (double*)calloc(data->n, sizeof(double));
          data->vtau = (double*)calloc(data->n, sizeof(double));
          break;
        case (2):
          data->v2rho2 = (double*)calloc(data->n, sizeof(double));
          data->v2tau2 = (double*)calloc(data->n, sizeof(double));
          data->v2lapl2 = (double*)calloc(data->n, sizeof(double));
          data->v2rhotau = (double*)calloc(data->n, sizeof(double));
          data->v2rholapl =(double*) calloc(data->n, sizeof(double));
          data->v2lapltau = (double*)calloc(data->n, sizeof(double));
          data->v2sigma2 = (double*)calloc(data->n, sizeof(double));
          data->v2rhosigma = (double*)calloc(data->n, sizeof(double));
          data->v2sigmatau = (double*)calloc(data->n, sizeof(double));
          data->v2sigmalapl = (double*)calloc(data->n, sizeof(double));
          break;
        case (3):
          data->v3rho3 = (double*)calloc(data->n, sizeof(double));
          data->v3rho2sigma = (double*)calloc(data->n, sizeof(double));
          data->v3rhosigma2 = (double*)calloc(data->n, sizeof(double));
          data->v3sigma3 = (double*)calloc(data->n, sizeof(double));
          break;
        default:
          fprintf(stderr, "order = %i not recognized.\n", order);
          exit(2);
      }
      break;

    case (XC_POLARIZED):
      data->rho = (double*)calloc(2*data->n, sizeof(double));
      data->sigma = (double*)calloc(3*data->n, sizeof(double));
      data->lapl = (double*)calloc(2*data->n, sizeof(double));
      data->tau = (double*)calloc(2*data->n, sizeof(double));
      switch (order) {
        case (0):
          data->zk = (double*)calloc(data->n, sizeof(double));
          break;
        case (1):
          data->vrho = (double*)calloc(2*data->n, sizeof(double));
          data->vsigma = (double*)calloc(3*data->n, sizeof(double));
          data->vlapl = (double*)calloc(2*data->n, sizeof(double));
          data->vtau = (double*)calloc(2*data->n, sizeof(double));
          break;
        case (2):
          data->v2rho2 = (double*)calloc(3*data->n, sizeof(double));
          data->v2tau2 = (double*)calloc(3*data->n, sizeof(double));
          data->v2lapl2 = (double*)calloc(3*data->n, sizeof(double));
          data->v2rhotau = (double*)calloc(4*data->n, sizeof(double));
          data->v2rholapl = (double*)calloc(4*data->n, sizeof(double));
          data->v2lapltau = (double*)calloc(4*data->n, sizeof(double));
          data->v2sigma2 = (double*)calloc(6*data->n, sizeof(double));
          data->v2rhosigma = (double*)calloc(6*data->n, sizeof(double));
          data->v2sigmatau = (double*)calloc(6*data->n, sizeof(double));
          data->v2sigmalapl = (double*)calloc(6*data->n, sizeof(double));
          break;
        case (3):
          data->v3rho3 = (double*)calloc(4*data->n, sizeof(double));
          data->v3rho2sigma = (double*)calloc(6*data->n, sizeof(double));
          data->v3rhosigma2 = (double*)calloc(6*data->n, sizeof(double));
          data->v3sigma3 = (double*)calloc(6*data->n, sizeof(double));
          break;
        default:
          fprintf(stderr, "order = %i not recognized.\n", order);
          exit(2);
      }
      break;

    default:
      fprintf(stderr, "nspin = %i not recognized.\n", nspin);
      exit(2);
  }

  // Allocate for CUDA
  copy_arrays_to_cuda(data);
}

void free_arrays_in_cuda_memory (values_t *data)
{
	//printf ("Comenzamos a liberar la memoria en el Device 1\n");

	// Liberamos la memoria de los arrays.
	if (data->zk != NULL) {
	    cudaFree(data->zkCUDA);
	}
	if (data->vrho != NULL) {
	    cudaFree(data->vrhoCUDA);
	}
	if (data->vsigma !=NULL) {
	    cudaFree(data->vsigmaCUDA);
	}

	//printf ("Comenzamos a liberar la memoria en el Device 2 \n");

	if (data->v2rho2 != NULL) {
	    cudaFree(data->v2rho2CUDA);
	}
	if (data->v2rhosigma != NULL) {
	    cudaFree(data->v2rhosigmaCUDA);
	}
	if (data->v2sigma2 != NULL) {
	    cudaFree(data->v2sigma2CUDA);
	}

	//printf ("Comenzamos a liberar la memoria en el Device 3 \n");

	if (data->v3rho3 != NULL) {
	    cudaFree(data->v3rho3CUDA);
	}
	if (data->v3rho2sigma != NULL) {
	    cudaFree(data->v3rho2sigmaCUDA);
	}
	if (data->v3rhosigma2 != NULL) {
	    cudaFree(data->v3rhosigma2CUDA);
	}
	if (data->v3sigma3 != NULL) {
	    cudaFree(data->v3sigma3CUDA);
	}

}


void free_memory(values_t val)
{
  // Free the cuda memory...
  free_arrays_in_cuda_memory (&val);

  // Now the normal boring memory...
  free(val.rho);
  free(val.sigma);
  free(val.lapl);
  free(val.tau);
  free(val.zk);
  free(val.vrho);
  free(val.vsigma);
  free(val.vlapl);
  free(val.vtau);
  free(val.v2rho2);
  free(val.v2tau2);
  free(val.v2lapl2);
  free(val.v2rhotau);
  free(val.v2rholapl);
  free(val.v2lapltau);
  free(val.v2sigma2);
  free(val.v2rhosigma);
  free(val.v2sigmatau);
  free(val.v2sigmalapl);
  free(val.v3rho3);
}

values_t read_data(const char *file, int nspin, int order) {
  /* Format string */
  static const char fmt[]="%lf %lf %lf %lf %lf %lf %lf %lf %lf";

  /* Data buffer */
  char buf[BUFSIZE];
  char *cp;
  /* Input data file */
  FILE *in;
  /* Loop index */
  int i;
  /* Amount of points succesfully read */
  int nsucc;
  /* Returned data */
  values_t data;

  /* Helper variables */
  double rhoa, rhob;
  double sigmaaa, sigmaab, sigmabb;
  double lapla, laplb;
  double taua, taub;

  /* Open file */
  in=fopen(file,"r");
  if(!in) {
    fprintf(stderr,"Error opening input file %s.\n",file);
    exit(3);
  }

  /* Read amount of data points */
  cp=fgets(buf,BUFSIZE,in);
  if(cp!=buf) {
    fprintf(stderr,"Error reading amount of data points.\n");
    exit(5);
  }
  nsucc=sscanf(buf,"%i",&data.n);
  if(nsucc!=1) {
    fprintf(stderr,"Error reading amount of input data points.\n");
    exit(4);
  }

  /* Allocate memory */
  allocate_memory(&data, nspin, order);

  for(i=0;i<data.n;i++) {
    /* Next line of input */
    cp=fgets(buf,BUFSIZE,in);
    if(cp!=buf) {
      fprintf(stderr,"Read error on line %i.\n",i+1);
      free_memory(data);
      exit(5);
    }
    /* Read data */
    nsucc=sscanf(buf, fmt, &rhoa, &rhob, &sigmaaa, &sigmaab, &sigmabb,	\
		 &lapla, &laplb, &taua, &taub);

    /* Error control */
    if(nsucc!=9) {
      fprintf(stderr,"Read error on line %i: only %i entries read.\n",i+1,nsucc);
      free_memory(data);
      exit(5);
    }

    /* Store data (if clause suboptimal here but better for code clarity) */
    if(nspin==XC_POLARIZED) {
      data.rho[2*i]=rhoa;
      data.rho[2*i+1]=rhob;
      data.sigma[3*i]=sigmaaa;
      data.sigma[3*i+1]=sigmaab;
      data.sigma[3*i+2]=sigmabb;
      data.lapl[2*i]=lapla;
      data.lapl[2*i+1]=laplb;
      data.tau[2*i]=taua;
      data.tau[2*i+1]=taub;
    } else {
      /* Construct full density data from alpha and beta channels */
      data.rho[i]=rhoa + rhob;
      data.sigma[i]=sigmaaa + sigmabb + 2.0*sigmaab;
      data.lapl[i]=lapla + laplb;
      data.tau[i]=taua + taub;
    }
  }

  /* Close input file */
  fclose(in);

  return data;
}


void copy_arrays_from_cuda (values_t *data)
{
	unsigned int mem_size_for_cuda_arrays = sizeof(double) * data->n;
	// Copiamos de vuelta lo que dejo en los arrays.
	if (data->zk != NULL) {
		cudaMemcpy(data->zk, data->zkCUDA, mem_size_for_cuda_arrays, cudaMemcpyDeviceToHost);
	}
	if (data->vrho != NULL) {
		cudaMemcpy(data->vrho, data->vrhoCUDA, mem_size_for_cuda_arrays, cudaMemcpyDeviceToHost);
	}
	if (data->vsigma != NULL) {
		cudaMemcpy(data->vsigma, data->vsigmaCUDA, mem_size_for_cuda_arrays, cudaMemcpyDeviceToHost);
	}
	//printf ("Copiamos las derivadas segundas de vuelta al Host \n");

	if (data->v2rho2 != NULL) {
		cudaMemcpy(data->v2rho2, data->v2rho2CUDA, mem_size_for_cuda_arrays, cudaMemcpyDeviceToHost);
	}
	if (data->v2rhosigma != NULL) {
		cudaMemcpy(data->v2rhosigma, data->v2rhosigmaCUDA, mem_size_for_cuda_arrays, cudaMemcpyDeviceToHost);
	}
	if (data->v2sigma2 != NULL) {
		cudaMemcpy(data->v2rhosigma, data->v2sigma2CUDA, mem_size_for_cuda_arrays, cudaMemcpyDeviceToHost);
	}

	//printf ("Copiamos las derivadas terceras de vuelta al Host \n");

	if (data->v3rho3 != NULL) {
		cudaMemcpy(data->v3rho3, data->v3rho3CUDA, mem_size_for_cuda_arrays, cudaMemcpyDeviceToHost);
	}
	if (data->v3rho2sigma != NULL) {
		cudaMemcpy(data->v3rho2sigma, data->v3rho2sigmaCUDA, mem_size_for_cuda_arrays, cudaMemcpyDeviceToHost);
	}
	if (data->v3rhosigma2 != NULL) {
		cudaMemcpy(data->v3rhosigma2, data->v3rhosigma2CUDA, mem_size_for_cuda_arrays, cudaMemcpyDeviceToHost);
	}
	if (data->v3sigma3 != NULL) {
		cudaMemcpy(data->v3sigma3, data->v3sigma3CUDA, mem_size_for_cuda_arrays, cudaMemcpyDeviceToHost);
	}
}

xc_func_type_cuda* copy_data_structures_to_cuda (xc_func_type_cuda* p)
{
    //printf("copy_data_structures_to_cuda(...)\n");
	// Variables en el device
	xc_func_info_type_cuda* infoTypeCUDA = NULL;
	xc_func_type_cuda* pCUDA = NULL;

	// Tamanios de los structs
	int func_type_size = sizeof(xc_func_type_cuda);
	int func_info_type_size = sizeof(xc_func_info_type_cuda);

	// Reservamos memoria y copiamos info_type
	cudaMalloc((void **)&infoTypeCUDA, func_info_type_size);
	cudaMemcpy(infoTypeCUDA, (xc_func_info_type_cuda*)(p->info), func_info_type_size, cudaMemcpyHostToDevice);

	// Reservamos memoria para le primer parametro.
	cudaMalloc((void **)&pCUDA, func_type_size);

	// Copiamos el primer parametro a memoria.
	cudaMemcpy(pCUDA, p, func_type_size, cudaMemcpyHostToDevice);

	// Deep copy de ->info.
	cudaMemcpy(&(pCUDA->info), &infoTypeCUDA, sizeof(infoTypeCUDA), cudaMemcpyHostToDevice);

	return pCUDA;
}


/*----------------------------------------------------------*/
int main(int argc, char *argv[])
{
  printf("REGRESSION modified by eduardito O_o \n");
  int func_id, nspin, order, i;
  /* Helpers for properties that may not have been implemented */
  double *zk, *vrho, *v2rho2, *v3rho3;

  static const char efmt[] =" % .16e";
  static const char efmt2[]=" % .16e % .16e";
  static const char efmt3[]=" % .16e % .16e % .16e";
  static const char sfmt[] =" %23s";
  static const char sfmt2[]=" %23s %23s";
  static const char sfmt3[]=" %23s %23s %23s";

  if(argc != 6) {
    fprintf(stderr, "Usage:\n%s funct nspin order input output\n", argv[0]);
    exit(1);
  }

  /* Get functional id */
  func_id = xc_functional_get_number(argv[1]);
  if(func_id <= 0) {
    fprintf(stderr, "Functional '%s' not found\n", argv[1]);
    exit(1);
  }

  /* Spin-polarized or unpolarized ? */
  nspin = atoi(argv[2]);

  /* Order of derivatives to compute */
  order = atoi(argv[3]);

  /* Data array */
  values_t d;
  /* Functional evaluator */
  xc_func_type_cuda func;
  /* Flags for functional */
  int flags;
  /* Functional family */
  int family;
  /* Output file */
  FILE *out;
  /* Output file name */
  char *fname;

  /* Read in data */
  d = read_data(argv[4], nspin, order);

  /* Initialize functional */
  if(xc_func_init_cuda(&func, func_id, nspin)) {
    fprintf(stderr, "Functional '%d' (%s) not found.\nPlease report a bug against functional_get_number.\n", func_id, argv[1]);
    exit(1);
  }
  /* Get flags */
  flags=func.info->flags;
  family=func.info->family;

  /* Set helpers */
  zk     = (flags & XC_FLAGS_HAVE_EXC) ? d.zkCUDA     : NULL;
  vrho   = (flags & XC_FLAGS_HAVE_VXC) ? d.vrhoCUDA   : NULL;
  v2rho2 = (flags & XC_FLAGS_HAVE_FXC) ? d.v2rho2CUDA : NULL;
  v3rho3 = (flags & XC_FLAGS_HAVE_KXC) ? d.v3rho3CUDA : NULL;

  /* Copy data structures to gpu*/
  xc_func_type_cuda* funcCUDA = NULL; //copy_data_structures_to_cuda(&func);

  //if (funcCUDA == NULL) {
    //fprintf(stderr, "Functional '%d' (%s) coulndt be copied to gpu memory.\nPlease report a bug against functional_get_number.\n", func_id, argv[1]);
    //exit(1);
  //}

  // TODO: copiar los parametros para CUDA
  copy_arrays_to_cuda(&d);

  // print parameters 
  print_work_gga_c_params(&func, d.n,
    d.rho, d.sigma, d.zk, d.vrho, d.vsigma,
    d.v2rho2, d.v2rhosigma, d.v2sigma2,
    NULL, NULL, NULL, NULL); 

  // TODO: copiar los parametros para CUDA
  //copy_arrays_to_cuda(&d);

  /* Evaluate xc functional */
  /*
  switch(family) {
  case XC_FAMILY_LDA:
    xc_lda_cuda(&func, d.n, d.rho, zk, vrho, v2rho2, v3rho3);
    break;
  case XC_FAMILY_GGA:
  case XC_FAMILY_HYB_GGA:
    xc_gga(&func, d.n, d.rho, d.sigma, zk, vrho, d.vsigma,\
    v2rho2, d.v2rhosigma, d.v2sigma2, NULL, NULL, NULL, NULL);
    break;
  case XC_FAMILY_MGGA:
  case XC_FAMILY_HYB_MGGA:
    xc_mgga_cuda(&func, d.n, d.rho, d.sigma, d.lapl, d.tau, zk, vrho, d.vsigma, d.vlapl, d.vtau, \
     v2rho2, d.v2sigma2, d.v2lapl2, d.v2tau2, d.v2rhosigma, d.v2rholapl, d.v2rhotau, \
     d.v2sigmalapl, d.v2sigmatau, d.v2lapltau);
    break;

  default:
    fprintf(stderr,"Support for family %i not implemented.\n",family);
    free_memory(d);
    if (funcCUDA != NULL) {
	cudaFree(funcCUDA->info);
        cudaFree(funcCUDA);
    }
    exit(1);
  }
  */

  /* Using the cuda arrays */
  switch(family) {
  case XC_FAMILY_LDA:
    xc_lda_cuda(&func, d.n, d.rhoCUDA, zk, vrho, v2rho2, v3rho3);
    break;
  case XC_FAMILY_GGA:
  case XC_FAMILY_HYB_GGA:
    xc_gga_cuda(&func, d.n, d.rhoCUDA, d.sigmaCUDA, zk, vrho, d.vsigmaCUDA,\
    v2rho2, d.v2rhosigmaCUDA, d.v2sigma2CUDA, NULL, NULL, NULL, NULL);
    break;
  case XC_FAMILY_MGGA:
  case XC_FAMILY_HYB_MGGA:
    xc_mgga_cuda(&func, d.n, d.rhoCUDA, d.sigmaCUDA, d.laplCUDA, d.tauCUDA, zk, vrho, d.vsigmaCUDA, d.vlaplCUDA, d.vtauCUDA, \
     v2rho2, d.v2sigma2CUDA, d.v2lapl2CUDA, d.v2tau2CUDA, d.v2rhosigmaCUDA, d.v2rholaplCUDA, d.v2rhotauCUDA, \
     d.v2sigmalaplCUDA, d.v2sigmatauCUDA, d.v2lapltauCUDA);
    break;

  default:
    fprintf(stderr,"Support for family %i not implemented.\n",family);
    free_memory(d);
    if (funcCUDA != NULL) {
	cudaFree((void*)(funcCUDA->info));
	cudaFree(funcCUDA);
    }
    exit(1);
  }


  /* Open output file */
  fname = argv[5];
  out = fopen(fname,"w");
  if(!out) {
    fprintf(stderr,"Error opening output file %s.\n",fname);
    free_memory(d);
    if (funcCUDA != NULL) {
	cudaFree((void*)(funcCUDA->info));
        cudaFree(funcCUDA);
    }
    exit(1);
  }

  copy_arrays_from_cuda(&d);

  // print parameters 
  print_work_gga_c_params(&func, d.n, 
    d.rho, d.sigma, d.zk, d.vrho, d.vsigma,
    d.v2rho2, d.v2rhosigma, d.v2sigma2,
    NULL, NULL, NULL, NULL); 


  /* Functional id and amount of lines in output */
  fprintf(out, "%i %i %i\n", func_id, d.n, order);

  switch (order) {
    case (0): /* energy */
      fprintf(out, sfmt, "zk");
      break;
    case (1): /* first order derivatives */
      if (nspin == XC_POLARIZED) {
        fprintf(out, sfmt2, "vrho(a)", "vrho(b)");
        if (family & (XC_FAMILY_GGA | XC_FAMILY_HYB_GGA | XC_FAMILY_MGGA | XC_FAMILY_HYB_MGGA))
          fprintf(out, sfmt3, "vsigma(aa)", "vsigma(ab)", "vsigma(bb)");
        if (family & (XC_FAMILY_MGGA | XC_FAMILY_HYB_MGGA)) {
          fprintf(out, sfmt2, "vlapl(a)", "vlapl(b)");
          fprintf(out, sfmt2, "vtau(a)", "vtau(b)");
        }
      } else {
        fprintf(out, sfmt, "vrho");
        if (family & (XC_FAMILY_GGA | XC_FAMILY_HYB_GGA | XC_FAMILY_MGGA | XC_FAMILY_HYB_MGGA))
          fprintf(out, sfmt, "vsigma");
        if(family & (XC_FAMILY_MGGA | XC_FAMILY_HYB_MGGA)) {
          fprintf(out, sfmt, "vlapl");
          fprintf(out, sfmt, "vtau");
        }
      }
      break;

    case (2): /* second order derivatives */
      if (nspin == XC_POLARIZED) {
        fprintf(out,sfmt3,"v2rho(aa)","v2rho(ab)","v2rho(bb)");
        if(family & (XC_FAMILY_GGA | XC_FAMILY_HYB_GGA | XC_FAMILY_MGGA | XC_FAMILY_HYB_MGGA)) {
          fprintf(out, sfmt3, "v2sigma2(aa-aa)", "v2sigma2(aa-ab)", "v2sigma2(aa-bb)");
          fprintf(out, sfmt3, "v2sigma2(ab-ab)", "v2sigma2(ab-bb)", "v2sigma2(bb-bb)");
          fprintf(out, sfmt3, "v2rho(a)sigma(aa)", "v2rho(a)sigma(ab)", "v2rho(a)sigma(bb)");
          fprintf(out, sfmt3, "v2rho(b)sigma(aa)", "v2rho(b)sigma(ab)", "v2rho(b)sigma(bb)");
        }
        if(family & (XC_FAMILY_MGGA | XC_FAMILY_HYB_MGGA)) {
          fprintf(out, sfmt3, "v2lapl2(aa)", "v2lapl2(ab)", "v2lapl2(bb)");
          fprintf(out, sfmt3, "v2tau2(aa)", "v2tau2(ab)", "v2tau2(bb)");
          fprintf(out, sfmt3, "v2rholapl(aa)", "v2rholapl(ab)", "v2rholapl(bb)");
          fprintf(out, sfmt3, "v2rhotau(aa)", "v2rhotau(ab)", "v2rhotau(bb)");
          fprintf(out, sfmt3, "v2lapltau(aa)", "v2lapltau(ab)", "v2lapltau(bb)");
          fprintf(out, sfmt3, "v2sigma(aa)tau(a)", "v2sigma(aa)tau(b)", "v2sigma(ab)tau(a)");
          fprintf(out, sfmt3, "v2sigma(ab)tau(b)", "v2sigma(bb)tau(a)", "v2sigma(bb)tau(b)");
          fprintf(out, sfmt3, "v2sigma(aa)lapl(a)", "v2sigma(aa)lapl(b)", "v2sigma(ab)lapl(a)");
          fprintf(out, sfmt3, "v2sigma(ab)lapl(b)", "v2sigma(bb)lapl(a)", "v2sigma(bb)lapl(b)");
        }
      } else {
        fprintf(out,sfmt,"v2rho");
        if(family & (XC_FAMILY_GGA | XC_FAMILY_HYB_GGA | XC_FAMILY_MGGA | XC_FAMILY_HYB_MGGA)) {
          fprintf(out, sfmt, "v2sigma2");
          fprintf(out, sfmt, "v2rhosigma");
        }

        if(family & (XC_FAMILY_MGGA | XC_FAMILY_HYB_MGGA)) {
          fprintf(out, sfmt, "v2lapl2");
          fprintf(out, sfmt, "v2tau2");
          fprintf(out, sfmt, "v2rholapl");
          fprintf(out, sfmt, "v2rhotau");
          fprintf(out, sfmt, "v2lapltau");
          fprintf(out, sfmt, "v2sigmatau");
          fprintf(out, sfmt, "v2sigmalapl");
        }
      }
      break;

    default: /* higher order derivatives ... to be done */
      fprintf(stderr, "order = %i not recognized.\n", order);
      exit(2);
  }
  fprintf(out,"\n");

  /* Loop over data points */
  for(i=0;i<d.n;i++) {

    switch (order) {
      case (0): /* energy */
        fprintf(out, efmt, d.zk[i]);
        break;
      case (1): /* first order derivatives */
        if (nspin == XC_POLARIZED) {
          fprintf(out, efmt2, d.vrho[2 * i], d.vrho[2 * i + 1]);
          if (family & (XC_FAMILY_GGA | XC_FAMILY_HYB_GGA | XC_FAMILY_MGGA | XC_FAMILY_HYB_MGGA))
            fprintf(out, efmt3, d.vsigma[3 * i], d.vsigma[3 * i + 1], d.vsigma[3 * i + 2]);
          if (family & (XC_FAMILY_MGGA | XC_FAMILY_HYB_MGGA)) {
            fprintf(out, efmt2, d.vlapl[2 * i], d.vlapl[2 * i + 1]);
            fprintf(out, efmt2, d.vtau[2 * i], d.vtau[2 * i + 1]);
          }
        } else {
          fprintf(out, efmt, d.vrho[i]);
          if (family & (XC_FAMILY_GGA | XC_FAMILY_HYB_GGA | XC_FAMILY_MGGA | XC_FAMILY_HYB_MGGA))
            fprintf(out, efmt, d.vsigma[i]);
          if (family & (XC_FAMILY_MGGA | XC_FAMILY_HYB_MGGA)) {
            fprintf(out, efmt, d.vlapl[i]);
            fprintf(out, efmt, d.vtau[i]);
          }
        }
        break;

      case (2): /* second order derivatives */
        if (nspin == XC_POLARIZED) {
          fprintf(out, efmt3, d.v2rho2[3*i], d.v2rho2[3*i + 1], d.v2rho2[3*i + 2]);
          if(family & (XC_FAMILY_GGA | XC_FAMILY_HYB_GGA | XC_FAMILY_MGGA | XC_FAMILY_HYB_MGGA)) {
            fprintf(out, efmt3, d.v2sigma2[6*i], d.v2sigma2[6*i + 1], d.v2sigma2[6*i + 2]);
            fprintf(out, efmt3, d.v2sigma2[6*i + 3], d.v2sigma2[6*i + 4], d.v2sigma2[6*i + 5]);
            fprintf(out, efmt3, d.v2rhosigma[6*i], d.v2rhosigma[6*i + 1], d.v2rhosigma[6*i + 2]);
            fprintf(out, efmt3, d.v2rhosigma[6*i + 3], d.v2rhosigma[6*i + 4], d.v2rhosigma[6*i + 5]);
          }
          if(family & (XC_FAMILY_MGGA | XC_FAMILY_HYB_MGGA)) {
            fprintf(out, efmt3, d.v2lapl2[3*i], d.v2lapl2[3*i + 1], d.v2lapl2[3*i + 2]);
            fprintf(out, efmt3, d.v2tau2[3*i], d.v2tau2[3*i + 1], d.v2tau2[3*i + 2]);
            fprintf(out, efmt3, d.v2rholapl[3*i], d.v2rholapl[3*i + 1], d.v2rholapl[3*i + 2]);
            fprintf(out, efmt3, d.v2rhotau[3*i], d.v2rhotau[3*i + 1], d.v2rhotau[3*i + 2]);
            fprintf(out, efmt3, d.v2lapltau[3*i], d.v2lapltau[3*i + 1], d.v2lapltau[3*i + 2]);
            fprintf(out, efmt3, d.v2sigmatau[3*i], d.v2sigmatau[3*i + 1], d.v2sigmatau[3*i + 2]);
            fprintf(out, efmt3, d.v2sigmatau[3*i + 3], d.v2sigmatau[3*i + 4], d.v2sigmatau[3*i + 5]);
            fprintf(out, efmt3, d.v2sigmalapl[3*i], d.v2sigmalapl[3*i + 1], d.v2sigmalapl[3*i + 2]);
            fprintf(out, efmt3, d.v2sigmalapl[3*i + 3], d.v2sigmalapl[3*i + 4], d.v2sigmalapl[3*i + 5]);
          }
        } else {
          fprintf(out, efmt, d.v2rho2[i]);
          if(family & (XC_FAMILY_GGA | XC_FAMILY_HYB_GGA | XC_FAMILY_MGGA | XC_FAMILY_HYB_MGGA)) {
            fprintf(out, efmt, d.v2sigma2[i]);
            fprintf(out, efmt, d.v2rhosigma[i]);
          }
          if(family & (XC_FAMILY_MGGA | XC_FAMILY_HYB_MGGA)) {
            fprintf(out, efmt, d.v2lapl2[i]);
            fprintf(out, efmt, d.v2tau2[i]);
            fprintf(out, efmt, d.v2rholapl[i]);
            fprintf(out, efmt, d.v2rhotau[i]);
            fprintf(out, efmt, d.v2lapltau[i]);
            fprintf(out, efmt, d.v2sigmatau[i]);
            fprintf(out, efmt, d.v2sigmalapl[i]);
          }
        }
        break;

     default: /* higher order derivatives ... to be done */
        fprintf(stderr, "order = %i not recognized.\n", order);
        exit(2);
    }

    fprintf(out,"\n");
  }

  xc_func_end_cuda(&func);
  free_memory(d);
  if (funcCUDA != NULL) {
      cudaFree((void*)(funcCUDA->info));
      cudaFree(funcCUDA);
  }
  fclose(out);
  return 0;
}
