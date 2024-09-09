/*
 * =====================================================================================
 *
 *       Filename:  fft.cpp
 *
 *    Description:  wrappers for fourier transform
 *
 *        Version:  1.0
 *        Created:  08/16/2019 10:01:15 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <omp.h>
#include <fftw3.h>
#include <mylibs.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

using namespace std;
using namespace Eigen;

// fftw wrapper for 3d fftw calls
// tensor are formulated as 1d vector, with the first index running fastest
VectorXcd dft3d_forward_r2c(VectorXd tensor_in, int K1, int K2, int K3)
{
  int K_complex = (K1/2+1)*K2*K3;
  VectorXcd tensor_out;
  tensor_out.resize(K_complex);
  double * p_in = tensor_in.data();
  fftw_complex * p_out;
  p_out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * K_complex);
  fftw_plan plan;
  plan = fftw_plan_dft_r2c_3d(K3, K2, K1, p_in, p_out, FFTW_ESTIMATE);
  fftw_execute(plan);
  fftw_destroy_plan(plan);
  // copy results to Eigen form
  for (int k3=0; k3<K3; k3++)
  {
    for (int k2=0; k2<K2; k2++)
    {
      for (int k1=0; k1<K1/2+1; k1++)
      {
        int k = ind_3d_to_1d(k1, k2, k3, K1/2+1, K2, K3);
        tensor_out(k) = p_out[k][0] + p_out[k][1]*J;
      }
    }
  }
//  Map<VectorXcd> tensor_out(p_out, K_complex);
  fftw_free(p_out);

  return tensor_out;
}
