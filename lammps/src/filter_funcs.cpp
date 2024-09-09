/*
 * =====================================================================================
 *
 *       Filename:  filter_funcs.cpp
 *
 *    Description:  module contains filter functions that are used to compute features
 *
 *        Version:  1.0
 *        Created:  01/05/2019 01:06:45 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (Kuang Yu), 
 *   Organization:  
 *
 * =====================================================================================
 */
#include <cmath>
#include <stdlib.h>
#include <mylibs.h>

using namespace std;
using namespace Eigen;

// tanh switch function
VectorXd fc_tanh(const VectorXd & x, double * Rc)
{
  ArrayXd results;
  results = 1. - x.array()/(*Rc);
  results = results.tanh().cube();
  return results.matrix();
}

// derivative of fc_tanh
VectorXd dfc_tanh(const VectorXd & x, double * Rc)
{
  ArrayXd results;
  results = 1. - x.array()/(*Rc);
  ArrayXd tanh2_x = results.tanh().square();
  results = -3*tanh2_x*(1-tanh2_x)/(*Rc);
  return results.matrix();
}

// an 1-d gaussian function without normalization coefficients
// exp(-\eta(r-Rs)^2)
// p is the parameter: Rs and eta
VectorXd gaussian_no_coeff(const VectorXd & x, double * p)
{
  double rs=p[0], eta=p[1];
  ArrayXd results = exp(-eta*(x.array()-rs).square());
  return results.matrix();
}

VectorXd dgaussian_no_coeff(const VectorXd & x, double * p)
{
  double rs=p[0], eta=p[1];
  ArrayXd results = -2.*eta*(x.array()-rs) * gaussian_no_coeff(x, p).array();
  return results.matrix();
}

// The G2 feature component: gauss(x)*fc(x)
// p: (rs, eta, Rc)
VectorXd feature_term_g2(const VectorXd & x, double * p)
{
  ArrayXd results;
  results = gaussian_no_coeff(x, p).array() * fc_tanh(x, p+2).array();
  return results.matrix();
}

VectorXd dfeature_term_g2(const VectorXd & x, double * p)
{
  ArrayXd results;
  results = dgaussian_no_coeff(x, p).array()*fc_tanh(x, p+2).array() 
          + gaussian_no_coeff(x, p).array()*dfc_tanh(x, p+2).array();
  return results.matrix();
}

// the g4 feature function, rlist1 is a list of vector rj-ri
//                          rlist2 is a list of vector rk-ri
VectorXd feature_term_g4(const MatrixXd & rlist1, const MatrixXd & rlist2, double * p)
{
  // unpack parameters
  double zeta = p[0];
  double lambda = p[1];
  double eta = p[2];
  double Rc = p[3];
  int n_data = rlist1.cols();
  assert(n_data == rlist2.cols());
  ArrayXd cos_A(n_data), results(n_data);
  ArrayXd rnorm1 = rlist1.colwise().norm();
  ArrayXd rnorm2 = rlist2.colwise().norm();
  ArrayXd term_exp = exp(-eta * (rnorm1.square()+rnorm2.square()));
  ArrayXd fc_ij = fc_tanh(rnorm1.matrix(), &Rc);
  ArrayXd fc_ik = fc_tanh(rnorm2.matrix(), &Rc);
  for (int i=0; i<n_data; i++) {
    cos_A(i) = rlist1.col(i).matrix().dot(rlist2.col(i).matrix());
  }
  cos_A = cos_A / rnorm1 / rnorm2;
  results = pow(2., 1-zeta) * (1+lambda*cos_A).pow(zeta) * term_exp * fc_ij * fc_ik;
  return results;
}

