/*
 * =====================================================================================
 *
 *       Filename:  spline.cpp
 *
 *    Description:  module that deals with b-spline
 *
 *        Version:  1.0
 *        Created:  08/15/2019 10:10:33 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (Kuang Yu), 
 *   Organization:  
 *
 * =====================================================================================
 */
#include <stdlib.h>
#include <cmath>
#include <vector>
#include <mylibs.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

using namespace std;
using namespace Eigen;
namespace py = pybind11;

VectorXd default_zero_func(const VectorXd & x, double *p)
{
  int N = x.size();
  VectorXd res = VectorXd::Zero(N);
  return res; 
}

// compute the M_n(u) table, NOTE: du must be divisible by 1.0
VectorXd compute_Mn(int order, double du)
{
  double ub = -1.0;
  double ue = order;
  double u;
  int N = (int) ((ue - ub)/du + 1.01);
  int di1 = 1.0/du;  // index interval for dx=1
  VectorXd ulist(N);
  VectorXd Mn(N);
  u = ub;
  // construct M2
  for(int i=0; i<N; i++)
  {
    ulist(i) = u;
    if (u < 0.0 or u > 2.0) 
    {
      Mn(i) = 0.0;
    }
    else if (u > 0.0 and u < 1.0)
    {
      Mn(i) = u;
    }
    else
    {
      Mn(i) = 2.0 - u;
    }
    u += du;
  }
  // high order spline basis
  for (int n=3; n<=order; n++)
  {
    for (int i=N-1; i>=di1; i--)
    {
      u = ulist(i);
      Mn(i) = u/(n-1) * Mn(i) + (n-u)/(n-1) * Mn(i-di1);
    }
  }
  return Mn;
}

func_table construct_Mn_table(int order, double du)
{
  VectorXd vals = compute_Mn(order, du);
  double ub = -1.0;
  double ue = order;
  double p;
  int N = (ue-ub)/du;
  func_table table = func_table(ub, ue, N, default_zero_func, &p);
  table.set_table(vals);
  return table;
}

PYBIND11_MODULE(_spline, m)
{
  m.doc() = "Cardinal B-spline module";
  m.def("compute_Mn", &compute_Mn);
}
