/*
 * =====================================================================================
 *
 *       Filename:  tabulate.cpp
 *
 *    Description:  module that deals with the tabulated functions
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
#include <stdlib.h>
#include <mylibs.h>

using namespace std;
using namespace Eigen;

VectorXd func_table::get_table()
{
  return this->table;
}

VectorXd func_table::get_xlist()
{
  return this->xlist;
}

void func_table::set_table(const VectorXd & table)
{
  this->table = table;
  return;
}

func_table::func_table(double xb, double xe, int N, VectorXd (*f) (const VectorXd & x, double* p), double *p)
{
  this->xb = xb;
  this->xe = xe;
  this->N = N;
  this->len_table = N + 1;
  this->dx = (xe-xb)/N;
  this->xlist = VectorXd::LinSpaced(N+1, xb, xe);
  this->table = (*f) (this->xlist, p);
}

// take values using linear interpolation
double func_table::value(double x) {
  // out of bound
  if (x < this->xb or x > this->xe) return 0;
  double s = (x - this->xb)/this->dx;
  int i0 = floor(s);
  int i1 = i0 + 1;
  // use simple linear interpolation
  return (i1-s) * (this->table(i0)) + (s-i0) * (this->table(i1));
}

// take values using linear interpolation, batch process
VectorXd func_table::value(VectorXd x) {
  int n = x.size();
  ArrayXd s = (x.array() - this->xb)/this->dx;
  ArrayXi i0 = s.floor().cast<int>();
  ArrayXi i1 = i0 + 1;
  ArrayXd lb, hb; // lower/higher bound
  lb.resize(n);
  hb.resize(n);
  for (int i=0; i<n; i++) {
    // check bounds
    if (i0(i) < 0 or i1(i) >= this->len_table) {
      lb(i) = 0;
      hb(i) = 0;
      continue;
    }
    lb(i) = this->table(i0(i));
    hb(i) = this->table(i1(i));
  }
  ArrayXd results = (i1.cast<double>()-s)*lb + (s-i0.cast<double>())*hb;
  return results.matrix();
}

