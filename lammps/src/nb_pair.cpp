/*
 * =====================================================================================
 *
 *       Filename:  nb_pair.cpp
 *
 *    Description:  module to calculate the real space pairwise nonbinding interactions
 *
 *        Version:  1.0
 *        Created:  08/29/2019 03:52:39 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (Kuang Yu), 
 *   Organization:  
 *
 * =====================================================================================
 */
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <omp.h>
#include <mylibs.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

using namespace std;
using namespace Eigen;
namespace py = pybind11;

double nb_pair(const MatrixXd & positions, const ArrayXd & sigma, const ArrayXd & epsil, int n_atoms,
    const MatrixXd & box, const MatrixXi & exclusions, int n_excl, double r_cut)
{
  MatrixXd box_inv = box.inverse();
  // construct neighbor list
  VectorXi atom_labels;
  atom_labels.resize(n_atoms);
  for(int i=0; i<n_atoms; i++) atom_labels(i) = i;
  Tensor<int, 4> celllist = construct_cell_list(positions, atom_labels, n_atoms, box, box_inv, r_cut);
  NbsRetType nbl = find_neighbours_for_all(positions, celllist, n_atoms, r_cut, box, box_inv);

  double energy = 0.0;

  energy += energy_lennard_jones(positions, sigma, epsil, n_atoms, box, box_inv, exclusions, n_excl,
      nbl, 1);

  return energy;
}


// combination rule:
// comb_rule = 1: sigma_ij = (sigma_i + sigma_j)/2;   spsil_ij = sqrt(epsil_i * epsil_j)
// comb_rule = 2: sigma_ij = sqrt(sigma_i * sigma_j); epsil_ij = sqrt(epsil_i * epsil_j)
double energy_lennard_jones(const MatrixXd & positions, const ArrayXd & sigma, const ArrayXd & epsil,
    int n_atoms, const MatrixXd & box, const MatrixXd & box_inv, const MatrixXi & exclusions, int n_excl, 
    const NbsRetType & nbl, int comb_rule)
{
  double energy=0.0;
  ArrayXd energies = ArrayXd::Zero(n_atoms);
#pragma omp parallel
#pragma omp for nowait
  for (int ia1=0; ia1<n_atoms; ia1++)
  {
    if (abs(epsil(ia1)) < 0.00000001) continue;
    energies(ia1) = energy_lj_peratom(ia1, positions, sigma, epsil, n_atoms, comb_rule, nbl);
  }
  energy += energies.sum();
  // deal with exclusions
  MatrixXd pos1, pos2;
  pos1.resize(3, n_excl);
  pos2.resize(3, n_excl);
  ArrayXd sig1, sig2, eps1, eps2;
  sig1.resize(n_excl);
  sig2.resize(n_excl);
  eps1.resize(n_excl);
  eps2.resize(n_excl);
#pragma omp parallel
#pragma omp for nowait
  for (int iexcl=0; iexcl<n_excl; iexcl++)
  {
    int ia1 = exclusions.col(iexcl)(0);
    int ia2 = exclusions.col(iexcl)(1);
    pos1.col(iexcl) = positions.col(ia1);
    pos2.col(iexcl) = positions.col(ia2);
    sig1(iexcl) = sigma(ia1);
    sig2(iexcl) = sigma(ia2);
    eps1(iexcl) = epsil(ia1);
    eps2(iexcl) = epsil(ia2);
  }
  ArrayXd sigma_ij, epsil_ij;
  sigma_ij.resize(n_excl);
  epsil_ij.resize(n_excl);
  sigma_ij = combine_pair_coeffs<ArrayXd, ArrayXd>(sig1, sig2, 1, comb_rule);
  epsil_ij = combine_pair_coeffs<ArrayXd, ArrayXd>(eps1, eps2, 2, comb_rule);
  // pbc shifts
  MatrixXd dr_vec = pos2 - pos1;
  MatrixXd ds_vec = box_inv * dr_vec;
  ds_vec -= (ds_vec.array()+0.5).floor().matrix();
  dr_vec = box * ds_vec;
  ArrayXd drdr = dr_vec.colwise().squaredNorm();
  ArrayXd sig2_r2 = sigma_ij * sigma_ij / drdr;
  ArrayXd ens_excl = 2 * epsil_ij * (sig2_r2.pow(6) - sig2_r2.pow(3));

  return energy - ens_excl.sum();
}


double energy_lj_peratom(int ia1, const MatrixXd & positions, const ArrayXd & sigma, const ArrayXd & epsil, 
    int n_atoms, int comb_rule, const NbsRetType & nbl)
{
  double sigma1 = sigma(ia1);
  double epsil1 = epsil(ia1);
  int n_nb = nbl.n_nbs(ia1);
  ArrayXd drdr, sigma_ij, epsil_ij;
  drdr.resize(n_nb);
  sigma_ij.resize(n_nb);
  epsil_ij.resize(n_nb);
  for (int iia2=0; iia2<n_nb; iia2++)
  {
    int ia2 = nbl.nbs.col(ia1)(iia2);
    sigma_ij(iia2) = sigma(ia2);
    epsil_ij(iia2) = epsil(ia2);
    drdr(iia2) = nbl.distances2.col(ia1)(iia2);
  }
  sigma_ij = combine_pair_coeffs<ArrayXd, double>(sigma_ij, sigma1, 1, comb_rule);
  epsil_ij = combine_pair_coeffs<ArrayXd, double>(epsil_ij, epsil1, 2, comb_rule);
  ArrayXd sig2_r2 = sigma_ij*sigma_ij/drdr;
  ArrayXd energy_pair = 2*epsil_ij*(sig2_r2.pow(6) - sig2_r2.pow(3));
  return energy_pair.sum();
}


Vector3d heat_flux_virial_lennard_jones(const MatrixXd & positions, const MatrixXd & velocities, int n_atoms,
    const MatrixXd & box, const MatrixXd & box_inv, const ArrayXd & sigma, const ArrayXd & epsil, int comb_rule,
    const NbsRetType & nbl, const MatrixXi & exclusions, int n_excl)
{
  Vector3d Jflux;
  double Jx, Jy, Jz;
  Jx = 0.0;
  Jy = 0.0;
  Jz = 0.0;
#pragma omp parallel for reduction(+: Jx, Jy, Jz)
  for (int ia1=0; ia1<n_atoms; ia1++)
  {
    double sigma1 = sigma(ia1);
    double epsil1 = epsil(ia1);
    if (abs(epsil1)<0.00000001) continue;
    int n_nb = nbl.n_nbs(ia1);
    for (int iia2=0; iia2<n_nb; iia2++)
    {
      int ia2 = nbl.nbs.col(ia1)(iia2);
      double sigma2 = sigma(ia2);
      double epsil2 = epsil(ia2);
      if (abs(epsil2)<0.00000001) continue;
      double sigma_ij = combine_pair_coeff(sigma1, sigma2, 1, comb_rule);
      double epsil_ij = combine_pair_coeff(epsil1, epsil2, 2, comb_rule);
      double drdr = nbl.distances2.col(ia1)(iia2);
      Vector3d dr_vec = dr_vec_pbc(positions.col(ia1), positions.col(ia2), box, box_inv, 0);
      double sig2_r2 = sigma_ij*sigma_ij/drdr;
      Vector3d vb = velocities.col(ia2);
      double tmp = 2*epsil_ij*(12*pow(sig2_r2, 6) - 6*pow(sig2_r2, 3))/drdr * dr_vec.dot(vb);
      Jx += tmp * dr_vec(0);
      Jy += tmp * dr_vec(1);
      Jz += tmp * dr_vec(2);
    }
  }
// exclusion
  double Jx_excl, Jy_excl, Jz_excl;
  Jx_excl = 0.0;
  Jy_excl = 0.0;
  Jz_excl = 0.0;
//#pragma omp parallel for reduction(+: Jx_excl, Jy_excl, Jz_excl)
  for (int i_excl=0; i_excl<n_excl; i_excl++)
  {
    int ia1 = exclusions.col(i_excl)(0);
    int ia2 = exclusions.col(i_excl)(1);
    double sig1 = sigma(ia1);
    double sig2 = sigma(ia2);
    double eps1 = epsil(ia1);
    double eps2 = epsil(ia2);
    double sigma_ij = combine_pair_coeff(sig1, sig2, 1, comb_rule);
    double epsil_ij = combine_pair_coeff(eps1, eps2, 2, comb_rule);
    if (abs(epsil_ij)<0.00000001) continue;
    Vector3d r1 = positions.col(ia1);
    Vector3d r2 = positions.col(ia2);
    Vector3d dr_vec = dr_vec_pbc(positions.col(ia1), positions.col(ia2), box, box_inv, 0);
    double drdr = dr_vec.dot(dr_vec);
    double sig2_r2 = sigma_ij*sigma_ij/drdr;
    Vector3d va = velocities.col(ia1);
    Vector3d vb = velocities.col(ia2);
    double tmp = 2*epsil_ij*(12*pow(sig2_r2, 6) - 6*pow(sig2_r2, 3))/drdr;
    Jx_excl += tmp * dr_vec.dot(va+vb) * dr_vec(0);
    Jy_excl += tmp * dr_vec.dot(va+vb) * dr_vec(1);
    Jz_excl += tmp * dr_vec.dot(va+vb) * dr_vec(2);
  }
  Jflux(0) = Jx - Jx_excl;
  Jflux(1) = Jy - Jy_excl;
  Jflux(2) = Jz - Jz_excl;
  return Jflux;
}


Vector3d heat_flux_convection_lennard_jones(const MatrixXd & positions, const MatrixXd & velocities, int n_atoms,
    const MatrixXd & box, const MatrixXd & box_inv, const ArrayXd & sigma, const ArrayXd & epsil, int comb_rule,
    const NbsRetType & nbl, const MatrixXi & exclusions, int n_excl)
{
  Vector3d Jflux=Vector3d::Zero();
  double Jx, Jy, Jz;
  Jx = 0.0;
  Jy = 0.0;
  Jz = 0.0;
#pragma omp parallel for reduction(+: Jx, Jy, Jz)
  for (int ia1=0; ia1<n_atoms; ia1++)
  {
    if (abs(epsil(ia1)) < 0.00000001) continue;
    double ene = energy_lj_peratom(ia1, positions, sigma, epsil, n_atoms, comb_rule, nbl);
    Jx += ene * velocities.col(ia1)(0);
    Jy += ene * velocities.col(ia1)(1);
    Jz += ene * velocities.col(ia1)(2);
  }
  Jflux(0) = Jx;
  Jflux(1) = Jy;
  Jflux(2) = Jz;
  return Jflux;
}


// ptype = 1: combine sigma
// ptype = 2: combine epsil
double combine_pair_coeff(double pi, double pj, int ptype, int comb_rule)
{
  if (comb_rule == 2 or ptype == 2) 
  {
    return sqrt(pi*pj);
  }
  else 
  {
    return (pi + pj) / 2;
  }
}

PYBIND11_MODULE(_nb_pair, m)
{
  m.doc() = "nonbonded pair interaction module";
  m.def("nb_pair", &nb_pair);
  m.def("energy_lennard_jones", &energy_lennard_jones);
}
