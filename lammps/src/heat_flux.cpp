/*
 * =====================================================================================
 *
 *       Filename:  heat_flux.cpp
 *
 *    Description:  Function wrappers to compute the heat flux operator, Note the detailed
 *                  implementations are located in the corresponding module.
 *
 *        Version:  1.0
 *        Created:  08/21/2019 02:14:48 PM
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

Vector3d compute_heat_flux_pot(const MatrixXd & positions, const MatrixXd & velocities,
    const VectorXd & chgs, const VectorXd & sigma, const VectorXd & epsilon, int comb_rule,
    int n_atoms, const MatrixXd & box, const MatrixXi & exclusions, int n_excl,
    double rcut, double ethresh)
{
  /* ***************************************** */
  /* Preparation for the heat flux calculation */
  /* ***************************************** */
  // PME parameters, determined the same way as OpenMM
  int K1, K2, K3;
  double kappa;
  kappa = sqrt(-log(2*ethresh))/rcut;
  K1 = ceil(2*kappa*box(0,0)/3/pow(ethresh, 0.2));
  K2 = ceil(2*kappa*box(1,1)/3/pow(ethresh, 0.2));
  K3 = ceil(2*kappa*box(2,2)/3/pow(ethresh, 0.2));
  // reciprocal box
  MatrixXd box_inv = box.inverse();
  // volume
  double V = box.determinant();
  // neighbour list
  VectorXi atom_labels;
  atom_labels.resize(n_atoms);
  for(int i=0; i<n_atoms; i++) atom_labels(i) = i;
  Tensor<int, 4> celllist = construct_cell_list(positions, atom_labels, n_atoms, box, box_inv, rcut);
  NbsRetType nbl = find_neighbours_for_all(positions, celllist, n_atoms, rcut, box, box_inv);
  // spline function table
  func_table Mn = construct_Mn_table(PME_ORDER, INTERVAL_TAB);
  // construct qi * v_alpha
  MatrixXd qv = compute_qv(velocities, chgs, n_atoms);
  // build kpoints
  MatrixXd kpts = construct_kpoints(box_inv, V, kappa, K1, K2, K3);
  // construct Cv
  ArrayXd Ck = construct_Ck_energy(kpts, V, kappa, K1, K2, K3);
  // construct Cv_virial
  MatrixXd Ck_virial = scale_Ck_virial(Ck, kpts, kappa, K1, K2, K3);
  // b1b2b3 matrix
  VectorXd b1b2b3 = construct_b1b2b3_squared(K1, K2, K3, Mn);
  // construct Ak
  MatrixXd Ak = construct_Ak(positions, chgs, qv, n_atoms, box_inv, K1, K2, K3, Mn);

  /* ************************************** */
  /* Heat flux operator calculation         */
  /* ************************************** */
  // electrostatic part with PME
  Vector3d J_convection_pmereal = heat_flux_convection_pmereal(positions, qv, chgs, n_atoms, 
      box, box_inv, nbl, kappa, exclusions, n_excl);
  
  Vector3d J_convection_pmerecip = heat_flux_convection_pmerecip(Ck, Ak, b1b2b3, K1, K2, K3);

  Vector3d J_virial_pmereal = heat_flux_virial_pmereal(positions, qv, chgs, n_atoms,
      box, box_inv, nbl, kappa, exclusions, n_excl);

  Vector3d J_virial_pmerecip = heat_flux_virial_pmerecip(Ck_virial, Ak, b1b2b3, K1, K2, K3);

  Vector3d J_self = heat_flux_self(velocities, chgs, n_atoms, kappa);

  Vector3d Jflux_es = J_convection_pmereal + J_convection_pmerecip + J_virial_pmereal + J_virial_pmerecip + J_self; 

  Jflux_es *= E2A2KCMOL ;  // report in real unit

  // nonbonding pair part
  Vector3d Jflux_virial_lj = heat_flux_virial_lennard_jones(positions, velocities, n_atoms,
      box, box_inv, sigma, epsilon, comb_rule, nbl, exclusions, n_excl); // unit is in accordance with the unit of epsil and v.

  Vector3d Jflux_convection_lj = heat_flux_convection_lennard_jones(positions, velocities, n_atoms, box, box_inv, 
      sigma, epsilon, comb_rule, nbl, exclusions, n_excl);

  Vector3d Jflux_lj = Jflux_convection_lj + Jflux_virial_lj;

  // total, in real unit 
  Vector3d Jflux = Jflux_es + Jflux_lj;

  return Jflux;
  // debug
//  return Vector3d::Zero();
}

Vector3d compute_heat_flux_kin(const ArrayXd & masses, const MatrixXd & velocities, int n_atoms)
{
  Vector3d Jflux=Vector3d::Zero();
  for(int i_atom=0; i_atom<n_atoms; i_atom++)
  {
    double K = 0.5*masses(i_atom)*velocities.col(i_atom).dot(velocities.col(i_atom));
    Jflux += K * velocities.col(i_atom);
  }
  // return in real unit
  return Jflux * 1.66054e-27*1e15 / EN_REAL2SI / V_REAL2SI;
}

/* ************************************************ */
/* heat flux correction due to virtual site effects */
/* ************************************************ */
// NOTE this code only supports virtual sites using linear combination coefficients
// the default unit is the REAL unit of LAMMPS
// format for ind_atom_virts:
// n_associated_atoms * n_virts: indices of atoms that are used to define the virtual
// format for coeffs_virts:
// n_associated_atoms * n_virts: linear combination coefficients
Vector3d compute_heat_flux_vcorr(const MatrixXd & positions, const MatrixXd & velocities, const MatrixXd & forces,
    int n_sites, const ArrayXd & indices_virts, const MatrixXi & ind_atom_virts, const MatrixXd & coeffs_virts, int n_virts,
    const MatrixXd & box)
{
  double Jx = 0.0;
  double Jy = 0.0;
  double Jz = 0.0;
  MatrixXd box_inv = box.inverse();
  for(int ii_v=0; ii_v<n_virts; ii_v++)
  {
    int iv = indices_virts(ii_v);
    Vector3d rM = positions.col(iv);
    Vector3d fM = forces.col(iv);  // forces applied on the virtual sites
    ArrayXi ind_atoms = ind_atom_virts.col(ii_v);
    ArrayXd ca_atoms = coeffs_virts.col(ii_v);
    for (int iia=0; iia<ca_atoms.size(); iia++)
    {
      int ia = ind_atoms(iia);
      if (ia < 0) break; // end of the associated atoms
      double ca = ca_atoms(iia);
      Vector3d va = velocities.col(ia);
      Vector3d ra = positions.col(ia);
      Vector3d rMa = dr_vec_pbc(rM, ra, box, box_inv, 0);
      double fdotv = fM.dot(va);
      Jx += ca * fdotv * rMa(0);
      Jy += ca * fdotv * rMa(1);
      Jz += ca * fdotv * rMa(2);
    }
  }
  Vector3d Jcorr;
  Jcorr(0) = Jx;
  Jcorr(1) = Jy;
  Jcorr(2) = Jz;
  return Jcorr;
}

PYBIND11_MODULE(_heat_flux, m)
{
  m.doc() = "Heat flux related module, compute heat flux given the positions and velocities.";
  m.def("compute_heat_flux_pot", &compute_heat_flux_pot);
  m.def("compute_heat_flux_kin", &compute_heat_flux_kin);
  m.def("compute_heat_flux_vcorr", &compute_heat_flux_vcorr);
}
