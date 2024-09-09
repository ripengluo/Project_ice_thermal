/*
 * =====================================================================================
 *
 *       Filename:  intra.cpp
 *
 *    Description:  intramolecular terms
 *
 *        Version:  1.0
 *        Created:  08/30/2019 07:55:01 PM
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


/********************************************/
/* source code to deal with bond terms      */
/********************************************/


/********************************************/
/* source code to deal with bond angle terms*/
/********************************************/
double energy_bond(int ia1, int ia2, const MatrixXd & positions, const MatrixXd & box,
    const MatrixXd & box_inv, const ArrayXd & param, int bond_type)
{
  double energy = 0.0;
  Vector3d r1 = positions.col(ia1);
  Vector3d r2 = positions.col(ia2);
  Vector3d dr = dr_vec_pbc(r1, r2, box, box_inv, 0);
  double drnorm = sqrt(dr.dot(dr));
  if (bond_type == 0)
  {
    energy = 0.5 * param(0) * pow(drnorm - param(1), 2);
  }
  else if (bond_type == 1) // the expanded Morse as in qtip4p-f
  {
    double D = param(0);
    double a = param(1);
    double r0 = param(2);  // param order consistent with OpenMM setting
    double a_dr = a * (drnorm - r0);
    energy = D * (pow(a_dr, 2) - pow(a_dr, 3) + 7./12*pow(a_dr, 4));
  }
  else if (bond_type == 2) // the truncated qtip4p-f
  {
    double D = param(0);
    double a = param(1);
    double r0 = param(2);
    double a_dr = a * (drnorm - r0);
    energy = D * pow(a_dr, 2);
  }
  else
  {
    cout << "WARNING: bond type not implemented" << endl;
  }
  return energy;
}

double energy_bonds(const MatrixXd & positions, int n_atoms, const MatrixXi & indices, const ArrayXd & iparams,
    int n_terms, const MatrixXd & params, int n_params, int bond_type, const MatrixXd & box)
{
  double energy = 0.0;
  MatrixXd box_inv = box.inverse();
#pragma omp parallel for reduction(+: energy)
  for (int i_term=0; i_term<n_terms; i_term++)
  {
    int ia1 = indices.col(i_term)(0);
    int ia2 = indices.col(i_term)(1);
    int iparam = iparams(i_term);
    ArrayXd param = params.col(iparam);
    energy += energy_bond(ia1, ia2, positions, box, box_inv, param, bond_type);
  }
  return energy;
}

double energy_angle(int ia1, int ia2, int ia3, const MatrixXd & positions, const MatrixXd & box,
    const MatrixXd & box_inv, const ArrayXd & param, int angle_type)
{
  double energy = 0.0;
  Vector3d r1 = positions.col(ia1);
  Vector3d r2 = positions.col(ia2);
  Vector3d r3 = positions.col(ia3);
  double theta = angle_pbc(r1, r2, r3, box, box_inv, 0);
  if (angle_type == 0) // harmonic, p(0) = k, p(1) = theta0, en = 0.5*k*(theta-theta0)
  {
    energy = 0.5 * param(0) * pow(theta-param(1)*PI/180, 2);
  }
  else
  {
    cout << "WARNING: angle type not implemented" << endl;
  }
  return energy;
}

double energy_angles(const MatrixXd & positions, int n_atoms, const MatrixXi & indices, const ArrayXd & iparams,
    int n_terms, const MatrixXd & params, int n_params, int angle_type, const MatrixXd & box)
{
  double energy = 0.0;
  MatrixXd box_inv = box.inverse();
#pragma omp parallel for reduction(+: energy)
  for (int i_term=0; i_term<n_terms; i_term++)
  {
    int ia1 = indices.col(i_term)(0);
    int ia2 = indices.col(i_term)(1);
    int ia3 = indices.col(i_term)(2);
    int iparam = iparams(i_term);
    ArrayXd param = params.col(iparam);  // looking for parameters
    energy += energy_angle(ia1, ia2, ia3, positions, box, box_inv, param, angle_type);
  }
  return energy;
}


Vector3d heat_flux_convection_bonds(const MatrixXd & positions, const MatrixXd & velocities, int n_atoms, const MatrixXi & indices,
    const ArrayXd & iparams, int n_terms, const MatrixXd & params, int n_params, int bond_type, const MatrixXd & box)
{
  Vector3d Jflux = Vector3d::Zero();
  MatrixXd box_inv = box.inverse();
  double Jx, Jy, Jz;
  Jx = 0.0;
  Jy = 0.0;
  Jz = 0.0;
#pragma omp parallel for reduction(+: Jx, Jy, Jz)
  for (int i_term=0; i_term<n_terms; i_term++)
  {
    int ia1 = indices.col(i_term)(0);
    int ia2 = indices.col(i_term)(1);
    int iparam = iparams(i_term);
    ArrayXd param = params.col(iparam);
    double en = energy_bond(ia1, ia2, positions, box, box_inv, param, bond_type);
    Vector3d vec_v = velocities.col(ia1) + velocities.col(ia2);
    vec_v /= 2;
    Jx += en * vec_v(0);
    Jy += en * vec_v(1);
    Jz += en * vec_v(2);
  }
  Jflux(0) = Jx;
  Jflux(1) = Jy;
  Jflux(2) = Jz;
  return Jflux;
}


Vector3d heat_flux_convection_angles(const MatrixXd & positions, const MatrixXd & velocities, int n_atoms, const MatrixXi & indices,
    const ArrayXd & iparams, int n_terms, const MatrixXd & params, int n_params, int angle_type, const MatrixXd & box)
{
  Vector3d Jflux = Vector3d::Zero();
  MatrixXd box_inv = box.inverse();
  double Jx, Jy, Jz;
  Jx = 0.0;
  Jy = 0.0;
  Jz = 0.0;
#pragma omp parallel for reduction(+: Jx, Jy, Jz)
  for (int i_term=0; i_term<n_terms; i_term++)
  {
    int ia1 = indices.col(i_term)(0);
    int ia2 = indices.col(i_term)(1);
    int ia3 = indices.col(i_term)(2);
    int iparam = iparams(i_term);
    ArrayXd param = params.col(iparam);  // looking for parameters
    double en = energy_angle(ia1, ia2, ia3, positions, box, box_inv, param, angle_type);
    Vector3d vec_v = velocities.col(ia1) + velocities.col(ia2) + velocities.col(ia3);
    vec_v /= 3;
    Jx += en * vec_v(0);
    Jy += en * vec_v(1);
    Jz += en * vec_v(2); 
  }
  Jflux(0) = Jx;
  Jflux(1) = Jy;
  Jflux(2) = Jz;
  return Jflux;
}


Vector3d heat_flux_virial_bonds(const MatrixXd & positions, const MatrixXd & velocities, int n_atoms, const MatrixXi & indices, 
    const ArrayXd & iparams, int n_terms, const MatrixXd & params, int n_params, int bond_type, const MatrixXd & box)
{
  Vector3d Jflux = Vector3d::Zero();
  MatrixXd box_inv = box.inverse();
  double Jx, Jy, Jz;
  Jx = 0.0;
  Jy = 0.0;
  Jz = 0.0;
#pragma omp parallel for reduction(+: Jx, Jy, Jz)
  for (int i_term=0; i_term<n_terms; i_term++)
  {
    int ia1 = indices.col(i_term)(0);
    int ia2 = indices.col(i_term)(1);
    Vector3d r1 = positions.col(ia1);
    Vector3d r2 = positions.col(ia2);
    Vector3d dr = dr_vec_pbc(r1, r2, box, box_inv, 0);
    int iparam = iparams(i_term);
    ArrayXd param = params.col(iparam);  // looking for parameters
    double drnorm = sqrt(dr.dot(dr));
    double du_dr = 0.0;
    if (bond_type == 0)
    {
      du_dr = param(0) * (drnorm - param(1));
    }
    else if (bond_type == 1)  // expanded Morse as in qtip4p/f
    {
      double D = param(0);
      double a = param(1);
      double r0 = param(2);  // param order consistent with OpenMM setting
      double a_dr = a * (drnorm - r0);
      du_dr = D * (2*a*a_dr - 3*a*pow(a_dr, 2) + 7./3*a*pow(a_dr, 3));
    }
    else if (bond_type == 2) // truncated qtip4p/f
    {
      double D = param(0);
      double a = param(1);
      double r0 = param(2);  // param order consistent with OpenMM setting
      double a_dr = a * (drnorm - r0);
      du_dr = D * 2 * a * a_dr;
    }
    else
    {
      cout << "WARNING: bond type not implemented" << endl;
    }
    Vector3d fij = -du_dr * dr / drnorm;
    Vector3d fji = -fij;
    double fji_dot_vi = fji.dot(velocities.col(ia1));
    double fij_dot_vj = fij.dot(velocities.col(ia2));
    Jx += (-fji_dot_vi + fij_dot_vj) * dr(0) / 2;
    Jy += (-fji_dot_vi + fij_dot_vj) * dr(1) / 2;
    Jz += (-fji_dot_vi + fij_dot_vj) * dr(2) / 2;
    // debug by Kuang
//    if (i_term==446)
//    {
//      Vector3d vi = velocities.col(ia1);
//      Vector3d vj = velocities.col(ia2);
//      Vector3d j = (-fji_dot_vi + fij_dot_vj)*dr;
//      cout << "--------------------" << endl;
//      cout << "ia1: " << ia1 << endl;
//      cout << "ia2: " << ia2 << endl;
//      cout << "fij: " << fij(0) << " " << fij(1) << " " << fij(2) << endl;
//      cout << "vi: " << vi(0) << " " << vi(1) << " " << vi(2) << endl;
//      cout << "vj: " << vj(0) << " " << vj(1) << " " << vj(2) << endl;
//      cout << "rij: " << dr(0) << " " << dr(1) << " " << dr(2) << endl;
//      cout << "j: " << j(0) << " " << j(1) << " " << j(2) << endl;
//    }
  }
  Jflux(0) = Jx;
  Jflux(1) = Jy;
  Jflux(2) = Jz;
  return Jflux;
}


MatrixXd heat_flux_virial_bonds_debug(const MatrixXd & positions, const MatrixXd & velocities, int n_atoms, 
    const MatrixXi & indices, const ArrayXd & iparams, int n_terms, const MatrixXd & params, int n_params, int bond_type, const MatrixXd & box)
{
  MatrixXd box_inv = box.inverse();
  double Jx, Jy, Jz;
  Jx = 0.0;
  Jy = 0.0;
  Jz = 0.0;
  MatrixXd jflux;
  jflux.resize(3, n_terms);
  jflux.setConstant(0.0);
#pragma omp parallel for
  for (int i_term=0; i_term<n_terms; i_term++)
  {
    int ia1 = indices.col(i_term)(0);
    int ia2 = indices.col(i_term)(1);
    Vector3d r1 = positions.col(ia1);
    Vector3d r2 = positions.col(ia2);
    Vector3d dr = dr_vec_pbc(r1, r2, box, box_inv, 0);
    int iparam = iparams(i_term);
    ArrayXd param = params.col(iparam);  // looking for parameters
    double drnorm = sqrt(dr.dot(dr));
    double du_dr = 0.0;
    if (bond_type == 0)
    {
      du_dr = param(0) * (drnorm - param(1));
    }
    else if (bond_type == 1)  // expanded Morse as in qtip4p/f
    {
      double D = param(0);
      double a = param(1);
      double r0 = param(2);  // param order consistent with OpenMM setting
      double a_dr = a * (drnorm - r0);
      du_dr = D * (2*a*a_dr - 3*a*pow(a_dr, 2) + 7./3*a*pow(a_dr, 3));
    }
    else
    {
      cout << "WARNING: bond type not implemented" << endl;
    }
    Vector3d fij = -du_dr * dr / drnorm;
    Vector3d fji = -fij;
    double fji_dot_vi = fji.dot(velocities.col(ia1));
    double fij_dot_vj = fij.dot(velocities.col(ia2));
//    jflux.col(i_term) = (-fji_dot_vi + fij_dot_vj) * dr / 2;
//    jflux.col(i_term) = fij;
    jflux(0, i_term) = (-fji_dot_vi + fij_dot_vj);
  }
  return jflux;
}



Vector3d heat_flux_virial_angles(const MatrixXd & positions, const MatrixXd & velocities, int n_atoms, const MatrixXi & indices, 
    const ArrayXd & iparams, int n_terms, const MatrixXd & params, int n_params, int angle_type, const MatrixXd & box)
{
  Vector3d Jflux = Vector3d::Zero();
  MatrixXd box_inv = box.inverse();
  double Jx, Jy, Jz;
  Jx = 0.0;
  Jy = 0.0;
  Jz = 0.0;
#pragma omp parallel for reduction(+: Jx, Jy, Jz)
  for (int i_term=0; i_term<n_terms; i_term++)
  {
    int ia1 = indices.col(i_term)(0);
    int ia2 = indices.col(i_term)(1);
    int ia3 = indices.col(i_term)(2);
    Vector3d r1 = positions.col(ia1);
    Vector3d r2 = positions.col(ia2);
    Vector3d r3 = positions.col(ia3);
    double theta = angle_pbc(r1, r2, r3, box, box_inv, 0);
    int iparam = iparams(i_term);
    ArrayXd param = params.col(iparam);  // looking for parameters
    MatrixXd dtheta_dr = dangle_pbc(r1, r2, r3, box, box_inv);
    double du_dtheta = 0.0;
    if (angle_type == 0)
    {
      du_dtheta = param(0) * (theta-param(1)*PI/180);
    }
    // insert new angle terms here
    else
    {
      cout << "WARNING: angle type not implemented" << endl;
    }

    /*  ----------------------------------------------------------------------------------- */
    /* New implementation                                                                   */ 
    /* See paper: https://doi.org/10.1021/acs.jctc.9b00252                                  */
    /*  ----------------------------------------------------------------------------------- */
    for (int iia=0; iia<3; iia++)
    {
      int ia = indices.col(i_term)(iia);
      double fdotv = du_dtheta * dtheta_dr.col(iia).dot(velocities.col(ia));
      Vector3d vec_r = Vector3d::Zero();
      for (int iib=0; iib<3; iib++)
      {
        int ib = indices.col(i_term)(iib);
        vec_r += dr_vec_pbc(positions.col(ia), positions.col(ib), box, box_inv, 0);
      }
      vec_r /= 3;
      Jx += fdotv * vec_r(0);
      Jy += fdotv * vec_r(1);
      Jz += fdotv * vec_r(2);
    }
    
    /*  ----------------------------------------------------------------------------------- */
    /* LAMMPS implementation, it is problematic, only for testing and comparing purposes    */ 
    /* See paper: https://doi.org/10.1021/acs.jctc.9b00252                                  */
    /*  ----------------------------------------------------------------------------------- */
    // lammps implementation
//    Vector3d vtot = velocities.col(ia1) + velocities.col(ia2) + velocities.col(ia3);
//    vtot /= 3;
//    Vector3d r21 = dr_vec_pbc(r2, r1, box, box_inv, 0);
//    Vector3d r23 = dr_vec_pbc(r2, r3, box, box_inv, 0);
//    Jx += -du_dtheta * (dtheta_dr.col(0).dot(vtot)*r21(0) + dtheta_dr.col(2).dot(vtot)*r23(0));
//    Jy += -du_dtheta * (dtheta_dr.col(0).dot(vtot)*r21(1) + dtheta_dr.col(2).dot(vtot)*r23(1));
//    Jz += -du_dtheta * (dtheta_dr.col(0).dot(vtot)*r21(2) + dtheta_dr.col(2).dot(vtot)*r23(2));
    /*  ----------------------------------------------------------------------------------- */


  }
  Jflux(0) = Jx;
  Jflux(1) = Jy;
  Jflux(2) = Jz;
  return Jflux;
}

PYBIND11_MODULE(_intra, m)
{
  m.doc() = "intramolecular interaction module";
  m.def("energy_angles", &energy_angles);
  m.def("energy_bonds", &energy_bonds);
  m.def("heat_flux_virial_bonds", &heat_flux_virial_bonds);
  m.def("heat_flux_convection_bonds", &heat_flux_convection_bonds);
  m.def("heat_flux_virial_angles", &heat_flux_virial_angles);
  m.def("heat_flux_convection_angles", &heat_flux_convection_angles);
  m.def("heat_flux_virial_bonds_debug", &heat_flux_virial_bonds_debug);
}
