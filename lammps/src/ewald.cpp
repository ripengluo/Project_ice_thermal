/*
 * =====================================================================================
 *
 *       Filename:  ewald.cpp
 *
 *    Description:  the module that deals with ewald summation and pme
 *
 *        Version:  1.0
 *        Created:  08/11/2019 05:36:49 PM
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

#pragma omp declare reduction (+: VectorXd: omp_out=omp_out+omp_in)\
               initializer(omp_priv=omp_orig)
#pragma omp declare reduction (+: Vector3d: omp_out=omp_out+omp_in)\
               initializer(omp_priv=omp_orig)

// PME, grid size and charge smears are automatically determined following
// this method used in openmm:
// http://docs.openmm.org/7.0.0/userguide/theory.html
double pme(const MatrixXd & positions, const VectorXd & chgs, int n_atoms, const MatrixXd & box,
    double rcut, double ethresh, const MatrixXi & exclusions, int n_excl)
{
  int K1, K2, K3;
  double kappa;

  // determine the PME parameters
  kappa = sqrt(-log(2*ethresh))/rcut;
  K1 = ceil(2*kappa*box(0,0)/3/pow(ethresh, 0.2));
  K2 = ceil(2*kappa*box(1,1)/3/pow(ethresh, 0.2));
  K3 = ceil(2*kappa*box(2,2)/3/pow(ethresh, 0.2));

  // reciprocal box
  MatrixXd box_inv = box.inverse();
  // volume
  double V = box.determinant();
  // construct neighbor list
  VectorXi atom_labels;
  atom_labels.resize(n_atoms);
  for(int i=0; i<n_atoms; i++) atom_labels(i) = i;
  Tensor<int, 4> celllist = construct_cell_list(positions, atom_labels, n_atoms, box, box_inv, rcut);
  NbsRetType nbl = find_neighbours_for_all(positions, celllist, n_atoms, rcut, box, box_inv);
  // spline functions tables
  func_table Mn = construct_Mn_table(PME_ORDER, INTERVAL_TAB);

  // energy in real space
  double E_pme_real = energy_pme_realspace(positions, chgs, n_atoms, box, box_inv, rcut, kappa, exclusions, n_excl, nbl);

  // construct kpoints and Ck
  MatrixXd kpts = construct_kpoints(box_inv, V, kappa, K1, K2, K3);
  ArrayXd Ck = construct_Ck_energy(kpts, V, kappa, K1, K2, K3);
  MatrixXd Ck_virial = scale_Ck_virial(Ck, kpts, kappa, K1, K2, K3);
  // construct the Q matrix
  VectorXd Qm = construct_Qm_matrix(positions, chgs, box_inv, n_atoms, K1, K2, K3, Mn);
  VectorXcd Qk = dft3d_forward_r2c(Qm, K1, K2, K3); // Note Qk is of shape (K1/2+1, K2, K3)
  // construct the b1b2b3, in half plane
  VectorXd b1b2b3 = construct_b1b2b3_squared(K1, K2, K3, Mn);
  // energy in reciprocal spaces
  double E_pme_recip = energy_pme_reciprocal(kpts, Ck, Qk, b1b2b3, Mn, K1, K2, K3);
  // self energy
  double E_pme_self = energy_pme_self(chgs, n_atoms, kappa);
   
  //
//  cout << "PME real:" << E_pme_real << endl;
//  cout << "PME recip:" << E_pme_recip << endl;
//  cout << "PME self:" << E_pme_self << endl;
  return E_pme_real + E_pme_recip + E_pme_self;
}

/* *************** */
/* Real space code */
/* *************** */
double energy_pme_realspace(const MatrixXd & positions, const VectorXd & chgs, int n_atoms, const MatrixXd & box,
    const MatrixXd & box_inv, double rcut, double kappa, const MatrixXi & exclusions, int n_excl, const NbsRetType & nbl)
{
  // realspace energy
  ArrayXd energies1 = ArrayXd::Zero(n_atoms);
  double energy = 0.0;
#pragma omp parallel
#pragma omp for nowait
  for(int ia1=0; ia1<n_atoms; ia1++)
  {
    if (abs(chgs(ia1))<0.00000001) continue;
    int n_nb = nbl.n_nbs(ia1);
    VectorXd q1 = VectorXd::Constant(n_nb, chgs(ia1));
    VectorXd q2, drdr;
    q2.resize(n_nb);
    drdr.resize(n_nb);
    for(int iia2=0; iia2<n_nb; iia2++)
    {
      int ia2 = nbl.nbs.col(ia1)(iia2);
      q2(iia2) = chgs(ia2);
      drdr(iia2) = nbl.distances2.col(ia1)(iia2);
    }
    ArrayXd dr = drdr.array().sqrt();
    ArrayXd energies2 = 0.5*q1.array()*q2.array()/dr * (dr*kappa).erfc();
    energies1(ia1) = energies2.sum();
//    cout << "------- " << ia1 << endl;
//    cout << " " << dr << endl;
  }
  energy = energies1.sum();
//  cout << "------- " << energy << endl;
//  cout << "------- " << energies1 << endl;

  // deal with exclusion
  MatrixXd pos1, pos2;
  pos1.resize(3, n_excl);
  pos2.resize(3, n_excl);
  VectorXd q1, q2;
  q1.resize(n_excl);
  q2.resize(n_excl);
  // loop over all exclusions, prepare the distances
#pragma omp parallel
#pragma omp for nowait
  for(int iexcl=0; iexcl<n_excl; iexcl++)
  {
    int ia1 = exclusions.col(iexcl)(0);
    int ia2 = exclusions.col(iexcl)(1);
    pos1.col(iexcl) = positions.col(ia1);
    pos2.col(iexcl) = positions.col(ia2);
    q1(iexcl) = chgs(ia1);
    q2(iexcl) = chgs(ia2);
  }
  // pbc shifts
  MatrixXd dr_vec = pos2 - pos1;
  MatrixXd ds_vec = box_inv * dr_vec;
  ds_vec -= (ds_vec.array()+0.5).floor().matrix();
  dr_vec = box * ds_vec;
  VectorXd drdr = dr_vec.colwise().squaredNorm();
  ArrayXd ens_excl = q1.array()*q2.array()/drdr.array().sqrt();
  double en_excl = ens_excl.sum();

  // deduct exclusion energy
//  cout << energy << endl;
//  cout << en_excl << endl;
  return (energy - en_excl)*E2A2KCMOL;
}

  
// pme heat flux in real space
Vector3d heat_flux_convection_pmereal(const MatrixXd & positions, const MatrixXd & qv, 
    const VectorXd & chgs, int n_atoms, const MatrixXd & box, const MatrixXd & box_inv,
    const NbsRetType & nbl, double kappa, const MatrixXi & exclusions, int n_excl)
{
  Vector3d Jflux = Vector3d::Zero();
  MatrixXd Jatom = MatrixXd::Zero(n_atoms, 3);
  // the convection part
#pragma omp parallel
#pragma omp for nowait
  for (int ia1=0; ia1<n_atoms; ia1++)
  {
    if (abs(chgs(ia1))<0.00000001) continue;
    int n_nb = nbl.n_nbs(ia1);
    double qi = chgs(ia1);
    Vector3d qiv = qv.row(ia1);
    for(int iia2=0; iia2<n_nb; iia2++)
    {
      int ia2 = nbl.nbs.col(ia1)(iia2);
      double qj = chgs(ia2);
      if (abs(qj)<0.00000001 or ia2<=ia1) continue;
      Vector3d qjv = qv.row(ia2);
      double drdr = nbl.distances2.col(ia1)(iia2);
      double dr = sqrt(drdr);
      Jatom.row(ia1) += 0.5 * (qi*qjv + qj*qiv) / dr * erfc(dr*kappa);
    }
  }

  Jflux += Jatom.colwise().sum();
  // deal with exclusions
  MatrixXd pos1, pos2;
  pos1.resize(3, n_excl);
  pos2.resize(3, n_excl);
  VectorXd q1, q2;
  MatrixXd qv1, qv2;
  q1.resize(n_excl);
  q2.resize(n_excl);
  qv1.resize(n_excl, 3);
  qv2.resize(n_excl, 3);
#pragma omp parallel
#pragma omp for nowait
  for (int i_excl=0; i_excl<n_excl; i_excl++)
  {
    int ia1 = exclusions.col(i_excl)(0);
    int ia2 = exclusions.col(i_excl)(1);
    q1(i_excl) = chgs(ia1);
    q2(i_excl) = chgs(ia2);
    qv1.row(i_excl) = qv.row(ia1);
    qv2.row(i_excl) = qv.row(ia2);
    pos1.col(i_excl) = positions.col(ia1);
    pos2.col(i_excl) = positions.col(ia2);
  }
  // pbc shifts
  MatrixXd dr_vec = pos2 - pos1;
  MatrixXd ds_vec = box_inv * dr_vec;
  ds_vec -= (ds_vec.array()+0.5).floor().matrix();
  dr_vec = box * ds_vec;
  ArrayXd dr = dr_vec.colwise().squaredNorm().array().sqrt();
  for (int idim=0; idim<3; idim++)
  {
    Jflux(idim) -= (0.5* q1.array() * qv2.col(idim).array() / dr).sum();
    Jflux(idim) -= (0.5* qv1.col(idim).array() * q2.array() / dr).sum();
  }

  return Jflux; 
}

Vector3d heat_flux_virial_pmereal(const MatrixXd & positions, const MatrixXd & qv, 
    const VectorXd & chgs, int n_atoms, const MatrixXd & box, const MatrixXd & box_inv, 
    const NbsRetType & nbl, double kappa, const MatrixXi & exclusions, int n_excl)
{
  Vector3d Jflux=Vector3d::Zero();
  MatrixXd Jflux_atom;
  Jflux_atom.resize(n_atoms, 3);
  Jflux_atom.setConstant(0);
#pragma omp parallel
#pragma omp for nowait
  for (int ia1=0; ia1<n_atoms; ia1++)
  {
    double qi = chgs(ia1);
    if (abs(qi)<0.00000001) continue;
    int n_nb = nbl.n_nbs(ia1);
    for(int iia2=0; iia2<n_nb; iia2++) {
      int ia2 = nbl.nbs.col(ia1)(iia2);
      if (ia2 <= ia1) continue;
      double qj = chgs(ia2);
      if (abs(qj)<0.00000001) continue;
      Vector3d dr_vec = dr_vec_pbc(positions.col(ia1), positions.col(ia2), box, box_inv, 0);
      // distance
      double drdr = nbl.distances2.col(ia1)(iia2);
      double dr = sqrt(drdr);
      double kappa_dr = kappa*dr;
      double factor = erfc(kappa_dr)/drdr/dr + 2*kappa/sqrt(PI)*exp(-kappa_dr*kappa_dr)/drdr;
      Jflux_atom.row(ia1) += 0.5* qi* factor * dr_vec * dr_vec.dot(qv.row(ia2));
      Jflux_atom.row(ia1) += 0.5* qj* factor * dr_vec * dr_vec.dot(qv.row(ia1));
    }
  }
  Jflux = Jflux_atom.colwise().sum();
  // deal with exclusion
  Jflux_atom.resize(n_excl, 3);
  Jflux_atom.setConstant(0);
#pragma omp parallel
#pragma omp for nowait
  for (int i_excl=0; i_excl<n_excl; i_excl++)
  {
    int ia1 = exclusions.col(i_excl)(0);
    int ia2 = exclusions.col(i_excl)(1);
    Vector3d ri = positions.col(ia1);
    Vector3d rj = positions.col(ia2);
    double qi = chgs(ia1);
    double qj = chgs(ia2);
    Vector3d dr_vec = dr_vec_pbc(ri, rj, box, box_inv, 0);
    double drdr = dr_vec.dot(dr_vec);
    double dr = sqrt(drdr);
    Vector3d qvi = qv.row(ia1);
    Vector3d qvj = qv.row(ia2);
    double rqvi = dr_vec.dot(qvi);
    double rqvj = dr_vec.dot(qvj);
    Jflux_atom.row(i_excl) = 0.5* (qi*rqvj + qj*rqvi)/drdr/dr * dr_vec;
  }
  Vector3d Jflux_excl = Jflux_atom.colwise().sum();
  return Jflux - Jflux_excl;
}


// compute qv(i, \alpha) = q_i * v_i^\alpha
MatrixXd compute_qv(const MatrixXd & velocities, const VectorXd & chgs, int n_atoms)
{
  MatrixXd qv;
  qv.resize(n_atoms, 3);
#pragma omp parallel
#pragma omp for nowait
  for (int i_atom=0; i_atom<n_atoms; i_atom++)
  {
    qv.row(i_atom) = chgs(i_atom) * velocities.col(i_atom);
  }
  return qv;
}



/* ********************* */
/* reciprocal space code */
/* ********************* */
// These functions construct the kpoints and C(k) in PME
// the returned results are ordered as: kx, ky, kz, k^2 for all the k-points, looping k1 fastest, and k3 slowest
MatrixXd construct_kpoints(const MatrixXd & box_inv, double V, double kappa, int K1, int K2, int K3)
{
  int K;
  K = K1 * K2 * K3;
  MatrixXd kpts;
  kpts.resize(4, K);
#pragma omp parallel
#pragma omp for nowait
  for(int k3=0; k3<K3; k3++)
  {
    int k3p=0;
    if (k3>0) k3p = K3 - k3;
    for(int k2=0; k2<K2; k2++)
    {
      int k2p=0;
      if (k2>0) k2p = K2 - k2;
      for(int k1=0; k1<K1/2+1; k1++)
      {
        int k1p=0;
        if (k1>0) k1p = K1 - k1;
        int ikpt = ind_3d_to_1d(k1, k2, k3, K1, K2, K3);
        int ikptp = ind_3d_to_1d(k1p, k2p, k3p, K1, K2, K3);
        // gamma center
        int kk1 = k1;
        int kk2 = k2;
        int kk3 = k3;
        if (kk1 > K1/2) kk1 -= K1;
        if (kk2 > K2/2) kk2 -= K2;
        if (kk3 > K3/2) kk3 -= K3;
        // compute \vec{k} and C(k)
        Vector3d kvec = kk1*box_inv.row(0) + kk2*box_inv.row(1) + kk3*box_inv.row(2);
        kvec = kvec * 2.0 * PI;
        double ksq = kvec.squaredNorm();
        kpts(0, ikpt) = kvec(0);
        kpts(1, ikpt) = kvec(1);
        kpts(2, ikpt) = kvec(2);
        kpts(3, ikpt) = ksq;
        kpts(0, ikptp) = -kvec(0);
        kpts(1, ikptp) = -kvec(1);
        kpts(2, ikptp) = -kvec(2);
        kpts(3, ikptp) = ksq;
      }
    }
  }
  return kpts;
}

// C(k) in energy calculation
ArrayXd construct_Ck_energy(const MatrixXd & kpts, double V, double kappa, int K1, int K2, int K3)
{
  int K = K1*K2*K3;
  ArrayXd Ck;
  Ck.resize(K);
#pragma omp parallel
#pragma omp for nowait
  for(int k3=0; k3<K3; k3++)
  {
    int k3p=0;
    if (k3>0) k3p = K3 - k3;
    for(int k2=0; k2<K2; k2++)
    {
      int k2p=0;
      if (k2>0) k2p = K2 - k2;
      for(int k1=0; k1<K1/2+1; k1++)
      {
        int k1p=0;
        if (k1>0) k1p = K1 - k1;
        int ikpt = ind_3d_to_1d(k1, k2, k3, K1, K2, K3);
        double ksq = kpts(3, ikpt);
        if (k1==0 and k2==0 and k3==0) // gamma point
        {
          Ck(ikpt) = 0.0;
          continue;
        }
        Ck(ikpt) = 2*PI/V/ksq * exp(-ksq/4/pow(kappa,2));
        // other half
        int ikptp = ind_3d_to_1d(k1p, k2p, k3p, K1, K2, K3);
        Ck(ikptp) = Ck(ikpt);
      }
    }
  }
  return Ck;
}

// Ck in virial calculation
// in full dimension K1*K2*K3
MatrixXd scale_Ck_virial(const ArrayXd & Ck_energy, const MatrixXd & kpts, double kappa, int K1, int K2, int K3)
{
  int K = K1*K2*K3;
  MatrixXd Ck_virial;
  Ck_virial.resize(9, K);
#pragma omp parallel
#pragma omp for nowait
  for(int k3=0; k3<K3; k3++)
  {
    int k3p=0;
    if (k3>0) k3p = K3 - k3;
    for(int k2=0; k2<K2; k2++)
    {
      int k2p=0;
      if (k2>0) k2p = K2 - k2;
      for(int k1=0; k1<K1/2+1; k1++)
      {
        int k1p=0;
        if (k1>0) k1p = K1 - k1;
        int ikpt = ind_3d_to_1d(k1, k2, k3, K1, K2, K3);
        int ikptp = ind_3d_to_1d(k1p, k2p, k3p, K1, K2, K3);
        double ksq = kpts(3, ikpt);
        if (k1==0 and k2==0 and k3==0)
        {
          for (int idim=0; idim<9; idim++) Ck_virial(idim, ikpt) = 0.0;
          continue;
        }
        for (int alpha=0; alpha<3; alpha++)
        {
          for (int beta=0; beta<3; beta++)
          {
            int idim = alpha + beta*3;
            double tmp = 2 * (1+ksq/4/pow(kappa,2)) / ksq * kpts(alpha, ikpt) * kpts(beta, ikpt);
            if (alpha == beta)
            {
              Ck_virial(idim, ikpt) = Ck_energy(ikpt) * (1 - tmp);
              Ck_virial(idim, ikptp) = Ck_energy(ikptp) * (1 - tmp);
            }
            else
            {
              Ck_virial(idim, ikpt) = Ck_energy(ikpt) * ( -tmp);
              Ck_virial(idim, ikptp) = Ck_energy(ikptp) * ( -tmp);
            }
          }
        }
      }
    }
  }
  return Ck_virial;
}

MatrixXd construct_Ck_virial(const MatrixXd & kpts, double V, double kappa, int K1, int K2, int K3)
{
  ArrayXd Ck_energy = construct_Ck_energy(kpts, V, kappa, K1, K2, K3);
  MatrixXd Ck_virial = scale_Ck_virial(Ck_energy, kpts, kappa, K1, K2, K3);
  return Ck_virial;
}

// construct Q matrix
VectorXd construct_Qm_matrix(const MatrixXd & positions, const VectorXd & chgs, const MatrixXd & box_inv,
    int n_atoms, int K1, int K2, int K3, func_table Mn)
{
  VectorXd Qm = VectorXd::Zero(K1*K2*K3);
  MatrixXd spositions = box_inv * positions;
//#pragma omp parallel
//#pragma omp for nowait
//#pragma omp parallel for reduction(+: Qm)
  for(int i_atom=0; i_atom<n_atoms; i_atom++)
  {
    if (abs(chgs(i_atom))<0.0000001) continue;
    Vector3d s = spositions.col(i_atom);
    Vector3d u;
    Array3i u_floor;
    u(0) = s(0) * K1;
    u(1) = s(1) * K2;
    u(2) = s(2) * K3;
    u_floor = u.array().floor().cast<int>();
    for (int di3=-PME_ORDER+1; di3<=0; di3++)
    {
      int m3 = u_floor(2) + di3;
      double Mn3 = Mn.value(u(2)-u_floor(2)-di3);
      m3 = m3 - m3/K3*K3;
      if (m3<0) m3 += K3;
      for (int di2=-PME_ORDER+1; di2<=0; di2++)
      {
        int m2 = u_floor(1) + di2;
        double Mn2 = Mn.value(u(1)-u_floor(1)-di2);
        m2 = m2 - m2/K2*K2;
        if (m2<0) m2 += K2;
        for (int di1=-PME_ORDER+1; di1<=0; di1++)
        {
          int m1 = u_floor(0) + di1;
          double Mn1 = Mn.value(u(0)-u_floor(0)-di1);
          m1 = m1 - m1/K1*K1;
          if(m1<0) m1 += K1;
          // add the meshed particle
          int ind1d = ind_3d_to_1d(m1, m2, m3, K1, K2, K3);
//#pragma omp critical
          {
            Qm(ind1d) += chgs(i_atom) * Mn1 * Mn2 * Mn3;
          }
        }
      }
    }
  }
  return Qm;
}

// add up pme reciprocal energy with calculated kpts, Ck, and Qk
// nkpt = K1 * K2 * K3
// kpts (4, nkpt): for each k-point, list kx, ky, kz, k^2, loop k1 fastest, full dimension
// Ck (nkpt): Ck at each k-point
// Qk (K1/2+1 * K2 * K3): Qk at each k-point, in only half-plane, folded along K1 direction
double energy_pme_reciprocal(const MatrixXd & kpts, const ArrayXd & Ck, const VectorXcd & Qk, const VectorXd & b1b2b3, 
    func_table Mn, int K1, int K2, int K3)
{
  double energy = 0.0;

#pragma omp parallel for reduction(+: energy)
  for(int k3=0; k3<K3; k3++)
  {
    for(int k2=0; k2<K2; k2++)
    {
      for(int k1=0; k1<K1/2+1; k1++)
      {
        if (k1==0 and k2==0 and k3==0) continue;
        int k = ind_3d_to_1d(k1, k2, k3, K1, K2, K3);
        int kq = ind_3d_to_1d(k1, k2, k3, K1/2+1, K2, K3);
        energy += Ck(k) * b1b2b3(kq) * (pow(real(Qk(kq)), 2) + pow(imag(Qk(kq)), 2));
      }
    }
  }

  return energy * E2A2KCMOL;
}

Vector3d heat_flux_convection_pmerecip(const MatrixXd & Ck, const MatrixXd & Ak, const MatrixXd & b1b2b3, 
    int K1, int K2, int K3)
{
  Vector3d Jflux=Vector3d::Zero();
  for(int alpha=0; alpha<3; alpha++)
  {
    double Jalpha = 0.0;
#pragma omp parallel for reduction(+: Jalpha)
    for(int k3=0; k3<K3; k3++)
    {
      for(int k2=0; k2<K2; k2++)
      {
        for(int k1=0; k1<K1/2+1; k1++)
        {
          if (k1==0 and k2==0 and k3==0) continue;
          int k = ind_3d_to_1d(k1, k2, k3, K1, K2, K3);
          int kq = ind_3d_to_1d(k1, k2, k3, K1/2+1, K2, K3);
          Jalpha += Ck(k) * b1b2b3(kq) * Ak(kq, alpha);
        }
      }
    }
    Jflux(alpha) = Jalpha;
  }
  return Jflux;
}

// NOTE: Ck_virial is of dimension (9, K), and it should be K symmetric 3*3 matrices
// Ak is of dimension (Khalf, 3)
Vector3d heat_flux_virial_pmerecip(const MatrixXd & Ck_virial, const MatrixXd & Ak, const MatrixXd & b1b2b3,
    int K1, int K2, int K3)
{
  Vector3d Jflux=Vector3d::Zero();
  double J0, J1, J2;
#pragma omp parallel for reduction(+: J0, J1, J2)
  for(int k3=0; k3<K3; k3++)
  {
    for(int k2=0; k2<K2; k2++)
    {
      for(int k1=0; k1<K1/2+1; k1++)
      {
        if (k1==0 and k2==0 and k3==0) continue;
        int k = ind_3d_to_1d(k1, k2, k3, K1, K2, K3);
        int kq = ind_3d_to_1d(k1, k2, k3, K1/2+1, K2, K3);
        VectorXd tmp = Ck_virial.col(k);
        Map<MatrixXd> Ck(tmp.data(), 3, 3);
        Vector3d Jcomp = Ck * Ak.row(kq).transpose();
        J0 += Jcomp(0) * b1b2b3(kq);
        J1 += Jcomp(1) * b1b2b3(kq);
        J2 += Jcomp(2) * b1b2b3(kq);
      }
    }
  }
  Jflux(0) = J0;
  Jflux(1) = J1;
  Jflux(2) = J2;
  return Jflux;
}

double energy_pme_self(const VectorXd & chgs, int n_atoms, double kappa)
{
  double energy = chgs.squaredNorm()*kappa/sqrt(PI);
  return -energy * E2A2KCMOL;
}

Vector3d heat_flux_self(const MatrixXd velocities, const VectorXd & chgs, int n_atoms, double kappa)
{
  Vector3d Jflux=Vector3d::Zero();
  for (int i_atom=0; i_atom<n_atoms; i_atom++) Jflux += pow(chgs(i_atom), 2) * velocities.col(i_atom);
  return -kappa/sqrt(PI) * Jflux;
}

VectorXd construct_b1b2b3_squared(int K1, int K2, int K3, func_table Mn)
{
  int Khalf = (K1/2+1)*K2*K3;
  VectorXd b1b2b3_squared;
  b1b2b3_squared.resize(Khalf);
  VectorXd b1, b2, b3;
  b1.resize(K1);
  b2.resize(K2);
  b3.resize(K3);
#pragma omp parallel
#pragma omp for nowait
  for (int k1=0; k1<K1; k1++)
  {
    complex<double> tmp = bfac(k1, K1, PME_ORDER, Mn);
    b1(k1) = pow(real(tmp), 2) + pow(imag(tmp), 2);
  }
#pragma omp parallel
#pragma omp for nowait
  for (int k2=0; k2<K2; k2++)
  {
    complex<double> tmp = bfac(k2, K2, PME_ORDER, Mn);
    b2(k2) = pow(real(tmp), 2) + pow(imag(tmp), 2);
  }
#pragma omp parallel
#pragma omp for nowait
  for (int k3=0; k3<K3; k3++)
  {
    complex<double> tmp = bfac(k3, K3, PME_ORDER, Mn);
    b3(k3) = pow(real(tmp), 2) + pow(imag(tmp), 2);
  }
#pragma omp parallel
#pragma omp for nowait
  for (int k3=0; k3<K3; k3++)
  {
    for(int k2=0; k2<K2; k2++)
    {
      for(int k1=0; k1<K1/2+1; k1++)
      {
        if (k1==0 and k2==0 and k3==0) continue;
        int kq = ind_3d_to_1d(k1, k2, k3, K1/2+1, K2, K3);
        double sfac=2.0;
        if (k1==0 or k1==K1/2) sfac = 1.0;
        b1b2b3_squared(kq) = b1(k1)*b2(k2)*b3(k3) * sfac;
      }
    }
  }
  return b1b2b3_squared;
}

complex<double> bfac(int k, int K, int n, func_table Mn)
{
  complex<double> b(0.0, 0.0);
  complex<double> denominator(0.0, 0.0);
  complex<double> numerator = exp(2*PI*J*(1.0*(n-1)*k/K));
  for (int m=0; m<=n-2; m++)
  {
    denominator += Mn.value(m+1) * exp(2*PI*J*(1.0*m*k/K));
  }
  return numerator / denominator;
}

// Ak is constructed as a real number matrix with dimension K1/2+1, K2, K3
// the resulting matrix is (|F(A1)|^2 - |F(A2)|^2 - |F(A3)|^2)
MatrixXd construct_Ak(const MatrixXd & positions, const VectorXd & chgs, const MatrixXd & qv, int n_atoms, 
    const MatrixXd & box_inv, int K1, int K2, int K3, func_table Mn)
{
  int K = K1*K2*K3;
  int Khalf = (K1/2+1)*K2*K3;
  MatrixXd Ak;
  Ak.resize(Khalf, 3);
  VectorXd Am2;
  Am2 = construct_Qm_matrix(positions, chgs, box_inv, n_atoms, K1, K2, K3, Mn);
  VectorXcd Ak2 = dft3d_forward_r2c(Am2, K1, K2, K3);
  VectorXd Ak2_sq = Ak2.array().abs2();
  for (int idim=0; idim<3; idim++)
  {
    VectorXd Am1, Am3;
    Am3 = construct_Qm_matrix(positions, qv.col(idim), box_inv, n_atoms, K1, K2, K3, Mn);
    VectorXcd Ak3 = dft3d_forward_r2c(Am3, K1, K2, K3);
    VectorXcd Ak1 = Ak3 + Ak2;
    Ak.col(idim) = Ak1.array().abs2() - Ak2_sq.array() - Ak3.array().abs2();
  }
  return Ak/2.0;
}

// python wrapper for Q matrix construction, in real space
VectorXd construct_Qm_matrix_py(const MatrixXd & positions, const VectorXd & chgs, const MatrixXd & box_inv,
    int n_atoms, int K1, int K2, int K3)
{
  func_table Mn = construct_Mn_table(PME_ORDER, INTERVAL_TAB);
  return construct_Qm_matrix(positions, chgs, box_inv, n_atoms, K1, K2, K3, Mn);
}

int ind_3d_to_1d(int k1, int k2, int k3, int K1, int K2, int K3)
{
  return k1 + k2*K1 + k3*K1*K2;
}

Array3i ind_1d_to_3d(int k, int K1, int K2, int K3)
{
  Array3i ind3d;
  ind3d(2) = k / (K1*K2);
  ind3d(1) = (k % (K1*K2)) / K1;
  ind3d(0) = k % K1;
  return ind3d;
}

PYBIND11_MODULE(_ewald, m)
{
  m.doc() = "Ewald summation and PME module";
  m.def("pme", &pme);
  m.def("construct_Qm_matrix", &construct_Qm_matrix_py);
  m.def("dft3d_forward_r2c", &dft3d_forward_r2c);
}
