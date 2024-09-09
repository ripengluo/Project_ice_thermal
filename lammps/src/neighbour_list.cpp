/*
 * =====================================================================================
 *
 *       Filename:  neighbour_list.cpp
 *
 *    Description:  library that takes care of neighbour list search
 *
 *        Version:  1.0
 *        Created:  12/26/2018 01:59:46 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (Kuang Yu), 
 *   Organization:  TBSI
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

// convert a three-dimensional indices to 1d-index
ArrayXi ind1d_to_ind3d(int n, int na, int nb, int nc)
{
  ArrayXi ind(3);
  ind(2) = n % nc;
  ind(1) = (n / nc) % nb;
  ind(0) = n / (nb*nc);
  return ind;
}

int ind3d_to_ind1d(int i, int j, int k, int na, int nb, int nc)
{
  return k + j*nc + i*nb*nc;
}

// judge if two cells are adjacent or not
bool is_neighbour_cell(ArrayXi ind0, ArrayXi ind1, int na, int nb, int nc)
{
  ArrayXi di = ind1 - ind0;
  di = di.abs();
  if (di(0) > 1 and di(0) < na-1) return false;
  if (di(1) > 1 and di(1) < nb-1) return false;
  if (di(2) > 1 and di(2) < nc-1) return false;
  return true;
}

// this function returns a cell neighbour list
// positions: 3*n_atoms array, storing the positions of the atoms you want to neighbour list
//              Here, positions MUST BE compact.
// atom_labels: n_atoms*1 array, just integers to label the atoms, usually just 0, 1, ... n_atoms-1
//              but can be different if we are constructing a neighbour list for only part of the system
//              for example, if we only want to neighbour list the real atoms, but not the v-sites, then
//              the v-site indices should be missing from atom_labels
//              that is, atom_labels must be compact.
// n_atoms: integer, must be compact
// box, box_inv: box dimensions and its inversion
// r_cut: realspace cutoff distance, in Angstrom
Tensor<int, 4> construct_cell_list(const MatrixXd & positions, const ArrayXi & atom_labels,
    int n_atoms, const MatrixXd & box, const MatrixXd & box_inv, double r_cut)
{
  Tensor<int, 4> celllist;
  double vol = volume(box);
  double ab, bc, ca;
  int na, nb, nc, n_max;
  Vector3d a, b, c;
  a = box.col(0);
  b = box.col(1);
  c = box.col(2);
  ab = (a.cross(b)).norm();
  bc = (b.cross(c)).norm();
  ca = (c.cross(a)).norm();
  // perpendicular heights along each dimension
  double ha, hb, hc;
  ha = vol / bc;
  hb = vol / ca;
  hc = vol / ab;
  // number of cells in each dimension
  na = floor(ha/r_cut);
  nb = floor(hb/r_cut);
  nc = floor(hc/r_cut);
  double density = (double)n_atoms/na/nb/nc;
  // Note here +N is the magic number, we assume the density fluctuation is small enough
  // so +N is large enough buffer.
  n_max = floor(density) + CELL_LIST_BUFFER;
  // create celllist, initialize with -1
  // -1 signals the end of the list
  celllist.resize(n_max, nc, nb, na);
  celllist.setConstant(-1);

  MatrixXd cell=box, cell_inv;
  cell.col(0) /= na;
  cell.col(1) /= nb;
  cell.col(2) /= nc;
  cell_inv = cell.inverse();

  // cell indices of each particle
  MatrixXd spositions = cell_inv * positions;
  MatrixXi indices = spositions.array().floor().cast<int>();
  for (int i_atom=0; i_atom<n_atoms; i_atom++) {
    int label = atom_labels(i_atom);
    int ia = indices(0, i_atom) % na;
    int ib = indices(1, i_atom) % nb;
    int ic = indices(2, i_atom) % nc;
    if (ia < 0) ia += na;
    if (ib < 0) ib += nb;
    if (ic < 0) ic += nc;
    // assign atom to a particular cell
    int i;
    for (i=0; i<n_max; i++) {
      if (celllist(i, ic, ib, ia) < 0) {
        celllist(i, ic, ib, ia) = label;
        break;
      }
    }
    // overflow
    if (i == n_max) {
      cout << "ERROR: cell list overflow, increase CELL_LIST_BUFFER" << endl;
      throw exception();
    }
  }
  // debug
//  cout << "ok2" << endl;
//  cout << celllist.dimension(0) << endl;
//  cout << celllist.dimension(1) << endl;
//  cout << celllist.dimension(2) << endl;
//  cout << celllist.dimension(3) << endl;
//  cout << "--------" << endl;
//  cout << celllist << endl;

  return celllist;
}


// construct a n_atoms*3 array, stores the cell indices in which each particle resides
// Watchout: for a system with virtual sites, n_atoms should be the number of all sites
// including both atoms (which you want to neighbour list), and the vsites (which you do
// not want to neighbour list.
// Hence, the returned cell_index_map will contain -1 for virtual sites, indicating that 
// they are not assigned to a particular cell.
// n_atoms: integer, must be not compact
// return values not compact 
MatrixXi build_cell_index_map(const Tensor<int, 4> & celllist, int n_atoms) 
{
  MatrixXi cell_index_map;
  cell_index_map.resize(3, n_atoms);
  cell_index_map.setConstant(-1);
  int na = celllist.dimension(3);
  int nb = celllist.dimension(2);
  int nc = celllist.dimension(1);
  int nmax = celllist.dimension(0);
  for (int ia=0; ia<na; ia++){
    for (int ib=0; ib<nb; ib++) {
      for (int ic=0; ic<nc; ic++) {
        for (int n=0; n<nmax; n++) {
          if (celllist(n, ic, ib, ia) < 0) break;
          int iatom = celllist(n, ic, ib, ia);
          cell_index_map(0, iatom) = ia;
          cell_index_map(1, iatom) = ib;
          cell_index_map(2, iatom) = ic;
        }
      }
    }
  }
  return cell_index_map;
}

// Here, positions must NOT be compact
// cell_index_map must NOT be compact
// i_atom must NOT be compact
ArrayXi find_neighbours(const MatrixXd & positions, const Tensor<int, 4> & celllist,
    int i_atom, const VectorXi & ind0, double r_cut, const MatrixXd & box, const MatrixXd & box_inv)
{
  // original cell indices
  ArrayXi neighbours;
  neighbours.resize(MAX_N_NEIGHBOURS);
  neighbours.setConstant(-1);
  int n_neighbour = 0;
  int n1 = celllist.dimension(3);
  int n2 = celllist.dimension(2);
  int n3 = celllist.dimension(1);
  int nmax = celllist.dimension(0);
  int ncell = n1*n2*n3;
  double r_cut2 = r_cut*r_cut;
  VectorXd r0 = positions.col(i_atom);
  Map<MatrixXd> pos0(r0.data(), 3, 1);

  for (int icell=0; icell<ncell; icell++) {
    VectorXi ind1 = ind1d_to_ind3d(icell, n1, n2, n3);
    if (not is_neighbour_cell(ind0, ind1, n1, n2, n3)) continue;
    // find number of atoms in the cell1
    int na_cell;
    for (na_cell=0; na_cell<nmax; na_cell++) {
      if (celllist(na_cell, ind1(2), ind1(1), ind1(0)) < 0) break;
    }
    MatrixXd d2;
    if (r_cut>0) {
      MatrixXd pos1;
      pos1.resize(3, na_cell);
      for (int iia=0; iia<na_cell; iia++) {
        pos1.col(iia) = positions.col(celllist(iia, ind1(2), ind1(1), ind1(0)));
      }
      d2 = distance2_pbc_batch(pos0, pos1, box, box_inv, 0);
    }
    for (int iia=0; iia<na_cell; iia++) {
      int i_atom1 = celllist(iia, ind1(2), ind1(1), ind1(0));
      if (i_atom1 == i_atom) continue;
      bool is_nb = false;
      if (r_cut <= 0.0) {
        is_nb = true;
      }
      else if (d2(iia, 0) < r_cut2) {
        is_nb = true;
      }
      if (is_nb) {
        if (n_neighbour == MAX_N_NEIGHBOURS) {
          cout << "ERROR: overflow in neighbour search, increase MAX_N_NEIGHBOURS" << endl;
          throw exception();
        }
        neighbours(n_neighbour) = i_atom1;
        n_neighbour += 1;
      }
    }
  } 

  return neighbours;
}


ArrayXi find_neighbours_for_atom(const MatrixXd & positions, const Tensor<int, 4> & celllist,
    int i_atom0, const MatrixXi & cell_index_map, double r_cut, const MatrixXd & box, const MatrixXd & box_inv)
{
  VectorXi ind0 = cell_index_map.col(i_atom0);
  return find_neighbours(positions, celllist, i_atom0, ind0, r_cut, box, box_inv);
}


// Positions here are not compact
// n_atoms here is not compact
// distances2 is the squared distances between neighbours, it is an in-out argument
NbsRetType find_neighbours_for_all(const MatrixXd & positions, const Tensor<int, 4> & celllist,
    int n_atoms, double r_cut, const MatrixXd & box, const MatrixXd & box_inv)
{
  MatrixXi nb_list, nb_list_tmp;
  MatrixXd distances2;
  nb_list.resize(MAX_N_NEIGHBOURS, n_atoms);
  distances2.resize(MAX_N_NEIGHBOURS, n_atoms);
  nb_list.setConstant(-1);
  nb_list_tmp.resize(MAX_N_NEIGHBOURS, n_atoms);
  nb_list_tmp.setConstant(-1);
  distances2.setConstant(0.0);
  ArrayXi n_nbs;
  n_nbs.resize(n_atoms);
  n_nbs.setConstant(0);
  int n1 = celllist.dimension(3);
  int n2 = celllist.dimension(2);
  int n3 = celllist.dimension(1);
  int ncell = n1*n2*n3;
  int nmax = celllist.dimension(0);
  double r_cut2 = r_cut * r_cut;
  
  // construct the new position matrix, which is contiguous within each cell
  // the new position matrix is in direct coordinates
  vector<MatrixXd> spos_c;
  for (int icell=0; icell<ncell; icell++) {
    VectorXi ind = ind1d_to_ind3d(icell, n1, n2, n3);
    int na_cell;
    // count number of atoms in each cell, for memory allocation
    for (na_cell=0; na_cell<nmax; na_cell++) {
      if (celllist(na_cell, ind(2), ind(1), ind(0)) < 0) break;
    }
    // make the contiguous position matrix in this cell
    MatrixXd spos_cell;
    spos_cell.resize(3, na_cell);
    for (int n=0; n<na_cell; n++) {
      spos_cell.col(n) = box_inv * positions.col(celllist(n, ind(2), ind(1), ind(0)));
    }
    spos_c.push_back(spos_cell);
  }

  // loop over all cell pairs
#pragma omp parallel
#pragma omp for nowait
  for (int icell0=0; icell0<ncell; icell0++) {
    ArrayXi ind0 = ind1d_to_ind3d(icell0, n1, n2, n3);
    int na0 = spos_c[icell0].cols();
    if (na0 == 0) continue;
    for (int icell1=icell0; icell1<ncell; icell1++) {
      ArrayXi ind1 = ind1d_to_ind3d(icell1, n1, n2, n3);
      // two cells are not adjacent
      if (not is_neighbour_cell(ind0, ind1, n1, n2, n3)) continue;
      int na1 = spos_c[icell1].cols();
      // empty cells
      if (na1 == 0) continue;
      // need to judge distance
      MatrixXd d2;
      if (r_cut > 0.0) {
        d2 = distance2_pbc_batch(spos_c[icell0], spos_c[icell1], box, box_inv, 1);
      }
      for (int iia0=0; iia0<na0; iia0++) {
        int i_atom0 = celllist(iia0, ind0(2), ind0(1), ind0(0));
        for (int iia1=0; iia1<na1; iia1++) {
          int i_atom1 = celllist(iia1, ind1(2), ind1(1), ind1(0));
          bool is_nb = false;
          // if it is the same cell, then cut the interacting pair by half
          if (icell0 == icell1 and i_atom0 <= i_atom1) {
            is_nb = false;
          }
          // do not judge distance
          else if (r_cut <= 0.0) {
            is_nb = true;
          }
          // distance within r_cut
          else if (d2(iia1, iia0) < r_cut2) {
            is_nb = true;
          }
          if (is_nb) {
            if (n_nbs(i_atom0) == MAX_N_NEIGHBOURS or n_nbs(i_atom1) == MAX_N_NEIGHBOURS) {
              cout << i_atom0 << endl;
              cout << "---" << endl;
              cout << nb_list_tmp.col(i_atom0) << endl;
              cout << "---" << endl;
              cout << distances2.col(i_atom0) << endl;
              cout << "ERROR: overflow in neighbour search, increase MAX_N_NEIGHBOURS" << endl;
              throw exception();
            }
            // add neighbour to atom0
            nb_list_tmp(n_nbs(i_atom0), i_atom0) = i_atom1;
            distances2(n_nbs(i_atom0), i_atom0) = d2(iia1, iia0);
            n_nbs(i_atom0) += 1;
            // add neighbour to atom1, deal separately in below, due to the writing conflicts in OMP
            // need omp locks here
            //nb_list(n_nbs(i_atom1), i_atom1) = i_atom0;
            //distances2(n_nbs(i_atom1), i_atom1) = d2(iia1, iia0);
            //n_nbs(i_atom1) += 1;
          }
        }
      }
    }
  }

  // add neighbour to atom1
  nb_list = nb_list_tmp;
  for (int i_atom0=0; i_atom0<n_atoms; i_atom0++) {
    for (int iia1=0; iia1<MAX_N_NEIGHBOURS; iia1++) {
      int i_atom1 = nb_list_tmp(iia1, i_atom0);
      if (i_atom1 < 0) break;
      nb_list(n_nbs(i_atom1), i_atom1) = i_atom0;
      distances2(n_nbs(i_atom1), i_atom1) = distances2(iia1, i_atom0);
      n_nbs(i_atom1) += 1;
    }
  }

  NbsRetType results;
  results.n_nbs = n_nbs;
  results.nbs = nb_list;
  results.distances2 = distances2;
  return results;
}


/* ************************************************ */
/* Python wrappers for the neighbour list functions */
/* ************************************************ */

// pybind11 does not recognize the multidimensional Tensor datatype, hence the beaucracy

// convert a 4d tensor to 1d array (assume column based layout)
// T(n1, n2, n3, n4) -> A = {n1, n2, n3, n4, T{1111}, T{2111}, ..., T{n1,n2,n3,n4}}
ArrayXi Tensor_4d_to_1d(const Tensor<int, 4> & t_4d)
{
  ArrayXi results;
  Map<const ArrayXi> data(t_4d.data(), t_4d.size());
  results.resize(t_4d.size()+4);
  // copy the dimension information
  results(0) = t_4d.dimension(0);
  results(1) = t_4d.dimension(1);
  results(2) = t_4d.dimension(2);
  results(3) = t_4d.dimension(3);
  // copy the real data
  results.segment(4, t_4d.size()) = data;
  return results;
}

// convert a 1d array to 4d tensor
Tensor<int, 4> Tensor_1d_to_4d(const ArrayXi & A_1d)
{
  int n1 = A_1d(0);
  int n2 = A_1d(1);
  int n3 = A_1d(2);
  int n4 = A_1d(3);
  TensorMap<Tensor<const int, 4>> T_4d(A_1d.segment(4, A_1d.size()-4).data(), n1, n2, n3, n4);
  return T_4d;
}

ArrayXi construct_cell_list_py(const MatrixXd & positions, const ArrayXi & atom_labels,
    int n_atoms, const MatrixXd & box, const MatrixXd & box_inv, double r_cut)
{
  Tensor<int, 4> celllist;
  celllist = construct_cell_list(positions, atom_labels, n_atoms, box, box_inv, r_cut);
  return Tensor_4d_to_1d(celllist);
}


MatrixXi build_cell_index_map_py(const ArrayXi & celllist_py, int n_atoms) 
{
  Tensor<int, 4> celllist;
  celllist = Tensor_1d_to_4d(celllist_py);
  return build_cell_index_map(celllist, n_atoms);
}


ArrayXi find_neighbours_for_atom_py(const MatrixXd & positions, const ArrayXi & celllist_py,
    int i_atom0, const MatrixXi & cell_index_map, double r_cut, const MatrixXd & box, const MatrixXd & box_inv)
{
  Tensor<int, 4> celllist;
  celllist = Tensor_1d_to_4d(celllist_py);
  return find_neighbours_for_atom(positions, celllist, i_atom0, cell_index_map, r_cut, box, box_inv);
}


NbsRetType find_neighbours_for_all_py(const MatrixXd & positions, const ArrayXi & celllist_py,
    int n_atoms, double r_cut, const MatrixXd & box, const MatrixXd & box_inv)
{
  Tensor<int, 4> celllist;
  celllist = Tensor_1d_to_4d(celllist_py);
  return find_neighbours_for_all(positions, celllist, n_atoms, r_cut, box, box_inv);
}  


PYBIND11_MODULE(_neighbour_list, m)
{
  m.doc() = "Neighbour cell list module";
  m.def("construct_cell_list", &construct_cell_list_py);
  m.def("build_cell_index_map", &build_cell_index_map_py);
  m.def("find_neighbours_for_atom", &find_neighbours_for_atom_py);
  m.def("find_neighbours_for_all", &find_neighbours_for_all_py);
  py::class_<NbsRetType>(m, "NbsRetType")
    .def_readwrite("nbs", &NbsRetType::nbs)
    .def_readwrite("distances2", &NbsRetType::distances2)
    .def_readwrite("n_nbs", &NbsRetType::n_nbs);
}
