/*
 * =====================================================================================
 *
 *       Filename:  features_mod.cpp
 *
 *    Description:  module that deals with the feature calculations, see Behler's 
 *                  tutorial review.
 *
 *        Version:  1.0
 *        Created:  01/06/2019 02:26:29 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (Kuang Yu), 
 *   Organization:  
 *
 * =====================================================================================
 */
#include <stdlib.h>
#include <vector>
#include <mylibs.h>
#include <omp.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

using namespace std;
using namespace Eigen;
namespace py = pybind11;

typedef Array<bool,Dynamic,1> ArrayXb;

// return two objects: the \vec{dr} for all neighbors (vector<matrix(3, n_nbs)>)
//                     and the element types for all neighbors (vector<ArrayXi(n_nbs)>)
tuple<vector<MatrixXd>, vector<ArrayXi>> find_drvec_nbs_for_all(const MatrixXd & positions,
   ArrayXi list_elem, int n_atoms, const MatrixXi & nbs, const MatrixXd & box, const MatrixXd & box_inv,
   bool calc_deriv)
{
  vector<MatrixXd> dr_nbs_all;
  vector<ArrayXi> elem_nbs_all;
  for (int i_atom=0; i_atom<n_atoms; i_atom++){
    VectorXd r0 = positions.col(i_atom);
    // count number of atoms
    int n_nbs = (nbs.col(i_atom).array()>=0).count();
    MatrixXd r_nbs(3, n_nbs);
    ArrayXi elem_nbs(n_nbs);
    ArrayXi indices_nbs = nbs.col(i_atom);
    // if calc derivatives, then sort the neightbour list in ascending order 
    // for faster insertion in sparse derivative matrix later on
    // NOTE: the order of dr_nbs_all and elem_nbs_all shall not affect the feature calc, 
    // as long as they are consistent with each other
    if (calc_deriv) {
      sort(indices_nbs.data(), indices_nbs.data()+n_nbs);
    }
    // copy neighbour coordinates and elements
    for (int i_nb=0; i_nb<n_nbs; i_nb++){
      elem_nbs(i_nb) = list_elem(indices_nbs(i_nb));
      r_nbs.col(i_nb) = positions.col(indices_nbs(i_nb)) - r0;
    }
    // pbc shifts
    MatrixXd s_nbs = box_inv * r_nbs;
    s_nbs -= (s_nbs.array()+0.5).floor().matrix();
    r_nbs = box * s_nbs;
    dr_nbs_all.push_back(r_nbs);
    elem_nbs_all.push_back(elem_nbs);
  }
  return make_tuple(dr_nbs_all, elem_nbs_all);
}


// calculate the g2 matrix (n_features_g2, n_atoms)
FeatureRetType calc_features_g2_kernel(
    int n_atoms, const vector<MatrixXd> & dr_nbs_all, const vector<ArrayXi> & elem_nbs_all,
    const MatrixXi & nbs,
    int n_element, const ArrayXi & elements, const ArrayXi & list_elem, double r_cut,
    int n_g2, const ArrayXd & g2_shifts, const ArrayXd & g2_exponents, bool calc_deriv) 
{
  int n_features_g2 = n_g2 * n_element;
  MatrixXd features_g2(n_features_g2, n_atoms);
  // Reserve the derivative matrix, in sparse matrix format
  // dimension is (n_atoms*n_features_g2, n_atoms*3), in column major
  // represents \parital feature / \partial \vec{r}
  // features are arranged first as atom blocks, within which are type(element) blocks,
  // within which are differrent filters
  SparseMatrix<double> dfeatures_g2(n_atoms*n_features_g2, n_atoms*3);
  VectorXi n_deriv(n_atoms*3); // number of nonzero element in the derivative matrix
  if (calc_deriv) {
    for (int i_atom=0; i_atom<n_atoms; i_atom++) {
      // virtual sites
      if (list_elem(i_atom) == 0) {
        n_deriv.segment(i_atom*3, 3) = VectorXi::Zero(3);
        continue;
      }
      int n_nbs = elem_nbs_all[i_atom].size();
      // number of derivative terms
      // each atom position impacts: all its own features, regardless of mask type (n_features)
      //                           + all its neighbours' features with the right element mask (n_filter)
      n_deriv.segment(i_atom*3, 3) = VectorXi::Constant(3, n_features_g2 + n_nbs*n_g2);
    }
    dfeatures_g2.reserve(n_deriv);
  }

#pragma omp parallel
#pragma omp for nowait
  for (int i_atom=0; i_atom<n_atoms; i_atom++) {
    // virtual site, skip
    if (list_elem(i_atom) == 0) {
      features_g2.block(0, i_atom, n_features_g2, 1).setConstant(0);
      continue;
    }
    // find all the distances
    int n_nbs = dr_nbs_all[i_atom].cols();
    ArrayXd dr = dr_nbs_all[i_atom].colwise().norm();
    // indices of neighbours, sorted, prepare for derivative calculation
    ArrayXi indices_nbs = nbs.col(i_atom);
    if( calc_deriv ) {
      sort(indices_nbs.data(), indices_nbs.data()+n_nbs);
    }
    MatrixXd feature_terms(n_nbs, n_g2), dfeature_terms(n_nbs, n_g2);
    // \parital feature / \partial r(center_atom)
    MatrixXd dfeature_terms_dr0;
    if (calc_deriv) dfeature_terms_dr0.resize(n_nbs*3, n_g2);
    // loop over all filters, find out all feature terms
    for (int i_filter=0; i_filter<n_g2; i_filter++) {
      double p[3];
      p[0] = g2_shifts[i_filter];
      p[1] = g2_exponents[i_filter];
      p[2] = r_cut;
      feature_terms.col(i_filter) = feature_term_g2(dr, p);
      // calculate the derivative of feature term to center atom positions
      if (calc_deriv) {
        dfeature_terms.col(i_filter) = dfeature_term_g2(dr, p);
        for (int i_nb=0; i_nb<n_nbs; i_nb++) {
          // \vec{dr}/dr
          dfeature_terms_dr0.block(i_nb*3, i_filter, 3, 1) = - dr_nbs_all[i_atom].col(i_nb)
                                                            / dr(i_nb) * dfeature_terms(i_nb, i_filter);
        }
      }
    }

    // deal with features of neighbours that are before the center atom
    // doing everything in order, make sure we can use the SparseMatrix.insert() method
    int i_nb_deriv=0;
    int itype0;
    // find the itype that corresponsds to elem0
    for (itype0=0; itype0<n_element; itype0++){
      if (elements[itype0] == list_elem(i_atom)) break;
    }
    if (calc_deriv) {
      for (i_nb_deriv=0; i_nb_deriv<n_nbs; i_nb_deriv++) {
        int i_atom1 = indices_nbs(i_nb_deriv);
        if (i_atom1 > i_atom) break;
        for(int i_filter=0; i_filter<n_g2; i_filter++) {
          int i_feature = i_atom1*n_features_g2 + itype0*n_g2 + i_filter;
          for(int idim=0; idim<3; idim++) {
            dfeatures_g2.insert(i_feature, i_atom*3+idim) = dfeature_terms_dr0(i_nb_deriv*3+idim, i_filter);
          }
        }
      }
    }

    // collecting the right feature terms for the corresponding feature
    for (int i_type=0; i_type<n_element; i_type++) {
      // element mask
      ArrayXd mask = (elem_nbs_all[i_atom] == elements[i_type]).cast<double>();
      MatrixXd feature_terms_masked(n_nbs, n_g2);
      MatrixXd dfeature_terms_dr0_masked(n_nbs*3, n_g2);
      for (int i_filter=0; i_filter<n_g2; i_filter++) {
        feature_terms_masked.col(i_filter) = feature_terms.col(i_filter).array() * mask;
        if (calc_deriv) {
          for (int i_nb=0; i_nb<n_nbs; i_nb++) {
            dfeature_terms_dr0_masked.block(i_nb*3, i_filter, 3, 1) = 
              dfeature_terms_dr0.block(i_nb*3, i_filter, 3, 1) * mask(i_nb);
          }
        }
      }
      features_g2.block(i_type*n_g2, i_atom, n_g2, 1) = feature_terms_masked.colwise().sum().transpose();
      if (calc_deriv) {
        // deal with the center atom's features
        for (int i_filter=0; i_filter<n_g2; i_filter++) {
          // rearrange the 3*n_nbs array into a (3, n_nbs) matrix
          Map<MatrixXd> dfeature_terms_dr0_col_block(dfeature_terms_dr0_masked.col(i_filter).data(),
                                                    3, n_nbs);
          // then sum over row, giving a 3d vector, which is ds/d\vec{r0}
          Vector3d dfeature_dr0 = dfeature_terms_dr0_col_block.rowwise().sum();
          int i_feature = i_atom*n_features_g2 + i_type*n_g2 + i_filter;
          for(int idim=0; idim<3; idim++) {
            // add the three xyz components of the derivative
            dfeatures_g2.insert(i_feature, i_atom*3+idim) = dfeature_dr0(idim);
          }
        }
      }
    }

    // now deal with the second half of the neighbours
    if (calc_deriv) {
      for (; i_nb_deriv<n_nbs; i_nb_deriv++) {
        int i_atom1 = indices_nbs(i_nb_deriv);
        for(int i_filter=0; i_filter<n_g2; i_filter++) {
          int i_feature = i_atom1*n_features_g2 + itype0*n_g2 + i_filter;
          for(int idim=0; idim<3; idim++) {
            dfeatures_g2.insert(i_feature, i_atom*3+idim) = dfeature_terms_dr0(i_nb_deriv*3+idim, i_filter);
          }
        }
      }
    }
  }

  FeatureRetType results;
  results.features = features_g2;
  results.dfeatures = dfeatures_g2;
  return results;
}


FeatureRetType calc_features_g4_kernel(
    int n_atoms, const vector<MatrixXd> & dr_nbs_all, const vector<ArrayXi> & elem_nbs_all,
    const MatrixXi & nbs,
    int n_element, const ArrayXi & elements, const ArrayXi & list_elem, double r_cut,
    int n_g4, const ArrayXd & g4_zeta, const ArrayXd & g4_lambda, const ArrayXd & g4_eta, 
    bool calc_deriv) 
{
  int n_pair_types = n_element*(n_element+1)/2;
  int n_features_g4 = n_g4*n_pair_types;
  MatrixXd features_g4(n_features_g4, n_atoms);
  SparseMatrix<double> dfeatures_g4(n_atoms*n_features_g4, n_atoms*3); 

#pragma omp parallel
#pragma omp for nowait
  for (int i_atom=0; i_atom<n_atoms; i_atom++) {
    // virtual site skip
    if (list_elem(i_atom) == 0) {
      features_g4.block(0, i_atom, n_features_g4, 1).setConstant(0);
      continue;
    }
    int n_nbs = dr_nbs_all[i_atom].cols();
    int n_angles = n_nbs*(n_nbs-1)/2;
    MatrixXd rlist1(3, n_angles);
    MatrixXd rlist2(3, n_angles);
    MatrixXd feature_terms(n_angles, n_g4);
    // construct rij and rik list for angle calculation
    int i_angle = 0;
    // nb_pair_types is the tag that marks the angle types: e.g., H-X-H, or O-X-H?
    ArrayXi nb_pair_types(n_angles);
    for (int inb=0; inb<n_nbs; inb++) {
      int elem1 = elem_nbs_all[i_atom](inb);
      for (int jnb=inb+1; jnb<n_nbs; jnb++){
        int elem2 = elem_nbs_all[i_atom](jnb);
        rlist1.col(i_angle) = dr_nbs_all[i_atom].col(inb);
        rlist2.col(i_angle) = dr_nbs_all[i_atom].col(jnb);
        nb_pair_types(i_angle) = elem1*MAX_ELEM + elem2;
        i_angle += 1;
      }
    }
    // loop over all filters, calculate feature terms
    for (int i_filter=0; i_filter<n_g4; i_filter++) {
      double p[4];
      p[0] = g4_zeta[i_filter];
      p[1] = g4_lambda[i_filter];
      p[2] = g4_eta[i_filter];
      p[3] = r_cut;
      feature_terms.col(i_filter) = feature_term_g4(rlist1, rlist2, p);
    }
    // collecting the right feature terms for the cooresponding feature
    int i_type = 0;
    for (int i_type1=0; i_type1<n_element; i_type1++) {
      int elem1 = elements(i_type1);
      for (int i_type2=i_type1; i_type2<n_element; i_type2++) {
        int elem2 = elements(i_type2);
        // note we do not judge order: O-X-H and H-X-O are the same angle type
        ArrayXb mask1 = (nb_pair_types == elem1*MAX_ELEM + elem2);
        ArrayXb mask2 = (nb_pair_types == elem2*MAX_ELEM + elem1);
        ArrayXd mask = (mask1 || mask2).cast<double>();
        MatrixXd feature_terms_masked(n_angles, n_g4);
        for (int i_filter=0; i_filter<n_g4; i_filter++) {
          feature_terms_masked.col(i_filter) = feature_terms.col(i_filter).array() * mask;
        }
        features_g4.block(i_type*n_g4, i_atom, n_g4, 1) = feature_terms_masked.colwise().sum().transpose();
        i_type ++;
      }
    }
  }

  FeatureRetType results;
  results.features = features_g4;
  results.dfeatures = dfeatures_g4;
  return results;
}
  
  
// NOTE: here we assume that we use the same filters for different element pairs
// Reference: IJQC (2015), 115, 1032
// Inputs: 
// positions: dim - 3*n_atoms
//    atom positions in Angstrom
// list_elem: dim - n_atoms
//    list of element of each atom, put zero if it's virtual site, no feature calculation for virtual sites
// elements: dim - n_element
//    how many elements are there in total? the list of all existing elements
// box, box_inv: dim - 3*3
//    the col-based box and inverse box matrix. cartesian/direct transformation: r = box * s; s = box^-1 * r
// r_cut: double
//    the distance cut-off
// nbs: dim - MAX_N_NEIGHBOURS * n_atoms
//    the neighbours' indices for each atom. Indices are used to find coordinates in positions
// shifts: dim - n_filters
//    shifts for the gaussian filters
// exponents: dim - n_filters
//    exponents for the gaussian filters
// n_filters:
//    number of filter functions
// calc_deriv:
//    if calculate derivatives of features w.r.t positions 
FeatureRetType calc_features(
    int n_atoms, const MatrixXd & positions, const ArrayXi & list_elem, // structure definitions
    const MatrixXd & box, const MatrixXd & box_inv, // box definition
    const ArrayXi & elements, int n_element,       // elements definition
    double r_cut, const MatrixXi & nbs,            // neighbour search related
    int n_g2, const ArrayXd & g2_shifts, const ArrayXd & g2_exponents,  // g2 parameters
    int n_g4, const ArrayXd & g4_zeta, const ArrayXd & g4_lambda, const ArrayXd & g4_eta, // g4 parameters
    const bool calc_deriv)                        // if calculate derivative
{
  // how many types of pairs? 
  // note: we distinguish the center atom and the neighbour atom
  int n_pair_types_g2 = n_element;
  int n_features_g2 = n_g2 * n_pair_types_g2;
  // how many types of angle?
  int n_triple_types_g4 = n_element*(n_element+1)/2; 
  int n_features_g4 = n_g4 * n_triple_types_g4;
  // total number of features
  int n_features = n_features_g2 + n_features_g4;
  //MatrixXd features = MatrixXd::Zero(n_features, n_atoms);

  /************************************/
  /*  Do the real feature calculation */
  /************************************/
  // find out all the dr_vec for neighbors
  vector<MatrixXd> dr_nbs_all;
  vector<ArrayXi> elem_nbs_all;
  tie(dr_nbs_all, elem_nbs_all) = find_drvec_nbs_for_all(positions, list_elem, n_atoms, nbs, box, box_inv,
                                                         calc_deriv);

  FeatureRetType tuple_g2, tuple_g4;
  tuple_g2 = calc_features_g2_kernel(n_atoms, dr_nbs_all, elem_nbs_all, nbs,
      n_element, elements, list_elem, r_cut, n_g2, g2_shifts, g2_exponents, calc_deriv);
  
  tuple_g4 = calc_features_g4_kernel(n_atoms, dr_nbs_all,
      elem_nbs_all, nbs, n_element, elements, list_elem, r_cut, n_g4, g4_zeta, g4_lambda, g4_eta, calc_deriv);

  FeatureRetType results;
  results.features = MatrixXd::Zero(n_features, n_atoms);
  results.dfeatures.resize(n_atoms*n_features, n_atoms*3);

  // combine the feature matrix
  results.features.block(0, 0, n_features_g2, n_atoms) = tuple_g2.features;
  results.features.block(n_features_g2, 0, n_features_g4, n_atoms) = tuple_g4.features;

  // combine the dfeature matrix
  ArrayXi sizes(n_atoms*3);
  for (int icol=0; icol<n_atoms*3; icol++) {
    sizes = tuple_g2.dfeatures.col(icol).nonZeros() + tuple_g4.dfeatures.col(icol).nonZeros();
  }
  results.dfeatures.reserve(sizes);
  // filling the full dfeature matrix
  for (int icol=0; icol<n_atoms*3; icol++) {
    // filling the upper half
    for (SparseMatrix<double>::InnerIterator it(tuple_g2.dfeatures, icol); it; ++it) {
      int irow = it.row();
      results.dfeatures.insert(irow, icol) = it.value();
    }
    // filling the lower half
    for (SparseMatrix<double>::InnerIterator it(tuple_g4.dfeatures, icol); it; ++it) {
      int irow = it.row() + n_atoms*n_features_g2;
      results.dfeatures.insert(irow, icol) = it.value();
    }
  }

  return results;
}


// a separate wrapper for g2 calculation only
FeatureRetType calc_features_g2(
    int n_atoms, const MatrixXd & positions, const ArrayXi & list_elem, // structure definitions
    const MatrixXd & box, const MatrixXd & box_inv, // box definition
    const ArrayXi & elements, int n_element,       // elements definition
    double r_cut, const MatrixXi & nbs,            // neighbour search related
    int n_g2, const ArrayXd & g2_shifts, const ArrayXd & g2_exponents,  // g2 parameters
    const bool calc_deriv) 
{
  int n_pair_types_g2 = n_element;
  int n_features = n_g2 * n_pair_types_g2;
  MatrixXd features = MatrixXd::Zero(n_features, n_atoms);
  vector<MatrixXd> dr_nbs_all;
  vector<ArrayXi> elem_nbs_all;
  tie(dr_nbs_all, elem_nbs_all) = find_drvec_nbs_for_all(positions, list_elem, n_atoms, nbs, box, box_inv,
                                                         calc_deriv);

  FeatureRetType results = calc_features_g2_kernel(n_atoms, dr_nbs_all, elem_nbs_all, nbs,
      n_element, elements, list_elem, r_cut, n_g2, g2_shifts, g2_exponents, calc_deriv);
  
  return results;
}


FeatureRetType calc_features_g4(
    int n_atoms, const MatrixXd & positions, const ArrayXi & list_elem, // structure definitions
    const MatrixXd & box, const MatrixXd & box_inv, // box definition
    const ArrayXi & elements, int n_element,       // elements definition
    double r_cut, const MatrixXi & nbs,            // neighbour search related
    int n_g4, const ArrayXd & g4_zeta, const ArrayXd & g4_lambda, const ArrayXd & g4_eta, // g4 parameters
    const bool calc_deriv)
{
  // how many types of angle?
  int n_triple_types_g4 = n_element*(n_element+1)/2; 
  int n_features = n_g4 * n_triple_types_g4;
  // total number of features
  MatrixXd features = MatrixXd::Zero(n_features, n_atoms);
  vector<MatrixXd> dr_nbs_all;
  vector<ArrayXi> elem_nbs_all;
  tie(dr_nbs_all, elem_nbs_all) = find_drvec_nbs_for_all(positions, list_elem, n_atoms, nbs, box, box_inv,
                                                         calc_deriv);

  FeatureRetType results = calc_features_g4_kernel(n_atoms, dr_nbs_all, elem_nbs_all, nbs,
      n_element, elements, list_elem, r_cut, n_g4, g4_zeta, g4_lambda, g4_eta, calc_deriv);
  
  return results;
}


PYBIND11_MODULE(_features_mod, m)
{
  m.doc() = "feature calculation module";
  m.def("calc_features", &calc_features);
  m.def("calc_features_g2", &calc_features_g2);
  m.def("calc_features_g4", &calc_features_g4);
  py::class_<FeatureRetType>(m, "FeatureRetType")
    .def_readwrite("features", &FeatureRetType::features)
    .def_readwrite("dfeatures", &FeatureRetType::dfeatures);
}
