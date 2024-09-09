#include <iostream>
#include "mylibs.h"
using namespace std;
using namespace Eigen;

int main(int argc, char *argv[]) 
{
  int nat;
  int dim[2];

  // dynamics matrix
  Eigen::MatrixXd d, box, box_o, box_inv;
  // Fixed size matrix

  d = Eigen::MatrixXd::Random(3, 10);
  cout << d << endl;

  // test read_dmatrix2
  read_dmatrix2d_from_file(argv[1], dim, box_o);
  // transpose into column-major layout
  box.resize(3, 3);
  box = box_o.transpose();
  box_inv = box.inverse();
  
//  cout << "------" << endl;
//  cout << box << endl;
//  cout << "------" << endl;

  // test read_vector
  Eigen::VectorXd atypes;
  read_vector_from_file<Eigen::VectorXd>(argv[2], &nat, atypes);
//  cout << atypes/2 << endl;

  // test distance calc with periodic boundary condition
  Eigen::VectorXd r1, r2, r3, r4, s1, s2;
  s1 = Eigen::VectorXd::Random(3);
  s2 = Eigen::VectorXd::Random(3);
  for (int d=0; d<3; d++) s1(d) -= floor(s1(d));
  for (int d=0; d<3; d++) s2(d) -= floor(s2(d));
  r1 = box * s1;
  r2 = box * s2;
  cout << r1(0) << ", " << r1(1) << ", " << r1(2) << endl;
  cout << r2(0) << ", " << r2(1) << ", " << r2(2) << endl;
  cout << "-----" << endl;
  cout << s1(0) << ", " << s1(1) << ", " << s1(2) << endl;
  cout << s2(0) << ", " << s2(1) << ", " << s2(2) << endl;
  cout << "-----" << endl;
  cout << distance_pbc(r1, r2, box, box_inv, 0) << endl;
  cout << distance_pbc(s1, s2, box, box_inv, 1) << endl;
  cout << "----------" << endl;

  // test angle & dihedral calc
  r1 = d.col(0);
  r2 = d.col(1);
  r3 = d.col(2);
  r4 = d.col(3);
  cout << angle_pbc(r1, r2, r3, box, box_inv, 0)/PI*180. << endl;
  cout << dihedral_pbc(r1, r2, r3, r4, box, box_inv, 0)/PI*180. << endl;


  // test matrix multiplication
//  Eigen::MatrixXd M1, M2, M;
//  M1 = Eigen::MatrixXd::Random(1000, 100000);
//  M2 = Eigen::MatrixXd::Random(100000, 1000);
//  M = M1*M2;
//

  // test neighbour list
  int n_atoms = 4000;
  double r_cut = 4.0;
  ArrayXi atom_labels = ArrayXi::LinSpaced(n_atoms, 0, n_atoms-1);
  MatrixXd positions;
  positions = MatrixXd::Random(3, n_atoms);
  //positions = (positions.array()+1.0)/2;
  positions = box * positions;
  Tensor<int, 4> cell_list = construct_cell_list(positions, atom_labels, n_atoms, box, box_inv, r_cut);
  NbsRetType nbs_obj;
  for (int i=0; i<1; i++) {
    nbs_obj = find_neighbours_for_all(positions, cell_list, n_atoms, r_cut, box, box_inv);
  }
  MatrixXi nbs = nbs_obj.nbs;
  MatrixXd distances2 = nbs_obj.distances2;
  cout << nbs.col(10).transpose() << endl;
  cout << distances2.col(10).transpose() << endl;

  // test tabulation
  double Rc = 4.0;
  int N = 1000;
  double p[3] = {3.0, 1.2, 4.0};
  VectorXd xlist = VectorXd::LinSpaced(N, 0.0, Rc);
  VectorXd g2, dg2, fc, dfc;
  // table read
  func_table table_fc(0.0, 4.0, MAX_N_TABLE, &fc_tanh, p+2);
  func_table table_dfc(0.0, 4.0, MAX_N_TABLE, &dfc_tanh, p+2);
  func_table table_g2(0.0, 4.0, MAX_N_TABLE, &feature_term_g2, p);
  func_table table_dg2(0.0, 4.0, MAX_N_TABLE, &dfeature_term_g2, p);
  g2 = table_g2.value(xlist);
  dg2 = table_dg2.value(xlist);
  // direct calculate
//  g2 = fc_tanh(xlist, &Rc);
//  dg2 = dfc_tanh(xlist, &Rc);
  for (int i=0; i<N; i++){
    cout << xlist(i) << " " << g2(i) << " " << dg2(i) << endl;
  }
}
