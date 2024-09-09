/*
 * =====================================================================================
 *
 *       Filename:  file_io.cpp
 *
 *    Description:  lib that takes care of file i/o
 *
 *        Version:  1.0
 *        Created:  12/26/2018 02:19:38 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (Kuang Yu), 
 *   Organization:  TBSI
 *
 * =====================================================================================
 */

#include <iostream>
#include <fstream>
#include <string>
#include <mylibs.h>
using namespace std;

void read_dmatrix2d_from_file(string ifn, int * size, Eigen::MatrixXd & matrix)
{
  string line;
  ifstream ifile;
  ifile.open(ifn, ios::in);
  if (ifile.is_open())
  {
    getline(ifile, line);
    stringstream stream(line);
    stream >> size[0] >> size[1];
    matrix.resize(size[0], size[1]);
    for (int i=0; i<size[0]; i++)
    {
      getline(ifile, line);
      stringstream stream(line);
      for (int j=0; j<size[0]; j++)
      {
        stream >> matrix(i, j);
      }
    }
  }
  else
  {
    cout << "ERROR: can't open file: " << ifn << endl;
  }
  ifile.close(); 
  //cout << "inside read:" << endl;
  //cout << matrix << endl;
  //cout << "-----" << endl;
}

