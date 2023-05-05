#include "applu.incl"

//---------------------------------------------------------------------
//
//   compute the exact solution at (i,j,k)
//
//---------------------------------------------------------------------
void exact(int i, int j, int k, double u000ijk[])
{
  //---------------------------------------------------------------------
  // local variables
  //---------------------------------------------------------------------
  int m;
  double xi, eta, zeta;

  xi   = ( (double)i ) / ( nx0 - 1 );
  eta  = ( (double)j ) / ( ny0 - 1 );
  zeta = ( (double)k ) / ( nz - 1 );

  for (m = 0; m < 5; m++) {
    u000ijk[m] =  ce[m][0]
      + (ce[m][1]
      + (ce[m][4]
      + (ce[m][7]
      +  ce[m][10] * xi) * xi) * xi) * xi
      + (ce[m][2]
      + (ce[m][5]
      + (ce[m][8]
      +  ce[m][11] * eta) * eta) * eta) * eta
      + (ce[m][3]
      + (ce[m][6]
      + (ce[m][9]
      +  ce[m][12] * zeta) * zeta) * zeta) * zeta;
  }
}

