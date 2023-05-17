#include "applu.incl"

//---------------------------------------------------------------------
// set the boundary values of dependent variables
//---------------------------------------------------------------------
void setbv()
{
  //---------------------------------------------------------------------
  // local variables
  //---------------------------------------------------------------------
  int i, j, k, m;
  double temp1[5], temp2[5];

  //---------------------------------------------------------------------
  // set the dependent variable values along the top and bottom faces
  //---------------------------------------------------------------------
  //#pragma omp parallel default(shared) private(i,j,k,m,temp1,temp2) \
                                       shared(nx,ny,nz)
  {
  //#pragma omp for schedule(static)
  //#pragma acc parallel loop private(i,j,m,temp1,temp2)
  for (j = 0; j < ny; j++) {
    for (i = 0; i < nx; i++) {
      exact( i, j, 0, temp1 );
      exact( i, j, nz-1, temp2 );
      for (m = 0; m < 5; m++) {
        u[0][j][i][m] = temp1[m];
        u[nz-1][j][i][m] = temp2[m];
      }
    }
  }

  //---------------------------------------------------------------------
  // set the dependent variable values along north and south faces
  //---------------------------------------------------------------------
  //#pragma omp for schedule(static) nowait
  //#pragma acc parallel loop private(i,k,m,temp1,temp2)
  for (k = 0; k < nz; k++) {
    for (i = 0; i < nx; i++) {
      exact( i, 0, k, temp1 );
      exact( i, ny-1, k, temp2 );
      for (m = 0; m < 5; m++) {
        u[k][0][i][m] = temp1[m];
        u[k][ny-1][i][m] = temp2[m];
      }
    }
  }

  //---------------------------------------------------------------------
  // set the dependent variable values along east and west faces
  //---------------------------------------------------------------------
  //#pragma omp for schedule(static) nowait
  //#pragma acc parallel loop private(j,k,m,temp1,temp2)
  for (k = 0; k < nz; k++) {
    for (j = 0; j < ny; j++) {
      exact( 0, j, k, temp1 );
      exact( nx-1, j, k, temp2 );
      for (m = 0; m < 5; m++) {
        u[k][j][0][m] = temp1[m];
        u[k][j][nx-1][m] = temp2[m];
      }
    }
  }
  } //end parallel
}

