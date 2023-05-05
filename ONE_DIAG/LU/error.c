#include <stdio.h>
#include <math.h>
#include "applu.incl"

//---------------------------------------------------------------------
// 
// compute the solution error
// 
//---------------------------------------------------------------------
void error()
{
  //---------------------------------------------------------------------
  // local variables
  //---------------------------------------------------------------------
  int i, j, k, m;
  double tmp;
  double u000ijk[5];
  double errnm_local[5];
  for (m = 0; m < 5; m++) {
    errnm[m] = 0.0;
  }

  //#pragma omp parallel default(shared) private(i,j,k,m,tmp,u000ijk,errnm_local)
  //#pragma acc parallel private(i,j,k,m,tmp,u000ijk,errnm_local)
  {
  for (m = 0; m < 5; m++) {
    errnm_local[m] = 0.0;
  }
  //#pragma omp for nowait
  //#pragma acc loop
  for (k = 1; k < nz-1; k++) {
    for (j = jst; j < jend; j++) {
      for (i = ist; i < iend; i++) {
        exact( i, j, k, u000ijk );
        for (m = 0; m < 5; m++) {
          tmp = ( u000ijk[m] - u[k][j][i][m] );
          errnm_local[m] = errnm_local[m] + tmp * tmp;
        }
      }
    }
  }
  for (m = 0; m < 5; m++) {
    //#pragma omp atomic
    //#pragma acc atomic
    errnm[m] += errnm_local[m];
  }
  } //end parallel

  for (m = 0; m < 5; m++) {
    errnm[m] = sqrt ( errnm[m] / ( (nx0-2)*(ny0-2)*(nz0-2) ) );
  }

  /*
  printf(" \n RMS-norm of error in soln. to first pde  = %12.5E\n"
         " RMS-norm of error in soln. to second pde = %12.5E\n"
         " RMS-norm of error in soln. to third pde  = %12.5E\n"
         " RMS-norm of error in soln. to fourth pde = %12.5E\n"
         " RMS-norm of error in soln. to fifth pde  = %12.5E\n",
         errnm[0], errnm[1], errnm[2], errnm[3], errnm[4]);
  */
}

