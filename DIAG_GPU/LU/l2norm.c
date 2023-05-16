#include <math.h>
#include "applu.incl"

//---------------------------------------------------------------------
// to compute the l2-norm of vector v.
//---------------------------------------------------------------------
//---------------------------------------------------------------------
// To improve cache performance, second two dimensions padded by 1 
// for even number sizes only.  Only needed in v.
//---------------------------------------------------------------------
void l2norm (int ldx, int ldy, int ldz, int nx0, int ny0, int nz0,
     int ist, int iend, int jst, int jend,
     double v[][ldy/2*2+1][ldx/2*2+1][5], double sum[5])
{
  //---------------------------------------------------------------------
  // local variables
  //---------------------------------------------------------------------
  double sum_local[5];
  int i, j, k, m;

  for (m = 0; m < 5; m++) {
    sum[m] = 0.0;
  }

  //#pragma omp parallel default(shared) private(i,j,k,m,sum_local)
  {
  //#pragma acc parallel loop private(m, sum_local)
  for (m = 0; m < 5; m++) {
    sum_local[m] = 0.0;
  }
  //#pragma omp for nowait
  //#pragma acc parallel loop private(k, j, i, m, sum_local)
  for (k = 1; k < nz0-1; k++) {
    for (j = jst; j < jend; j++) {
      for (i = ist; i < iend; i++) {
        for (m = 0; m < 5; m++) {
          sum_local[m] = sum_local[m] + v[k][j][i][m] * v[k][j][i][m];
        }
      }
    }
  }
  for (m = 0; m < 5; m++) {
    //#pragma omp atomic
    //#pragma acc atomic
    sum[m] += sum_local[m];
  }
  } //end parallel

  for (m = 0; m < 5; m++) {
    sum[m] = sqrt ( sum[m] / ( (nx0-2)*(ny0-2)*(nz0-2) ) );
  }
}

