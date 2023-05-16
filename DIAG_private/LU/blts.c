#include "applu.incl"

//---------------------------------------------------------------------
// 
// compute the regular-sparse, block lower triangular solution:
// 
// v <-- ( L-inv ) * v
// 
//---------------------------------------------------------------------
//---------------------------------------------------------------------
// To improve cache performance, second two dimensions padded by 1 
// for even number sizes only.  Only needed in v.
//---------------------------------------------------------------------
void blts(int ldmx, int ldmy, int ldmz, int nx, int ny, int nz, int k,
    double omega,
    double rsd[ldmz][ldmy/2*2+1][ldmx/2*2+1][5], 
    double a[ldmy][ldmx/2*2+1][5][5],
    double b[ldmy][ldmx/2*2+1][5][5],
    double c[ldmy][ldmx/2*2+1][5][5],
    double d[ldmy][ldmx/2*2+1][5][5],
    int ist, int iend, int jst, int jend, int nx0, int ny0)
{
  // v->rsd ldz->a, ldy->b, ldx->c, d->d
  //---------------------------------------------------------------------
  // local variables
  //---------------------------------------------------------------------
  int i, j, m, diag, t, j_start, j_end;
  double tmp, tmp1;  
  double tmat[5][5], tv[5];

  //sync_left( ldmx, ldmy, ldmz, v );

  //double (*vk)[ldmx/2*2+1][5] = rsd[k];
  //double (*rsd[k]m1)[ldmx/2*2+1][5] = rsd[k-1];


  //#pragma omp for schedule(static) nowait
  #pragma acc parallel loop private(i, j, m)
  for (j = jst; j < jend; j++) {
    for (i = ist; i < iend; i++) {
      for (m = 0; m < 5; m++) {
        rsd[k][j][i][m] =  rsd[k][j][i][m]
          - omega * (  a[j][i][0][m] * rsd[k-1][j][i][0]
                     + a[j][i][1][m] * rsd[k-1][j][i][1]
                     + a[j][i][2][m] * rsd[k-1][j][i][2]
                     + a[j][i][3][m] * rsd[k-1][j][i][3]
                     + a[j][i][4][m] * rsd[k-1][j][i][4] );
      }
    }
  }


  //#pragma omp for schedule(static) nowait
  // k, i, j [1, ISIZ1 - 2]
  for (diag = 2; diag <= (ISIZ1 - 2) * 2; diag++) {
    j_start = max(1, diag - ISIZ1 + 2);
    j_end = min(diag, ISIZ1 - 1);
    #pragma acc parallel loop private(i, j, m, tmp, tmp1, tmat, tv)
    for (j = j_start; j < j_end; j++) {
      i = diag - j;
      for (m = 0; m < 5; m++) {
        tv[m] =  rsd[k][j][i][m]
          - omega * ( b[j][i][0][m] * rsd[k][j-1][i][0]
                    + c[j][i][0][m] * rsd[k][j][i-1][0]
                    + b[j][i][1][m] * rsd[k][j-1][i][1]
                    + c[j][i][1][m] * rsd[k][j][i-1][1]
                    + b[j][i][2][m] * rsd[k][j-1][i][2]
                    + c[j][i][2][m] * rsd[k][j][i-1][2]
                    + b[j][i][3][m] * rsd[k][j-1][i][3]
                    + c[j][i][3][m] * rsd[k][j][i-1][3]
                    + b[j][i][4][m] * rsd[k][j-1][i][4]
                    + c[j][i][4][m] * rsd[k][j][i-1][4] );
      }

      //---------------------------------------------------------------------
      // diagonal block inversion
      // 
      // forward elimination
      //---------------------------------------------------------------------
      for (m = 0; m < 5; m++) {
        tmat[m][0] = d[j][i][0][m];
        tmat[m][1] = d[j][i][1][m];
        tmat[m][2] = d[j][i][2][m];
        tmat[m][3] = d[j][i][3][m];
        tmat[m][4] = d[j][i][4][m];
      }

      tmp1 = 1.0 / tmat[0][0];
      tmp = tmp1 * tmat[1][0];
      tmat[1][1] =  tmat[1][1] - tmp * tmat[0][1];
      tmat[1][2] =  tmat[1][2] - tmp * tmat[0][2];
      tmat[1][3] =  tmat[1][3] - tmp * tmat[0][3];
      tmat[1][4] =  tmat[1][4] - tmp * tmat[0][4];
      tv[1] = tv[1] - tv[0] * tmp;

      tmp = tmp1 * tmat[2][0];
      tmat[2][1] =  tmat[2][1] - tmp * tmat[0][1];
      tmat[2][2] =  tmat[2][2] - tmp * tmat[0][2];
      tmat[2][3] =  tmat[2][3] - tmp * tmat[0][3];
      tmat[2][4] =  tmat[2][4] - tmp * tmat[0][4];
      tv[2] = tv[2] - tv[0] * tmp;

      tmp = tmp1 * tmat[3][0];
      tmat[3][1] =  tmat[3][1] - tmp * tmat[0][1];
      tmat[3][2] =  tmat[3][2] - tmp * tmat[0][2];
      tmat[3][3] =  tmat[3][3] - tmp * tmat[0][3];
      tmat[3][4] =  tmat[3][4] - tmp * tmat[0][4];
      tv[3] = tv[3] - tv[0] * tmp;

      tmp = tmp1 * tmat[4][0];
      tmat[4][1] =  tmat[4][1] - tmp * tmat[0][1];
      tmat[4][2] =  tmat[4][2] - tmp * tmat[0][2];
      tmat[4][3] =  tmat[4][3] - tmp * tmat[0][3];
      tmat[4][4] =  tmat[4][4] - tmp * tmat[0][4];
      tv[4] = tv[4] - tv[0] * tmp;

      tmp1 = 1.0 / tmat[1][1];
      tmp = tmp1 * tmat[2][1];
      tmat[2][2] =  tmat[2][2] - tmp * tmat[1][2];
      tmat[2][3] =  tmat[2][3] - tmp * tmat[1][3];
      tmat[2][4] =  tmat[2][4] - tmp * tmat[1][4];
      tv[2] = tv[2] - tv[1] * tmp;

      tmp = tmp1 * tmat[3][1];
      tmat[3][2] =  tmat[3][2] - tmp * tmat[1][2];
      tmat[3][3] =  tmat[3][3] - tmp * tmat[1][3];
      tmat[3][4] =  tmat[3][4] - tmp * tmat[1][4];
      tv[3] = tv[3] - tv[1] * tmp;

      tmp = tmp1 * tmat[4][1];
      tmat[4][2] =  tmat[4][2] - tmp * tmat[1][2];
      tmat[4][3] =  tmat[4][3] - tmp * tmat[1][3];
      tmat[4][4] =  tmat[4][4] - tmp * tmat[1][4];
      tv[4] = tv[4] - tv[1] * tmp;

      tmp1 = 1.0 / tmat[2][2];
      tmp = tmp1 * tmat[3][2];
      tmat[3][3] =  tmat[3][3] - tmp * tmat[2][3];
      tmat[3][4] =  tmat[3][4] - tmp * tmat[2][4];
      tv[3] = tv[3] - tv[2] * tmp;

      tmp = tmp1 * tmat[4][2];
      tmat[4][3] =  tmat[4][3] - tmp * tmat[2][3];
      tmat[4][4] =  tmat[4][4] - tmp * tmat[2][4];
      tv[4] = tv[4] - tv[2] * tmp;

      tmp1 = 1.0 / tmat[3][3];
      tmp = tmp1 * tmat[4][3];
      tmat[4][4] =  tmat[4][4] - tmp * tmat[3][4];
      tv[4] = tv[4] - tv[3] * tmp;

      //---------------------------------------------------------------------
      // back substitution
      //---------------------------------------------------------------------
      rsd[k][j][i][4] = tv[4] / tmat[4][4];

      tv[3] = tv[3] 
        - tmat[3][4] * rsd[k][j][i][4];
      rsd[k][j][i][3] = tv[3] / tmat[3][3];

      tv[2] = tv[2]
        - tmat[2][3] * rsd[k][j][i][3]
        - tmat[2][4] * rsd[k][j][i][4];
      rsd[k][j][i][2] = tv[2] / tmat[2][2];

      tv[1] = tv[1]
        - tmat[1][2] * rsd[k][j][i][2]
        - tmat[1][3] * rsd[k][j][i][3]
        - tmat[1][4] * rsd[k][j][i][4];
      rsd[k][j][i][1] = tv[1] / tmat[1][1];

      tv[0] = tv[0]
        - tmat[0][1] * rsd[k][j][i][1]
        - tmat[0][2] * rsd[k][j][i][2]
        - tmat[0][3] * rsd[k][j][i][3]
        - tmat[0][4] * rsd[k][j][i][4];
      rsd[k][j][i][0] = tv[0] / tmat[0][0];
    }
  }

  //sync_right( ldmx, ldmy, ldmz, v );
}

