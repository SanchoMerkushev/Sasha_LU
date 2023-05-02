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
    double v[ldmz][ldmy/2*2+1][ldmx/2*2+1][5], 
    double ldz[ldmy][ldmx/2*2+1][5][5],
    double ldy[ldmy][ldmx/2*2+1][5][5],
    double ldx[ldmy][ldmx/2*2+1][5][5],
    double d[ldmy][ldmx/2*2+1][5][5],
    int ist, int iend, int jst, int jend, int nx0, int ny0)
{
  //---------------------------------------------------------------------
  // local variables
  //---------------------------------------------------------------------
  int i, j, m, diag, t;
  double tmp_blts, tmp1_blts;  
  double tmat_blts[ISIZ1][5][5], tv_blts[ISIZ1][5];

  //sync_left( ldmx, ldmy, ldmz, v );

  //double (*vk)[ldmx/2*2+1][5] = rsd[k];
  //double (*rsd[k]m1)[ldmx/2*2+1][5] = rsd[k-1];


  //#pragma omp for schedule(static) nowait
  //#pragma acc parallel loop private(i, j, m)
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
  //#pragma acc data create(tmat_blts[:ISIZ1][:5][:5], tv_blts[:ISIZ1][:5])
  for (diag = jst; diag < jend; diag++) {
    //#pragma acc parallel loop private(t, diag, i, j, m, tmp_blts, tmp1_blts)
    for (t = 0; t <= diag - jst; t++) {
      j = diag - t;
      i = jst + t;
      for (m = 0; m < 5; m++) {
        tv_blts[j][m] =  rsd[k][j][i][m]
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
        tmat_blts[j][m][0] = d[j][i][0][m];
        tmat_blts[j][m][1] = d[j][i][1][m];
        tmat_blts[j][m][2] = d[j][i][2][m];
        tmat_blts[j][m][3] = d[j][i][3][m];
        tmat_blts[j][m][4] = d[j][i][4][m];
      }

      tmp1_blts = 1.0 / tmat_blts[j][0][0];
      tmp_blts = tmp1_blts * tmat_blts[j][1][0];
      tmat_blts[j][1][1] =  tmat_blts[j][1][1] - tmp_blts * tmat_blts[j][0][1];
      tmat_blts[j][1][2] =  tmat_blts[j][1][2] - tmp_blts * tmat_blts[j][0][2];
      tmat_blts[j][1][3] =  tmat_blts[j][1][3] - tmp_blts * tmat_blts[j][0][3];
      tmat_blts[j][1][4] =  tmat_blts[j][1][4] - tmp_blts * tmat_blts[j][0][4];
      tv_blts[j][1] = tv_blts[j][1] - tv_blts[j][0] * tmp_blts;

      tmp_blts = tmp1_blts * tmat_blts[j][2][0];
      tmat_blts[j][2][1] =  tmat_blts[j][2][1] - tmp_blts * tmat_blts[j][0][1];
      tmat_blts[j][2][2] =  tmat_blts[j][2][2] - tmp_blts * tmat_blts[j][0][2];
      tmat_blts[j][2][3] =  tmat_blts[j][2][3] - tmp_blts * tmat_blts[j][0][3];
      tmat_blts[j][2][4] =  tmat_blts[j][2][4] - tmp_blts * tmat_blts[j][0][4];
      tv_blts[j][2] = tv_blts[j][2] - tv_blts[j][0] * tmp_blts;

      tmp_blts = tmp1_blts * tmat_blts[j][3][0];
      tmat_blts[j][3][1] =  tmat_blts[j][3][1] - tmp_blts * tmat_blts[j][0][1];
      tmat_blts[j][3][2] =  tmat_blts[j][3][2] - tmp_blts * tmat_blts[j][0][2];
      tmat_blts[j][3][3] =  tmat_blts[j][3][3] - tmp_blts * tmat_blts[j][0][3];
      tmat_blts[j][3][4] =  tmat_blts[j][3][4] - tmp_blts * tmat_blts[j][0][4];
      tv_blts[j][3] = tv_blts[j][3] - tv_blts[j][0] * tmp_blts;

      tmp_blts = tmp1_blts * tmat_blts[j][4][0];
      tmat_blts[j][4][1] =  tmat_blts[j][4][1] - tmp_blts * tmat_blts[j][0][1];
      tmat_blts[j][4][2] =  tmat_blts[j][4][2] - tmp_blts * tmat_blts[j][0][2];
      tmat_blts[j][4][3] =  tmat_blts[j][4][3] - tmp_blts * tmat_blts[j][0][3];
      tmat_blts[j][4][4] =  tmat_blts[j][4][4] - tmp_blts * tmat_blts[j][0][4];
      tv_blts[j][4] = tv_blts[j][4] - tv_blts[j][0] * tmp_blts;

      tmp1_blts = 1.0 / tmat_blts[j][1][1];
      tmp_blts = tmp1_blts * tmat_blts[j][2][1];
      tmat_blts[j][2][2] =  tmat_blts[j][2][2] - tmp_blts * tmat_blts[j][1][2];
      tmat_blts[j][2][3] =  tmat_blts[j][2][3] - tmp_blts * tmat_blts[j][1][3];
      tmat_blts[j][2][4] =  tmat_blts[j][2][4] - tmp_blts * tmat_blts[j][1][4];
      tv_blts[j][2] = tv_blts[j][2] - tv_blts[j][1] * tmp_blts;

      tmp_blts = tmp1_blts * tmat_blts[j][3][1];
      tmat_blts[j][3][2] =  tmat_blts[j][3][2] - tmp_blts * tmat_blts[j][1][2];
      tmat_blts[j][3][3] =  tmat_blts[j][3][3] - tmp_blts * tmat_blts[j][1][3];
      tmat_blts[j][3][4] =  tmat_blts[j][3][4] - tmp_blts * tmat_blts[j][1][4];
      tv_blts[j][3] = tv_blts[j][3] - tv_blts[j][1] * tmp_blts;

      tmp_blts = tmp1_blts * tmat_blts[j][4][1];
      tmat_blts[j][4][2] =  tmat_blts[j][4][2] - tmp_blts * tmat_blts[j][1][2];
      tmat_blts[j][4][3] =  tmat_blts[j][4][3] - tmp_blts * tmat_blts[j][1][3];
      tmat_blts[j][4][4] =  tmat_blts[j][4][4] - tmp_blts * tmat_blts[j][1][4];
      tv_blts[j][4] = tv_blts[j][4] - tv_blts[j][1] * tmp_blts;

      tmp1_blts = 1.0 / tmat_blts[j][2][2];
      tmp_blts = tmp1_blts * tmat_blts[j][3][2];
      tmat_blts[j][3][3] =  tmat_blts[j][3][3] - tmp_blts * tmat_blts[j][2][3];
      tmat_blts[j][3][4] =  tmat_blts[j][3][4] - tmp_blts * tmat_blts[j][2][4];
      tv_blts[j][3] = tv_blts[j][3] - tv_blts[j][2] * tmp_blts;

      tmp_blts = tmp1_blts * tmat_blts[j][4][2];
      tmat_blts[j][4][3] =  tmat_blts[j][4][3] - tmp_blts * tmat_blts[j][2][3];
      tmat_blts[j][4][4] =  tmat_blts[j][4][4] - tmp_blts * tmat_blts[j][2][4];
      tv_blts[j][4] = tv_blts[j][4] - tv_blts[j][2] * tmp_blts;

      tmp1_blts = 1.0 / tmat_blts[j][3][3];
      tmp_blts = tmp1_blts * tmat_blts[j][4][3];
      tmat_blts[j][4][4] =  tmat_blts[j][4][4] - tmp_blts * tmat_blts[j][3][4];
      tv_blts[j][4] = tv_blts[j][4] - tv_blts[j][3] * tmp_blts;

      //---------------------------------------------------------------------
      // back substitution
      //---------------------------------------------------------------------
      rsd[k][j][i][4] = tv_blts[j][4] / tmat_blts[j][4][4];

      tv_blts[j][3] = tv_blts[j][3] 
        - tmat_blts[j][3][4] * rsd[k][j][i][4];
      rsd[k][j][i][3] = tv_blts[j][3] / tmat_blts[j][3][3];

      tv_blts[j][2] = tv_blts[j][2]
        - tmat_blts[j][2][3] * rsd[k][j][i][3]
        - tmat_blts[j][2][4] * rsd[k][j][i][4];
      rsd[k][j][i][2] = tv_blts[j][2] / tmat_blts[j][2][2];

      tv_blts[j][1] = tv_blts[j][1]
        - tmat_blts[j][1][2] * rsd[k][j][i][2]
        - tmat_blts[j][1][3] * rsd[k][j][i][3]
        - tmat_blts[j][1][4] * rsd[k][j][i][4];
      rsd[k][j][i][1] = tv_blts[j][1] / tmat_blts[j][1][1];

      tv_blts[j][0] = tv_blts[j][0]
        - tmat_blts[j][0][1] * rsd[k][j][i][1]
        - tmat_blts[j][0][2] * rsd[k][j][i][2]
        - tmat_blts[j][0][3] * rsd[k][j][i][3]
        - tmat_blts[j][0][4] * rsd[k][j][i][4];
      rsd[k][j][i][0] = tv_blts[j][0] / tmat_blts[j][0][0];
    }
  }
  //#pragma acc data create(tmat_blts[:ISIZ1][:5][:5], tv_blts[:ISIZ1][:5])
  for (diag = jst + 1; diag < jend; diag++) {
    //#pragma acc parallel loop private(t, diag, i, j, m, tmp_blts, tmp1_blts)
    for (t = 0; t <= (jend - jst) - diag; t++) {
      j = jend - 1 - t;
      i = diag + t;
      for (m = 0; m < 5; m++) {
        tv_blts[j][m] =  rsd[k][j][i][m]
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
        tmat_blts[j][m][0] = d[j][i][0][m];
        tmat_blts[j][m][1] = d[j][i][1][m];
        tmat_blts[j][m][2] = d[j][i][2][m];
        tmat_blts[j][m][3] = d[j][i][3][m];
        tmat_blts[j][m][4] = d[j][i][4][m];
      }

      tmp1_blts = 1.0 / tmat_blts[j][0][0];
      tmp_blts = tmp1_blts * tmat_blts[j][1][0];
      tmat_blts[j][1][1] =  tmat_blts[j][1][1] - tmp_blts * tmat_blts[j][0][1];
      tmat_blts[j][1][2] =  tmat_blts[j][1][2] - tmp_blts * tmat_blts[j][0][2];
      tmat_blts[j][1][3] =  tmat_blts[j][1][3] - tmp_blts * tmat_blts[j][0][3];
      tmat_blts[j][1][4] =  tmat_blts[j][1][4] - tmp_blts * tmat_blts[j][0][4];
      tv_blts[j][1] = tv_blts[j][1] - tv_blts[j][0] * tmp_blts;

      tmp_blts = tmp1_blts * tmat_blts[j][2][0];
      tmat_blts[j][2][1] =  tmat_blts[j][2][1] - tmp_blts * tmat_blts[j][0][1];
      tmat_blts[j][2][2] =  tmat_blts[j][2][2] - tmp_blts * tmat_blts[j][0][2];
      tmat_blts[j][2][3] =  tmat_blts[j][2][3] - tmp_blts * tmat_blts[j][0][3];
      tmat_blts[j][2][4] =  tmat_blts[j][2][4] - tmp_blts * tmat_blts[j][0][4];
      tv_blts[j][2] = tv_blts[j][2] - tv_blts[j][0] * tmp_blts;

      tmp_blts = tmp1_blts * tmat_blts[j][3][0];
      tmat_blts[j][3][1] =  tmat_blts[j][3][1] - tmp_blts * tmat_blts[j][0][1];
      tmat_blts[j][3][2] =  tmat_blts[j][3][2] - tmp_blts * tmat_blts[j][0][2];
      tmat_blts[j][3][3] =  tmat_blts[j][3][3] - tmp_blts * tmat_blts[j][0][3];
      tmat_blts[j][3][4] =  tmat_blts[j][3][4] - tmp_blts * tmat_blts[j][0][4];
      tv_blts[j][3] = tv_blts[j][3] - tv_blts[j][0] * tmp_blts;

      tmp_blts = tmp1_blts * tmat_blts[j][4][0];
      tmat_blts[j][4][1] =  tmat_blts[j][4][1] - tmp_blts * tmat_blts[j][0][1];
      tmat_blts[j][4][2] =  tmat_blts[j][4][2] - tmp_blts * tmat_blts[j][0][2];
      tmat_blts[j][4][3] =  tmat_blts[j][4][3] - tmp_blts * tmat_blts[j][0][3];
      tmat_blts[j][4][4] =  tmat_blts[j][4][4] - tmp_blts * tmat_blts[j][0][4];
      tv_blts[j][4] = tv_blts[j][4] - tv_blts[j][0] * tmp_blts;

      tmp1_blts = 1.0 / tmat_blts[j][1][1];
      tmp_blts = tmp1_blts * tmat_blts[j][2][1];
      tmat_blts[j][2][2] =  tmat_blts[j][2][2] - tmp_blts * tmat_blts[j][1][2];
      tmat_blts[j][2][3] =  tmat_blts[j][2][3] - tmp_blts * tmat_blts[j][1][3];
      tmat_blts[j][2][4] =  tmat_blts[j][2][4] - tmp_blts * tmat_blts[j][1][4];
      tv_blts[j][2] = tv_blts[j][2] - tv_blts[j][1] * tmp_blts;

      tmp_blts = tmp1_blts * tmat_blts[j][3][1];
      tmat_blts[j][3][2] =  tmat_blts[j][3][2] - tmp_blts * tmat_blts[j][1][2];
      tmat_blts[j][3][3] =  tmat_blts[j][3][3] - tmp_blts * tmat_blts[j][1][3];
      tmat_blts[j][3][4] =  tmat_blts[j][3][4] - tmp_blts * tmat_blts[j][1][4];
      tv_blts[j][3] = tv_blts[j][3] - tv_blts[j][1] * tmp_blts;

      tmp_blts = tmp1_blts * tmat_blts[j][4][1];
      tmat_blts[j][4][2] =  tmat_blts[j][4][2] - tmp_blts * tmat_blts[j][1][2];
      tmat_blts[j][4][3] =  tmat_blts[j][4][3] - tmp_blts * tmat_blts[j][1][3];
      tmat_blts[j][4][4] =  tmat_blts[j][4][4] - tmp_blts * tmat_blts[j][1][4];
      tv_blts[j][4] = tv_blts[j][4] - tv_blts[j][1] * tmp_blts;

      tmp1_blts = 1.0 / tmat_blts[j][2][2];
      tmp_blts = tmp1_blts * tmat_blts[j][3][2];
      tmat_blts[j][3][3] =  tmat_blts[j][3][3] - tmp_blts * tmat_blts[j][2][3];
      tmat_blts[j][3][4] =  tmat_blts[j][3][4] - tmp_blts * tmat_blts[j][2][4];
      tv_blts[j][3] = tv_blts[j][3] - tv_blts[j][2] * tmp_blts;

      tmp_blts = tmp1_blts * tmat_blts[j][4][2];
      tmat_blts[j][4][3] =  tmat_blts[j][4][3] - tmp_blts * tmat_blts[j][2][3];
      tmat_blts[j][4][4] =  tmat_blts[j][4][4] - tmp_blts * tmat_blts[j][2][4];
      tv_blts[j][4] = tv_blts[j][4] - tv_blts[j][2] * tmp_blts;

      tmp1_blts = 1.0 / tmat_blts[j][3][3];
      tmp_blts = tmp1_blts * tmat_blts[j][4][3];
      tmat_blts[j][4][4] =  tmat_blts[j][4][4] - tmp_blts * tmat_blts[j][3][4];
      tv_blts[j][4] = tv_blts[j][4] - tv_blts[j][3] * tmp_blts;

      //---------------------------------------------------------------------
      // back substitution
      //---------------------------------------------------------------------
      rsd[k][j][i][4] = tv_blts[j][4] / tmat_blts[j][4][4];

      tv_blts[j][3] = tv_blts[j][3] 
        - tmat_blts[j][3][4] * rsd[k][j][i][4];
      rsd[k][j][i][3] = tv_blts[j][3] / tmat_blts[j][3][3];

      tv_blts[j][2] = tv_blts[j][2]
        - tmat_blts[j][2][3] * rsd[k][j][i][3]
        - tmat_blts[j][2][4] * rsd[k][j][i][4];
      rsd[k][j][i][2] = tv_blts[j][2] / tmat_blts[j][2][2];

      tv_blts[j][1] = tv_blts[j][1]
        - tmat_blts[j][1][2] * rsd[k][j][i][2]
        - tmat_blts[j][1][3] * rsd[k][j][i][3]
        - tmat_blts[j][1][4] * rsd[k][j][i][4];
      rsd[k][j][i][1] = tv_blts[j][1] / tmat_blts[j][1][1];

      tv_blts[j][0] = tv_blts[j][0]
        - tmat_blts[j][0][1] * rsd[k][j][i][1]
        - tmat_blts[j][0][2] * rsd[k][j][i][2]
        - tmat_blts[j][0][3] * rsd[k][j][i][3]
        - tmat_blts[j][0][4] * rsd[k][j][i][4];
      rsd[k][j][i][0] = tv_blts[j][0] / tmat_blts[j][0][0];
    }
  }

  //sync_right( ldmx, ldmy, ldmz, v );
}

