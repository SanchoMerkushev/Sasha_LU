#include "applu.incl"  

//---------------------------------------------------------------------
// 
// compute the regular-sparse, block upper triangular solution:
// 
// v <-- ( U-inv ) * v
// 
//---------------------------------------------------------------------
//---------------------------------------------------------------------
// To improve cache performance, second two dimensions padded by 1 
// for even number sizes only.  Only needed in v.
//---------------------------------------------------------------------
void buts(int ldmx, int ldmy, int ldmz, int nx, int ny, int nz, int k,
    double omega,
    double v[ldmz][ldmy/2*2+1][ldmx/2*2+1][5],
    double tv[ldmy][ldmx/2*2+1][5],
    double d[ldmy][ldmx/2*2+1][5][5],
    double udx[ldmy][ldmx/2*2+1][5][5],
    double udy[ldmy][ldmx/2*2+1][5][5],
    double udz[ldmy][ldmx/2*2+1][5][5],
    int ist, int iend, int jst, int jend, int nx0, int ny0)
{
  //---------------------------------------------------------------------
  // local variables
  //---------------------------------------------------------------------
  int i, j, m, diag,t;
  double tmp_buts, tmp1_buts;
  double tmat_buts[ISIZ1][5][5];
  //sync_left( ldmx, ldmy, ldmz, v );

   //#pragma acc parallel loop private(i, j, m)
  //#pragma omp for schedule(static) nowait
  //#pragma acc parallel loop private(i, j, m)
  for (j = jend - 1; j >= jst; j--) {
    for (i = iend - 1; i >= ist; i--) {
      for (m = 0; m < 5; m++) {
        tv[j][i][m] = 
          omega * (  cu[j][i][0][m] * rsd[k+1][j][i][0]
                   + cu[j][i][1][m] * rsd[k+1][j][i][1]
                   + cu[j][i][2][m] * rsd[k+1][j][i][2]
                   + cu[j][i][3][m] * rsd[k+1][j][i][3]
                   + cu[j][i][4][m] * rsd[k+1][j][i][4] );
      }
    }
  }

  //#pragma omp for schedule(static) nowait
  //#pragma acc parallel loop private(i, j, m, tmp_buts, tmp1_buts)
  //#pragma acc data create(tmat_buts[:ISIZ1][:5][:5])
  for (diag = jend - 1; diag > jst; diag--) {
    //#pragma acc parallel loop private(i, j, m, tmp_buts, tmp1_buts, diag, t)
    for (t = 0; t <= (jend - jst) - diag; t++) {
       j = jend - 1 - t;
       i = diag + t;
      for (m = 0; m < 5; m++) {
        tv[j][i][m] = tv[j][i][m]
          + omega * ( bu[j][i][0][m] * rsd[k][j+1][i][0]
                    + au[j][i][0][m] * rsd[k][j][i+1][0]
                    + bu[j][i][1][m] * rsd[k][j+1][i][1]
                    + au[j][i][1][m] * rsd[k][j][i+1][1]
                    + bu[j][i][2][m] * rsd[k][j+1][i][2]
                    + au[j][i][2][m] * rsd[k][j][i+1][2]
                    + bu[j][i][3][m] * rsd[k][j+1][i][3]
                    + au[j][i][3][m] * rsd[k][j][i+1][3]
                    + bu[j][i][4][m] * rsd[k][j+1][i][4]
                    + au[j][i][4][m] * rsd[k][j][i+1][4] );
      }

      //---------------------------------------------------------------------
      // diagonal block inversion
      //---------------------------------------------------------------------
      for (m = 0; m < 5; m++) {
        tmat_buts[j][m][0] = du[j][i][0][m];
        tmat_buts[j][m][1] = du[j][i][1][m];
        tmat_buts[j][m][2] = du[j][i][2][m];
        tmat_buts[j][m][3] = du[j][i][3][m];
        tmat_buts[j][m][4] = du[j][i][4][m];
      }

      tmp1_buts = 1.0 / tmat_buts[j][0][0];
      tmp_buts = tmp1_buts * tmat_buts[j][1][0];
      tmat_buts[j][1][1] =  tmat_buts[j][1][1] - tmp_buts * tmat_buts[j][0][1];
      tmat_buts[j][1][2] =  tmat_buts[j][1][2] - tmp_buts * tmat_buts[j][0][2];
      tmat_buts[j][1][3] =  tmat_buts[j][1][3] - tmp_buts * tmat_buts[j][0][3];
      tmat_buts[j][1][4] =  tmat_buts[j][1][4] - tmp_buts * tmat_buts[j][0][4];
      tv[j][i][1] = tv[j][i][1] - tv[j][i][0] * tmp_buts;

      tmp_buts = tmp1_buts * tmat_buts[j][2][0];
      tmat_buts[j][2][1] =  tmat_buts[j][2][1] - tmp_buts * tmat_buts[j][0][1];
      tmat_buts[j][2][2] =  tmat_buts[j][2][2] - tmp_buts * tmat_buts[j][0][2];
      tmat_buts[j][2][3] =  tmat_buts[j][2][3] - tmp_buts * tmat_buts[j][0][3];
      tmat_buts[j][2][4] =  tmat_buts[j][2][4] - tmp_buts * tmat_buts[j][0][4];
      tv[j][i][2] = tv[j][i][2] - tv[j][i][0] * tmp_buts;

      tmp_buts = tmp1_buts * tmat_buts[j][3][0];
      tmat_buts[j][3][1] =  tmat_buts[j][3][1] - tmp_buts * tmat_buts[j][0][1];
      tmat_buts[j][3][2] =  tmat_buts[j][3][2] - tmp_buts * tmat_buts[j][0][2];
      tmat_buts[j][3][3] =  tmat_buts[j][3][3] - tmp_buts * tmat_buts[j][0][3];
      tmat_buts[j][3][4] =  tmat_buts[j][3][4] - tmp_buts * tmat_buts[j][0][4];
      tv[j][i][3] = tv[j][i][3] - tv[j][i][0] * tmp_buts;

      tmp_buts = tmp1_buts * tmat_buts[j][4][0];
      tmat_buts[j][4][1] =  tmat_buts[j][4][1] - tmp_buts * tmat_buts[j][0][1];
      tmat_buts[j][4][2] =  tmat_buts[j][4][2] - tmp_buts * tmat_buts[j][0][2];
      tmat_buts[j][4][3] =  tmat_buts[j][4][3] - tmp_buts * tmat_buts[j][0][3];
      tmat_buts[j][4][4] =  tmat_buts[j][4][4] - tmp_buts * tmat_buts[j][0][4];
      tv[j][i][4] = tv[j][i][4] - tv[j][i][0] * tmp_buts;

      tmp1_buts = 1.0 / tmat_buts[j][1][1];
      tmp_buts = tmp1_buts * tmat_buts[j][2][1];
      tmat_buts[j][2][2] =  tmat_buts[j][2][2] - tmp_buts * tmat_buts[j][1][2];
      tmat_buts[j][2][3] =  tmat_buts[j][2][3] - tmp_buts * tmat_buts[j][1][3];
      tmat_buts[j][2][4] =  tmat_buts[j][2][4] - tmp_buts * tmat_buts[j][1][4];
      tv[j][i][2] = tv[j][i][2] - tv[j][i][1] * tmp_buts;

      tmp_buts = tmp1_buts * tmat_buts[j][3][1];
      tmat_buts[j][3][2] =  tmat_buts[j][3][2] - tmp_buts * tmat_buts[j][1][2];
      tmat_buts[j][3][3] =  tmat_buts[j][3][3] - tmp_buts * tmat_buts[j][1][3];
      tmat_buts[j][3][4] =  tmat_buts[j][3][4] - tmp_buts * tmat_buts[j][1][4];
      tv[j][i][3] = tv[j][i][3] - tv[j][i][1] * tmp_buts;

      tmp_buts = tmp1_buts * tmat_buts[j][4][1];
      tmat_buts[j][4][2] =  tmat_buts[j][4][2] - tmp_buts * tmat_buts[j][1][2];
      tmat_buts[j][4][3] =  tmat_buts[j][4][3] - tmp_buts * tmat_buts[j][1][3];
      tmat_buts[j][4][4] =  tmat_buts[j][4][4] - tmp_buts * tmat_buts[j][1][4];
      tv[j][i][4] = tv[j][i][4] - tv[j][i][1] * tmp_buts;

      tmp1_buts = 1.0 / tmat_buts[j][2][2];
      tmp_buts = tmp1_buts * tmat_buts[j][3][2];
      tmat_buts[j][3][3] =  tmat_buts[j][3][3] - tmp_buts * tmat_buts[j][2][3];
      tmat_buts[j][3][4] =  tmat_buts[j][3][4] - tmp_buts * tmat_buts[j][2][4];
      tv[j][i][3] = tv[j][i][3] - tv[j][i][2] * tmp_buts;

      tmp_buts = tmp1_buts * tmat_buts[j][4][2];
      tmat_buts[j][4][3] =  tmat_buts[j][4][3] - tmp_buts * tmat_buts[j][2][3];
      tmat_buts[j][4][4] =  tmat_buts[j][4][4] - tmp_buts * tmat_buts[j][2][4];
      tv[j][i][4] = tv[j][i][4] - tv[j][i][2] * tmp_buts;

      tmp1_buts = 1.0 / tmat_buts[j][3][3];
      tmp_buts = tmp1_buts * tmat_buts[j][4][3];
      tmat_buts[j][4][4] =  tmat_buts[j][4][4] - tmp_buts * tmat_buts[j][3][4];
      tv[j][i][4] = tv[j][i][4] - tv[j][i][3] * tmp_buts;

      //---------------------------------------------------------------------
      // back substitution
      //---------------------------------------------------------------------
      tv[j][i][4] = tv[j][i][4] / tmat_buts[j][4][4];

      tv[j][i][3] = tv[j][i][3] - tmat_buts[j][3][4] * tv[j][i][4];
      tv[j][i][3] = tv[j][i][3] / tmat_buts[j][3][3];

      tv[j][i][2] = tv[j][i][2]
        - tmat_buts[j][2][3] * tv[j][i][3]
        - tmat_buts[j][2][4] * tv[j][i][4];
      tv[j][i][2] = tv[j][i][2] / tmat_buts[j][2][2];

      tv[j][i][1] = tv[j][i][1]
        - tmat_buts[j][1][2] * tv[j][i][2]
        - tmat_buts[j][1][3] * tv[j][i][3]
        - tmat_buts[j][1][4] * tv[j][i][4];
      tv[j][i][1] = tv[j][i][1] / tmat_buts[j][1][1];

      tv[j][i][0] = tv[j][i][0]
        - tmat_buts[j][0][1] * tv[j][i][1]
        - tmat_buts[j][0][2] * tv[j][i][2]
        - tmat_buts[j][0][3] * tv[j][i][3]
        - tmat_buts[j][0][4] * tv[j][i][4];
      tv[j][i][0] = tv[j][i][0] / tmat_buts[j][0][0];

      rsd[k][j][i][0] = rsd[k][j][i][0] - tv[j][i][0];
      rsd[k][j][i][1] = rsd[k][j][i][1] - tv[j][i][1];
      rsd[k][j][i][2] = rsd[k][j][i][2] - tv[j][i][2];
      rsd[k][j][i][3] = rsd[k][j][i][3] - tv[j][i][3];
      rsd[k][j][i][4] = rsd[k][j][i][4] - tv[j][i][4];
    }
  }
  //#pragma acc data create(tmat_buts[:ISIZ1][:5][:5])
  for (diag = jend  - 1; diag >= jst; diag--) {
    //#pragma acc parallel loop private(i, j, m, tmp_buts, tmp1_buts, diag, t)
    for (t = 0; t <= diag - jst; t++) {
      j = diag - t;
      i = jst + t;
      for (m = 0; m < 5; m++) {
        tv[j][i][m] = tv[j][i][m]
          + omega * ( bu[j][i][0][m] * rsd[k][j+1][i][0]
                    + au[j][i][0][m] * rsd[k][j][i+1][0]
                    + bu[j][i][1][m] * rsd[k][j+1][i][1]
                    + au[j][i][1][m] * rsd[k][j][i+1][1]
                    + bu[j][i][2][m] * rsd[k][j+1][i][2]
                    + au[j][i][2][m] * rsd[k][j][i+1][2]
                    + bu[j][i][3][m] * rsd[k][j+1][i][3]
                    + au[j][i][3][m] * rsd[k][j][i+1][3]
                    + bu[j][i][4][m] * rsd[k][j+1][i][4]
                    + au[j][i][4][m] * rsd[k][j][i+1][4] );
      }

      //---------------------------------------------------------------------
      // diagonal block inversion
      //---------------------------------------------------------------------
      for (m = 0; m < 5; m++) {
        tmat_buts[j][m][0] = du[j][i][0][m];
        tmat_buts[j][m][1] = du[j][i][1][m];
        tmat_buts[j][m][2] = du[j][i][2][m];
        tmat_buts[j][m][3] = du[j][i][3][m];
        tmat_buts[j][m][4] = du[j][i][4][m];
      }

      tmp1_buts = 1.0 / tmat_buts[j][0][0];
      tmp_buts = tmp1_buts * tmat_buts[j][1][0];
      tmat_buts[j][1][1] =  tmat_buts[j][1][1] - tmp_buts * tmat_buts[j][0][1];
      tmat_buts[j][1][2] =  tmat_buts[j][1][2] - tmp_buts * tmat_buts[j][0][2];
      tmat_buts[j][1][3] =  tmat_buts[j][1][3] - tmp_buts * tmat_buts[j][0][3];
      tmat_buts[j][1][4] =  tmat_buts[j][1][4] - tmp_buts * tmat_buts[j][0][4];
      tv[j][i][1] = tv[j][i][1] - tv[j][i][0] * tmp_buts;

      tmp_buts = tmp1_buts * tmat_buts[j][2][0];
      tmat_buts[j][2][1] =  tmat_buts[j][2][1] - tmp_buts * tmat_buts[j][0][1];
      tmat_buts[j][2][2] =  tmat_buts[j][2][2] - tmp_buts * tmat_buts[j][0][2];
      tmat_buts[j][2][3] =  tmat_buts[j][2][3] - tmp_buts * tmat_buts[j][0][3];
      tmat_buts[j][2][4] =  tmat_buts[j][2][4] - tmp_buts * tmat_buts[j][0][4];
      tv[j][i][2] = tv[j][i][2] - tv[j][i][0] * tmp_buts;

      tmp_buts = tmp1_buts * tmat_buts[j][3][0];
      tmat_buts[j][3][1] =  tmat_buts[j][3][1] - tmp_buts * tmat_buts[j][0][1];
      tmat_buts[j][3][2] =  tmat_buts[j][3][2] - tmp_buts * tmat_buts[j][0][2];
      tmat_buts[j][3][3] =  tmat_buts[j][3][3] - tmp_buts * tmat_buts[j][0][3];
      tmat_buts[j][3][4] =  tmat_buts[j][3][4] - tmp_buts * tmat_buts[j][0][4];
      tv[j][i][3] = tv[j][i][3] - tv[j][i][0] * tmp_buts;

      tmp_buts = tmp1_buts * tmat_buts[j][4][0];
      tmat_buts[j][4][1] =  tmat_buts[j][4][1] - tmp_buts * tmat_buts[j][0][1];
      tmat_buts[j][4][2] =  tmat_buts[j][4][2] - tmp_buts * tmat_buts[j][0][2];
      tmat_buts[j][4][3] =  tmat_buts[j][4][3] - tmp_buts * tmat_buts[j][0][3];
      tmat_buts[j][4][4] =  tmat_buts[j][4][4] - tmp_buts * tmat_buts[j][0][4];
      tv[j][i][4] = tv[j][i][4] - tv[j][i][0] * tmp_buts;

      tmp1_buts = 1.0 / tmat_buts[j][1][1];
      tmp_buts = tmp1_buts * tmat_buts[j][2][1];
      tmat_buts[j][2][2] =  tmat_buts[j][2][2] - tmp_buts * tmat_buts[j][1][2];
      tmat_buts[j][2][3] =  tmat_buts[j][2][3] - tmp_buts * tmat_buts[j][1][3];
      tmat_buts[j][2][4] =  tmat_buts[j][2][4] - tmp_buts * tmat_buts[j][1][4];
      tv[j][i][2] = tv[j][i][2] - tv[j][i][1] * tmp_buts;

      tmp_buts = tmp1_buts * tmat_buts[j][3][1];
      tmat_buts[j][3][2] =  tmat_buts[j][3][2] - tmp_buts * tmat_buts[j][1][2];
      tmat_buts[j][3][3] =  tmat_buts[j][3][3] - tmp_buts * tmat_buts[j][1][3];
      tmat_buts[j][3][4] =  tmat_buts[j][3][4] - tmp_buts * tmat_buts[j][1][4];
      tv[j][i][3] = tv[j][i][3] - tv[j][i][1] * tmp_buts;

      tmp_buts = tmp1_buts * tmat_buts[j][4][1];
      tmat_buts[j][4][2] =  tmat_buts[j][4][2] - tmp_buts * tmat_buts[j][1][2];
      tmat_buts[j][4][3] =  tmat_buts[j][4][3] - tmp_buts * tmat_buts[j][1][3];
      tmat_buts[j][4][4] =  tmat_buts[j][4][4] - tmp_buts * tmat_buts[j][1][4];
      tv[j][i][4] = tv[j][i][4] - tv[j][i][1] * tmp_buts;

      tmp1_buts = 1.0 / tmat_buts[j][2][2];
      tmp_buts = tmp1_buts * tmat_buts[j][3][2];
      tmat_buts[j][3][3] =  tmat_buts[j][3][3] - tmp_buts * tmat_buts[j][2][3];
      tmat_buts[j][3][4] =  tmat_buts[j][3][4] - tmp_buts * tmat_buts[j][2][4];
      tv[j][i][3] = tv[j][i][3] - tv[j][i][2] * tmp_buts;

      tmp_buts = tmp1_buts * tmat_buts[j][4][2];
      tmat_buts[j][4][3] =  tmat_buts[j][4][3] - tmp_buts * tmat_buts[j][2][3];
      tmat_buts[j][4][4] =  tmat_buts[j][4][4] - tmp_buts * tmat_buts[j][2][4];
      tv[j][i][4] = tv[j][i][4] - tv[j][i][2] * tmp_buts;

      tmp1_buts = 1.0 / tmat_buts[j][3][3];
      tmp_buts = tmp1_buts * tmat_buts[j][4][3];
      tmat_buts[j][4][4] =  tmat_buts[j][4][4] - tmp_buts * tmat_buts[j][3][4];
      tv[j][i][4] = tv[j][i][4] - tv[j][i][3] * tmp_buts;

      //---------------------------------------------------------------------
      // back substitution
      //---------------------------------------------------------------------
      tv[j][i][4] = tv[j][i][4] / tmat_buts[j][4][4];

      tv[j][i][3] = tv[j][i][3] - tmat_buts[j][3][4] * tv[j][i][4];
      tv[j][i][3] = tv[j][i][3] / tmat_buts[j][3][3];

      tv[j][i][2] = tv[j][i][2]
        - tmat_buts[j][2][3] * tv[j][i][3]
        - tmat_buts[j][2][4] * tv[j][i][4];
      tv[j][i][2] = tv[j][i][2] / tmat_buts[j][2][2];

      tv[j][i][1] = tv[j][i][1]
        - tmat_buts[j][1][2] * tv[j][i][2]
        - tmat_buts[j][1][3] * tv[j][i][3]
        - tmat_buts[j][1][4] * tv[j][i][4];
      tv[j][i][1] = tv[j][i][1] / tmat_buts[j][1][1];

      tv[j][i][0] = tv[j][i][0]
        - tmat_buts[j][0][1] * tv[j][i][1]
        - tmat_buts[j][0][2] * tv[j][i][2]
        - tmat_buts[j][0][3] * tv[j][i][3]
        - tmat_buts[j][0][4] * tv[j][i][4];
      tv[j][i][0] = tv[j][i][0] / tmat_buts[j][0][0];

      rsd[k][j][i][0] = rsd[k][j][i][0] - tv[j][i][0];
      rsd[k][j][i][1] = rsd[k][j][i][1] - tv[j][i][1];
      rsd[k][j][i][2] = rsd[k][j][i][2] - tv[j][i][2];
      rsd[k][j][i][3] = rsd[k][j][i][3] - tv[j][i][3];
      rsd[k][j][i][4] = rsd[k][j][i][4] - tv[j][i][4];
    }
  }
  //sync_right( ldmx, ldmy, ldmz, v );
}

