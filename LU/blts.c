//-------------------------------------------------------------------------//
//                                                                         //
//  This benchmark is an OpenMP C version of the NPB LU code. This OpenMP  //
//  C version is developed by the Center for Manycore Programming at Seoul //
//  National University and derived from the OpenMP Fortran versions in    //
//  "NPB3.3-OMP" developed by NAS.                                         //
//                                                                         //
//  Permission to use, copy, distribute and modify this software for any   //
//  purpose with or without fee is hereby granted. This software is        //
//  provided "as is" without express or implied warranty.                  //
//                                                                         //
//  Information on NPB 3.3, including the technical report, the original   //
//  specifications, source code, results and information on how to submit  //
//  new results, is available at:                                          //
//                                                                         //
//           http://www.nas.nasa.gov/Software/NPB/                         //
//                                                                         //
//  Send comments or suggestions for this OpenMP C version to              //
//  cmp@aces.snu.ac.kr                                                     //
//                                                                         //
//          Center for Manycore Programming                                //
//          School of Computer Science and Engineering                     //
//          Seoul National University                                      //
//          Seoul 151-744, Korea                                           //
//                                                                         //
//          E-mail:  cmp@aces.snu.ac.kr                                    //
//                                                                         //
//-------------------------------------------------------------------------//

//-------------------------------------------------------------------------//
// Authors: Sangmin Seo, Jungwon Kim, Jun Lee, Jeongho Nah, Gangwon Jo,    //
//          and Jaejin Lee                                                 //
//-------------------------------------------------------------------------//

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
  double tmp, tmp1;  
  double tmat[ISIZ1][5][5], tv[ISIZ1][5];

  //sync_left( ldmx, ldmy, ldmz, v );

  double (*vk)[ldmx/2*2+1][5] = v[k];
  double (*vkm1)[ldmx/2*2+1][5] = v[k-1];


  //#pragma omp for schedule(static) nowait
  //#pragma acc parallel loop private(i, j, m)
  for (j = jst; j < jend; j++) {
    for (i = ist; i < iend; i++) {
      for (m = 0; m < 5; m++) {
        vk[j][i][m] =  vk[j][i][m]
          - omega * (  ldz[j][i][0][m] * vkm1[j][i][0]
                     + ldz[j][i][1][m] * vkm1[j][i][1]
                     + ldz[j][i][2][m] * vkm1[j][i][2]
                     + ldz[j][i][3][m] * vkm1[j][i][3]
                     + ldz[j][i][4][m] * vkm1[j][i][4] );
      }
    }
  }
  //#pragma omp for schedule(static) nowait
  //#pragma acc enter data copyin(tv[:ISIZ1][:5], tmat[:ISIZ1][:5][:5], d[:ISIZ1][:ISIZ1/2*2+1][:5][:5], vk[:ISIZ1/2*2+1][:ISIZ1/2*2+1][:5], ldy[:ISIZ1][:ISIZ1/2*2+1][:5][:5], ldx[:ISIZ1][:ISIZ1/2*2+1][:5][:5])
  for (diag = jst; diag < jend; diag++) {
    #pragma acc parallel loop private(t, diag, i, j, m, tmp, tmp1)
    for (t = 0; t <= diag - jst; t++) {
      j = diag - t;
      i = jst + t;
      for (m = 0; m < 5; m++) {
        tv[j][m] =  vk[j][i][m]
          - omega * ( ldy[j][i][0][m] * vk[j-1][i][0]
                    + ldx[j][i][0][m] * vk[j][i-1][0]
                    + ldy[j][i][1][m] * vk[j-1][i][1]
                    + ldx[j][i][1][m] * vk[j][i-1][1]
                    + ldy[j][i][2][m] * vk[j-1][i][2]
                    + ldx[j][i][2][m] * vk[j][i-1][2]
                    + ldy[j][i][3][m] * vk[j-1][i][3]
                    + ldx[j][i][3][m] * vk[j][i-1][3]
                    + ldy[j][i][4][m] * vk[j-1][i][4]
                    + ldx[j][i][4][m] * vk[j][i-1][4] );
      }
      //---------------------------------------------------------------------
      // diagonal block inversion
      // 
      // forward elimination
      //---------------------------------------------------------------------
      for (m = 0; m < 5; m++) {
        tmat[j][m][0] = d[j][i][0][m];
        tmat[j][m][1] = d[j][i][1][m];
        tmat[j][m][2] = d[j][i][2][m];
        tmat[j][m][3] = d[j][i][3][m];
        tmat[j][m][4] = d[j][i][4][m];
      }
      tmp1 = 1.0 / tmat[j][0][0];
      tmp = tmp1 * tmat[j][1][0];
      tmat[j][1][1] =  tmat[j][1][1] - tmp * tmat[j][0][1];
      tmat[j][1][2] =  tmat[j][1][2] - tmp * tmat[j][0][2];
      tmat[j][1][3] =  tmat[j][1][3] - tmp * tmat[j][0][3];
      tmat[j][1][4] =  tmat[j][1][4] - tmp * tmat[j][0][4];
      tv[j][1] = tv[j][1] - tv[j][0] * tmp;

      tmp = tmp1 * tmat[j][2][0];
      tmat[j][2][1] =  tmat[j][2][1] - tmp * tmat[j][0][1];
      tmat[j][2][2] =  tmat[j][2][2] - tmp * tmat[j][0][2];
      tmat[j][2][3] =  tmat[j][2][3] - tmp * tmat[j][0][3];
      tmat[j][2][4] =  tmat[j][2][4] - tmp * tmat[j][0][4];
      tv[j][2] = tv[j][2] - tv[j][0] * tmp;

      tmp = tmp1 * tmat[j][3][0];
      tmat[j][3][1] =  tmat[j][3][1] - tmp * tmat[j][0][1];
      tmat[j][3][2] =  tmat[j][3][2] - tmp * tmat[j][0][2];
      tmat[j][3][3] =  tmat[j][3][3] - tmp * tmat[j][0][3];
      tmat[j][3][4] =  tmat[j][3][4] - tmp * tmat[j][0][4];
      tv[j][3] = tv[j][3] - tv[j][0] * tmp;

      tmp = tmp1 * tmat[j][4][0];
      tmat[j][4][1] =  tmat[j][4][1] - tmp * tmat[j][0][1];
      tmat[j][4][2] =  tmat[j][4][2] - tmp * tmat[j][0][2];
      tmat[j][4][3] =  tmat[j][4][3] - tmp * tmat[j][0][3];
      tmat[j][4][4] =  tmat[j][4][4] - tmp * tmat[j][0][4];
      tv[j][4] = tv[j][4] - tv[j][0] * tmp;

      tmp1 = 1.0 / tmat[j][1][1];
      tmp = tmp1 * tmat[j][2][1];
      tmat[j][2][2] =  tmat[j][2][2] - tmp * tmat[j][1][2];
      tmat[j][2][3] =  tmat[j][2][3] - tmp * tmat[j][1][3];
      tmat[j][2][4] =  tmat[j][2][4] - tmp * tmat[j][1][4];
      tv[j][2] = tv[j][2] - tv[j][1] * tmp;

      tmp = tmp1 * tmat[j][3][1];
      tmat[j][3][2] =  tmat[j][3][2] - tmp * tmat[j][1][2];
      tmat[j][3][3] =  tmat[j][3][3] - tmp * tmat[j][1][3];
      tmat[j][3][4] =  tmat[j][3][4] - tmp * tmat[j][1][4];
      tv[j][3] = tv[j][3] - tv[j][1] * tmp;

      tmp = tmp1 * tmat[j][4][1];
      tmat[j][4][2] =  tmat[j][4][2] - tmp * tmat[j][1][2];
      tmat[j][4][3] =  tmat[j][4][3] - tmp * tmat[j][1][3];
      tmat[j][4][4] =  tmat[j][4][4] - tmp * tmat[j][1][4];
      tv[j][4] = tv[j][4] - tv[j][1] * tmp;

      tmp1 = 1.0 / tmat[j][2][2];
      tmp = tmp1 * tmat[j][3][2];
      tmat[j][3][3] =  tmat[j][3][3] - tmp * tmat[j][2][3];
      tmat[j][3][4] =  tmat[j][3][4] - tmp * tmat[j][2][4];
      tv[j][3] = tv[j][3] - tv[j][2] * tmp;

      tmp = tmp1 * tmat[j][4][2];
      tmat[j][4][3] =  tmat[j][4][3] - tmp * tmat[j][2][3];
      tmat[j][4][4] =  tmat[j][4][4] - tmp * tmat[j][2][4];
      tv[j][4] = tv[j][4] - tv[j][2] * tmp;

      tmp1 = 1.0 / tmat[j][3][3];
      tmp = tmp1 * tmat[j][4][3];
      tmat[j][4][4] =  tmat[j][4][4] - tmp * tmat[j][3][4];
      tv[j][4] = tv[j][4] - tv[j][3] * tmp;
      //---------------------------------------------------------------------
      // back substitution
      //---------------------------------------------------------------------
      vk[j][i][4] = tv[j][4] / tmat[j][4][4];

      tv[j][3] = tv[j][3] 
        - tmat[j][3][4] * vk[j][i][4];
      vk[j][i][3] = tv[j][3] / tmat[j][3][3];

      tv[j][2] = tv[j][2]
        - tmat[j][2][3] * vk[j][i][3]
        - tmat[j][2][4] * vk[j][i][4];
      vk[j][i][2] = tv[j][2] / tmat[j][2][2];

      tv[j][1] = tv[j][1]
        - tmat[j][1][2] * vk[j][i][2]
        - tmat[j][1][3] * vk[j][i][3]
        - tmat[j][1][4] * vk[j][i][4];
      vk[j][i][1] = tv[j][1] / tmat[j][1][1];

      tv[j][0] = tv[j][0]
        - tmat[j][0][1] * vk[j][i][1]
        - tmat[j][0][2] * vk[j][i][2]
        - tmat[j][0][3] * vk[j][i][3]
        - tmat[j][0][4] * vk[j][i][4];
      vk[j][i][0] = tv[j][0] / tmat[j][0][0];
    }
  }
  //#pragma acc exit data copyout(tv, tmat, d, vk, ldy, ldy)
  for (diag = jst + 1; diag < jend; diag++) {
    //#pragma acc parallel loop private(t, diag, i, j, m, tmp, tmp1)
    for (t = 0; t <= (jend - jst) - diag; t++) {
      j = jend - 1 - t;
      i = diag + t;
      for (m = 0; m < 5; m++) {
        tv[j][m] =  vk[j][i][m]
          - omega * ( ldy[j][i][0][m] * vk[j-1][i][0]
                    + ldx[j][i][0][m] * vk[j][i-1][0]
                    + ldy[j][i][1][m] * vk[j-1][i][1]
                    + ldx[j][i][1][m] * vk[j][i-1][1]
                    + ldy[j][i][2][m] * vk[j-1][i][2]
                    + ldx[j][i][2][m] * vk[j][i-1][2]
                    + ldy[j][i][3][m] * vk[j-1][i][3]
                    + ldx[j][i][3][m] * vk[j][i-1][3]
                    + ldy[j][i][4][m] * vk[j-1][i][4]
                    + ldx[j][i][4][m] * vk[j][i-1][4] );
      }

      //---------------------------------------------------------------------
      // diagonal block inversion
      // 
      // forward elimination
      //---------------------------------------------------------------------
      for (m = 0; m < 5; m++) {
        tmat[j][m][0] = d[j][i][0][m];
        tmat[j][m][1] = d[j][i][1][m];
        tmat[j][m][2] = d[j][i][2][m];
        tmat[j][m][3] = d[j][i][3][m];
        tmat[j][m][4] = d[j][i][4][m];
      }

      tmp1 = 1.0 / tmat[j][0][0];
      tmp = tmp1 * tmat[j][1][0];
      tmat[j][1][1] =  tmat[j][1][1] - tmp * tmat[j][0][1];
      tmat[j][1][2] =  tmat[j][1][2] - tmp * tmat[j][0][2];
      tmat[j][1][3] =  tmat[j][1][3] - tmp * tmat[j][0][3];
      tmat[j][1][4] =  tmat[j][1][4] - tmp * tmat[j][0][4];
      tv[j][1] = tv[j][1] - tv[j][0] * tmp;

      tmp = tmp1 * tmat[j][2][0];
      tmat[j][2][1] =  tmat[j][2][1] - tmp * tmat[j][0][1];
      tmat[j][2][2] =  tmat[j][2][2] - tmp * tmat[j][0][2];
      tmat[j][2][3] =  tmat[j][2][3] - tmp * tmat[j][0][3];
      tmat[j][2][4] =  tmat[j][2][4] - tmp * tmat[j][0][4];
      tv[j][2] = tv[j][2] - tv[j][0] * tmp;

      tmp = tmp1 * tmat[j][3][0];
      tmat[j][3][1] =  tmat[j][3][1] - tmp * tmat[j][0][1];
      tmat[j][3][2] =  tmat[j][3][2] - tmp * tmat[j][0][2];
      tmat[j][3][3] =  tmat[j][3][3] - tmp * tmat[j][0][3];
      tmat[j][3][4] =  tmat[j][3][4] - tmp * tmat[j][0][4];
      tv[j][3] = tv[j][3] - tv[j][0] * tmp;

      tmp = tmp1 * tmat[j][4][0];
      tmat[j][4][1] =  tmat[j][4][1] - tmp * tmat[j][0][1];
      tmat[j][4][2] =  tmat[j][4][2] - tmp * tmat[j][0][2];
      tmat[j][4][3] =  tmat[j][4][3] - tmp * tmat[j][0][3];
      tmat[j][4][4] =  tmat[j][4][4] - tmp * tmat[j][0][4];
      tv[j][4] = tv[j][4] - tv[j][0] * tmp;

      tmp1 = 1.0 / tmat[j][1][1];
      tmp = tmp1 * tmat[j][2][1];
      tmat[j][2][2] =  tmat[j][2][2] - tmp * tmat[j][1][2];
      tmat[j][2][3] =  tmat[j][2][3] - tmp * tmat[j][1][3];
      tmat[j][2][4] =  tmat[j][2][4] - tmp * tmat[j][1][4];
      tv[j][2] = tv[j][2] - tv[j][1] * tmp;

      tmp = tmp1 * tmat[j][3][1];
      tmat[j][3][2] =  tmat[j][3][2] - tmp * tmat[j][1][2];
      tmat[j][3][3] =  tmat[j][3][3] - tmp * tmat[j][1][3];
      tmat[j][3][4] =  tmat[j][3][4] - tmp * tmat[j][1][4];
      tv[j][3] = tv[j][3] - tv[j][1] * tmp;

      tmp = tmp1 * tmat[j][4][1];
      tmat[j][4][2] =  tmat[j][4][2] - tmp * tmat[j][1][2];
      tmat[j][4][3] =  tmat[j][4][3] - tmp * tmat[j][1][3];
      tmat[j][4][4] =  tmat[j][4][4] - tmp * tmat[j][1][4];
      tv[j][4] = tv[j][4] - tv[j][1] * tmp;

      tmp1 = 1.0 / tmat[j][2][2];
      tmp = tmp1 * tmat[j][3][2];
      tmat[j][3][3] =  tmat[j][3][3] - tmp * tmat[j][2][3];
      tmat[j][3][4] =  tmat[j][3][4] - tmp * tmat[j][2][4];
      tv[j][3] = tv[j][3] - tv[j][2] * tmp;

      tmp = tmp1 * tmat[j][4][2];
      tmat[j][4][3] =  tmat[j][4][3] - tmp * tmat[j][2][3];
      tmat[j][4][4] =  tmat[j][4][4] - tmp * tmat[j][2][4];
      tv[j][4] = tv[j][4] - tv[j][2] * tmp;

      tmp1 = 1.0 / tmat[j][3][3];
      tmp = tmp1 * tmat[j][4][3];
      tmat[j][4][4] =  tmat[j][4][4] - tmp * tmat[j][3][4];
      tv[j][4] = tv[j][4] - tv[j][3] * tmp;

      //---------------------------------------------------------------------
      // back substitution
      //---------------------------------------------------------------------
      vk[j][i][4] = tv[j][4] / tmat[j][4][4];

      tv[j][3] = tv[j][3] 
        - tmat[j][3][4] * vk[j][i][4];
      vk[j][i][3] = tv[j][3] / tmat[j][3][3];

      tv[j][2] = tv[j][2]
        - tmat[j][2][3] * vk[j][i][3]
        - tmat[j][2][4] * vk[j][i][4];
      vk[j][i][2] = tv[j][2] / tmat[j][2][2];

      tv[j][1] = tv[j][1]
        - tmat[j][1][2] * vk[j][i][2]
        - tmat[j][1][3] * vk[j][i][3]
        - tmat[j][1][4] * vk[j][i][4];
      vk[j][i][1] = tv[j][1] / tmat[j][1][1];

      tv[j][0] = tv[j][0]
        - tmat[j][0][1] * vk[j][i][1]
        - tmat[j][0][2] * vk[j][i][2]
        - tmat[j][0][3] * vk[j][i][3]
        - tmat[j][0][4] * vk[j][i][4];
      vk[j][i][0] = tv[j][0] / tmat[j][0][0];
    }
  }

  //sync_right( ldmx, ldmy, ldmz, v );
}

