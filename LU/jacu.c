#include "applu.incl"

//---------------------------------------------------------------------
// compute the upper triangular part of the jacobian matrix
//---------------------------------------------------------------------
void jacu(int k)
{
  //---------------------------------------------------------------------
  // local variables
  //---------------------------------------------------------------------
  int i, j;
  double r43;
  double c1345;
  double c34;
  double tmp1_jacu, tmp2_jacu, tmp3_jacu;

  r43 = ( 4.0 / 3.0 );
  c1345 = C1 * C3 * C4 * C5;
  c34 = C3 * C4;

  //#pragma omp for schedule(static) nowait
  //#pragma acc parallel loop private(j, i, tmp1_jacu, tmp2_jacu, tmp3_jacu)
  for (j = jend - 1; j >= jst; j--) {
    for (i = iend - 1; i >= ist; i--) {
      //---------------------------------------------------------------------t 
      // form the block daigonal
      //---------------------------------------------------------------------
      tmp1_jacu = rho_i[k][j][i];
      tmp2_jacu = tmp1_jacu * tmp1_jacu;
      tmp3_jacu = tmp1_jacu * tmp2_jacu;

      du[j][i][0][0] = 1.0 + dt * 2.0 * ( tx1 * dx1 + ty1 * dy1 + tz1 * dz1 );
      du[j][i][1][0] = 0.0;
      du[j][i][2][0] = 0.0;
      du[j][i][3][0] = 0.0;
      du[j][i][4][0] = 0.0;

      du[j][i][0][1] =  dt * 2.0
        * ( - tx1 * r43 - ty1 - tz1 )
        * ( c34 * tmp2_jacu * u[k][j][i][1] );
      du[j][i][1][1] =  1.0
        + dt * 2.0 * c34 * tmp1_jacu 
        * (  tx1 * r43 + ty1 + tz1 )
        + dt * 2.0 * ( tx1 * dx2 + ty1 * dy2 + tz1 * dz2 );
      du[j][i][2][1] = 0.0;
      du[j][i][3][1] = 0.0;
      du[j][i][4][1] = 0.0;

      du[j][i][0][2] = dt * 2.0
        * ( - tx1 - ty1 * r43 - tz1 )
        * ( c34 * tmp2_jacu * u[k][j][i][2] );
      du[j][i][1][2] = 0.0;
      du[j][i][2][2] = 1.0
        + dt * 2.0 * c34 * tmp1_jacu
        * (  tx1 + ty1 * r43 + tz1 )
        + dt * 2.0 * ( tx1 * dx3 + ty1 * dy3 + tz1 * dz3 );
      du[j][i][3][2] = 0.0;
      du[j][i][4][2] = 0.0;

      du[j][i][0][3] = dt * 2.0
        * ( - tx1 - ty1 - tz1 * r43 )
        * ( c34 * tmp2_jacu * u[k][j][i][3] );
      du[j][i][1][3] = 0.0;
      du[j][i][2][3] = 0.0;
      du[j][i][3][3] = 1.0
        + dt * 2.0 * c34 * tmp1_jacu
        * (  tx1 + ty1 + tz1 * r43 )
        + dt * 2.0 * ( tx1 * dx4 + ty1 * dy4 + tz1 * dz4 );
      du[j][i][4][3] = 0.0;

      du[j][i][0][4] = -dt * 2.0
        * ( ( ( tx1 * ( r43*c34 - c1345 )
                + ty1 * ( c34 - c1345 )
                + tz1 * ( c34 - c1345 ) ) * ( u[k][j][i][1]*u[k][j][i][1] )
              + ( tx1 * ( c34 - c1345 )
                + ty1 * ( r43*c34 - c1345 )
                + tz1 * ( c34 - c1345 ) ) * ( u[k][j][i][2]*u[k][j][i][2] )
              + ( tx1 * ( c34 - c1345 )
                + ty1 * ( c34 - c1345 )
                + tz1 * ( r43*c34 - c1345 ) ) * (u[k][j][i][3]*u[k][j][i][3])
            ) * tmp3_jacu
            + ( tx1 + ty1 + tz1 ) * c1345 * tmp2_jacu * u[k][j][i][4] );

      du[j][i][1][4] = dt * 2.0
        * ( tx1 * ( r43*c34 - c1345 )
          + ty1 * (     c34 - c1345 )
          + tz1 * (     c34 - c1345 ) ) * tmp2_jacu * u[k][j][i][1];
      du[j][i][2][4] = dt * 2.0
        * ( tx1 * ( c34 - c1345 )
          + ty1 * ( r43*c34 -c1345 )
          + tz1 * ( c34 - c1345 ) ) * tmp2_jacu * u[k][j][i][2];
      du[j][i][3][4] = dt * 2.0
        * ( tx1 * ( c34 - c1345 )
          + ty1 * ( c34 - c1345 )
          + tz1 * ( r43*c34 - c1345 ) ) * tmp2_jacu * u[k][j][i][3];
      du[j][i][4][4] = 1.0
        + dt * 2.0 * ( tx1 + ty1 + tz1 ) * c1345 * tmp1_jacu
        + dt * 2.0 * ( tx1 * dx5 + ty1 * dy5 + tz1 * dz5 );

      //---------------------------------------------------------------------
      // form the first block sub-diagonal
      //---------------------------------------------------------------------
      tmp1_jacu = rho_i[k][j][i+1];
      tmp2_jacu = tmp1_jacu * tmp1_jacu;
      tmp3_jacu = tmp1_jacu * tmp2_jacu;

      au[j][i][0][0] = - dt * tx1 * dx1;
      au[j][i][1][0] =   dt * tx2;
      au[j][i][2][0] =   0.0;
      au[j][i][3][0] =   0.0;
      au[j][i][4][0] =   0.0;

      au[j][i][0][1] =  dt * tx2
        * ( - ( u[k][j][i+1][1] * tmp1_jacu ) * ( u[k][j][i+1][1] * tmp1_jacu )
            + C2 * qs[k][j][i+1] * tmp1_jacu )
        - dt * tx1 * ( - r43 * c34 * tmp2_jacu * u[k][j][i+1][1] );
      au[j][i][1][1] =  dt * tx2
        * ( ( 2.0 - C2 ) * ( u[k][j][i+1][1] * tmp1_jacu ) )
        - dt * tx1 * ( r43 * c34 * tmp1_jacu )
        - dt * tx1 * dx2;
      au[j][i][2][1] =  dt * tx2
        * ( - C2 * ( u[k][j][i+1][2] * tmp1_jacu ) );
      au[j][i][3][1] =  dt * tx2
        * ( - C2 * ( u[k][j][i+1][3] * tmp1_jacu ) );
      au[j][i][4][1] =  dt * tx2 * C2 ;

      au[j][i][0][2] =  dt * tx2
        * ( - ( u[k][j][i+1][1] * u[k][j][i+1][2] ) * tmp2_jacu )
        - dt * tx1 * ( - c34 * tmp2_jacu * u[k][j][i+1][2] );
      au[j][i][1][2] =  dt * tx2 * ( u[k][j][i+1][2] * tmp1_jacu );
      au[j][i][2][2] =  dt * tx2 * ( u[k][j][i+1][1] * tmp1_jacu )
        - dt * tx1 * ( c34 * tmp1_jacu )
        - dt * tx1 * dx3;
      au[j][i][3][2] = 0.0;
      au[j][i][4][2] = 0.0;

      au[j][i][0][3] = dt * tx2
        * ( - ( u[k][j][i+1][1]*u[k][j][i+1][3] ) * tmp2_jacu )
        - dt * tx1 * ( - c34 * tmp2_jacu * u[k][j][i+1][3] );
      au[j][i][1][3] = dt * tx2 * ( u[k][j][i+1][3] * tmp1_jacu );
      au[j][i][2][3] = 0.0;
      au[j][i][3][3] = dt * tx2 * ( u[k][j][i+1][1] * tmp1_jacu )
        - dt * tx1 * ( c34 * tmp1_jacu )
        - dt * tx1 * dx4;
      au[j][i][4][3] = 0.0;

      au[j][i][0][4] = dt * tx2
        * ( ( C2 * 2.0 * qs[k][j][i+1]
            - C1 * u[k][j][i+1][4] )
        * ( u[k][j][i+1][1] * tmp2_jacu ) )
        - dt * tx1
        * ( - ( r43*c34 - c1345 ) * tmp3_jacu * ( u[k][j][i+1][1]*u[k][j][i+1][1] )
            - (     c34 - c1345 ) * tmp3_jacu * ( u[k][j][i+1][2]*u[k][j][i+1][2] )
            - (     c34 - c1345 ) * tmp3_jacu * ( u[k][j][i+1][3]*u[k][j][i+1][3] )
            - c1345 * tmp2_jacu * u[k][j][i+1][4] );
      au[j][i][1][4] = dt * tx2
        * ( C1 * ( u[k][j][i+1][4] * tmp1_jacu )
            - C2
            * ( u[k][j][i+1][1]*u[k][j][i+1][1] * tmp2_jacu
              + qs[k][j][i+1] * tmp1_jacu ) )
        - dt * tx1
        * ( r43*c34 - c1345 ) * tmp2_jacu * u[k][j][i+1][1];
      au[j][i][2][4] = dt * tx2
        * ( - C2 * ( u[k][j][i+1][2]*u[k][j][i+1][1] ) * tmp2_jacu )
        - dt * tx1
        * (  c34 - c1345 ) * tmp2_jacu * u[k][j][i+1][2];
      au[j][i][3][4] = dt * tx2
        * ( - C2 * ( u[k][j][i+1][3]*u[k][j][i+1][1] ) * tmp2_jacu )
        - dt * tx1
        * (  c34 - c1345 ) * tmp2_jacu * u[k][j][i+1][3];
      au[j][i][4][4] = dt * tx2
        * ( C1 * ( u[k][j][i+1][1] * tmp1_jacu ) )
        - dt * tx1 * c1345 * tmp1_jacu
        - dt * tx1 * dx5;

      //---------------------------------------------------------------------
      // form the second block sub-diagonal
      //---------------------------------------------------------------------
      tmp1_jacu = rho_i[k][j+1][i];
      tmp2_jacu = tmp1_jacu * tmp1_jacu;
      tmp3_jacu = tmp1_jacu * tmp2_jacu;

      bu[j][i][0][0] = - dt * ty1 * dy1;
      bu[j][i][1][0] =   0.0;
      bu[j][i][2][0] =  dt * ty2;
      bu[j][i][3][0] =   0.0;
      bu[j][i][4][0] =   0.0;

      bu[j][i][0][1] =  dt * ty2
        * ( - ( u[k][j+1][i][1]*u[k][j+1][i][2] ) * tmp2_jacu )
        - dt * ty1 * ( - c34 * tmp2_jacu * u[k][j+1][i][1] );
      bu[j][i][1][1] =  dt * ty2 * ( u[k][j+1][i][2] * tmp1_jacu )
        - dt * ty1 * ( c34 * tmp1_jacu )
        - dt * ty1 * dy2;
      bu[j][i][2][1] =  dt * ty2 * ( u[k][j+1][i][1] * tmp1_jacu );
      bu[j][i][3][1] = 0.0;
      bu[j][i][4][1] = 0.0;

      bu[j][i][0][2] =  dt * ty2
        * ( - ( u[k][j+1][i][2] * tmp1_jacu ) * ( u[k][j+1][i][2] * tmp1_jacu )
            + C2 * ( qs[k][j+1][i] * tmp1_jacu ) )
        - dt * ty1 * ( - r43 * c34 * tmp2_jacu * u[k][j+1][i][2] );
      bu[j][i][1][2] =  dt * ty2
        * ( - C2 * ( u[k][j+1][i][1] * tmp1_jacu ) );
      bu[j][i][2][2] =  dt * ty2 * ( ( 2.0 - C2 )
          * ( u[k][j+1][i][2] * tmp1_jacu ) )
        - dt * ty1 * ( r43 * c34 * tmp1_jacu )
        - dt * ty1 * dy3;
      bu[j][i][3][2] =  dt * ty2
        * ( - C2 * ( u[k][j+1][i][3] * tmp1_jacu ) );
      bu[j][i][4][2] =  dt * ty2 * C2;

      bu[j][i][0][3] =  dt * ty2
        * ( - ( u[k][j+1][i][2]*u[k][j+1][i][3] ) * tmp2_jacu )
        - dt * ty1 * ( - c34 * tmp2_jacu * u[k][j+1][i][3] );
      bu[j][i][1][3] = 0.0;
      bu[j][i][2][3] =  dt * ty2 * ( u[k][j+1][i][3] * tmp1_jacu );
      bu[j][i][3][3] =  dt * ty2 * ( u[k][j+1][i][2] * tmp1_jacu )
        - dt * ty1 * ( c34 * tmp1_jacu )
        - dt * ty1 * dy4;
      bu[j][i][4][3] = 0.0;

      bu[j][i][0][4] =  dt * ty2
        * ( ( C2 * 2.0 * qs[k][j+1][i]
            - C1 * u[k][j+1][i][4] )
        * ( u[k][j+1][i][2] * tmp2_jacu ) )
        - dt * ty1
        * ( - (     c34 - c1345 )*tmp3_jacu*(u[k][j+1][i][1]*u[k][j+1][i][1])
            - ( r43*c34 - c1345 )*tmp3_jacu*(u[k][j+1][i][2]*u[k][j+1][i][2])
            - (     c34 - c1345 )*tmp3_jacu*(u[k][j+1][i][3]*u[k][j+1][i][3])
            - c1345*tmp2_jacu*u[k][j+1][i][4] );
      bu[j][i][1][4] =  dt * ty2
        * ( - C2 * ( u[k][j+1][i][1]*u[k][j+1][i][2] ) * tmp2_jacu )
        - dt * ty1
        * ( c34 - c1345 ) * tmp2_jacu * u[k][j+1][i][1];
      bu[j][i][2][4] =  dt * ty2
        * ( C1 * ( u[k][j+1][i][4] * tmp1_jacu )
            - C2 
            * ( qs[k][j+1][i] * tmp1_jacu
              + u[k][j+1][i][2]*u[k][j+1][i][2] * tmp2_jacu ) )
        - dt * ty1
        * ( r43*c34 - c1345 ) * tmp2_jacu * u[k][j+1][i][2];
      bu[j][i][3][4] =  dt * ty2
        * ( - C2 * ( u[k][j+1][i][2]*u[k][j+1][i][3] ) * tmp2_jacu )
        - dt * ty1 * ( c34 - c1345 ) * tmp2_jacu * u[k][j+1][i][3];
      bu[j][i][4][4] =  dt * ty2
        * ( C1 * ( u[k][j+1][i][2] * tmp1_jacu ) )
        - dt * ty1 * c1345 * tmp1_jacu
        - dt * ty1 * dy5;

      //---------------------------------------------------------------------
      // form the third block sub-diagonal
      //---------------------------------------------------------------------
      tmp1_jacu = rho_i[k+1][j][i];
      tmp2_jacu = tmp1_jacu * tmp1_jacu;
      tmp3_jacu = tmp1_jacu * tmp2_jacu;

      cu[j][i][0][0] = - dt * tz1 * dz1;
      cu[j][i][1][0] =   0.0;
      cu[j][i][2][0] =   0.0;
      cu[j][i][3][0] = dt * tz2;
      cu[j][i][4][0] =   0.0;

      cu[j][i][0][1] = dt * tz2
        * ( - ( u[k+1][j][i][1]*u[k+1][j][i][3] ) * tmp2_jacu )
        - dt * tz1 * ( - c34 * tmp2_jacu * u[k+1][j][i][1] );
      cu[j][i][1][1] = dt * tz2 * ( u[k+1][j][i][3] * tmp1_jacu )
        - dt * tz1 * c34 * tmp1_jacu
        - dt * tz1 * dz2;
      cu[j][i][2][1] = 0.0;
      cu[j][i][3][1] = dt * tz2 * ( u[k+1][j][i][1] * tmp1_jacu );
      cu[j][i][4][1] = 0.0;

      cu[j][i][0][2] = dt * tz2
        * ( - ( u[k+1][j][i][2]*u[k+1][j][i][3] ) * tmp2_jacu )
        - dt * tz1 * ( - c34 * tmp2_jacu * u[k+1][j][i][2] );
      cu[j][i][1][2] = 0.0;
      cu[j][i][2][2] = dt * tz2 * ( u[k+1][j][i][3] * tmp1_jacu )
        - dt * tz1 * ( c34 * tmp1_jacu )
        - dt * tz1 * dz3;
      cu[j][i][3][2] = dt * tz2 * ( u[k+1][j][i][2] * tmp1_jacu );
      cu[j][i][4][2] = 0.0;

      cu[j][i][0][3] = dt * tz2
        * ( - ( u[k+1][j][i][3] * tmp1_jacu ) * ( u[k+1][j][i][3] * tmp1_jacu )
            + C2 * ( qs[k+1][j][i] * tmp1_jacu ) )
        - dt * tz1 * ( - r43 * c34 * tmp2_jacu * u[k+1][j][i][3] );
      cu[j][i][1][3] = dt * tz2
        * ( - C2 * ( u[k+1][j][i][1] * tmp1_jacu ) );
      cu[j][i][2][3] = dt * tz2
        * ( - C2 * ( u[k+1][j][i][2] * tmp1_jacu ) );
      cu[j][i][3][3] = dt * tz2 * ( 2.0 - C2 )
        * ( u[k+1][j][i][3] * tmp1_jacu )
        - dt * tz1 * ( r43 * c34 * tmp1_jacu )
        - dt * tz1 * dz4;
      cu[j][i][4][3] = dt * tz2 * C2;

      cu[j][i][0][4] = dt * tz2
        * ( ( C2 * 2.0 * qs[k+1][j][i]
            - C1 * u[k+1][j][i][4] )
                 * ( u[k+1][j][i][3] * tmp2_jacu ) )
        - dt * tz1
        * ( - ( c34 - c1345 ) * tmp3_jacu * (u[k+1][j][i][1]*u[k+1][j][i][1])
            - ( c34 - c1345 ) * tmp3_jacu * (u[k+1][j][i][2]*u[k+1][j][i][2])
            - ( r43*c34 - c1345 )* tmp3_jacu * (u[k+1][j][i][3]*u[k+1][j][i][3])
            - c1345 * tmp2_jacu * u[k+1][j][i][4] );
      cu[j][i][1][4] = dt * tz2
        * ( - C2 * ( u[k+1][j][i][1]*u[k+1][j][i][3] ) * tmp2_jacu )
        - dt * tz1 * ( c34 - c1345 ) * tmp2_jacu * u[k+1][j][i][1];
      cu[j][i][2][4] = dt * tz2
        * ( - C2 * ( u[k+1][j][i][2]*u[k+1][j][i][3] ) * tmp2_jacu )
        - dt * tz1 * ( c34 - c1345 ) * tmp2_jacu * u[k+1][j][i][2];
      cu[j][i][3][4] = dt * tz2
        * ( C1 * ( u[k+1][j][i][4] * tmp1_jacu )
            - C2
            * ( qs[k+1][j][i] * tmp1_jacu
              + u[k+1][j][i][3]*u[k+1][j][i][3] * tmp2_jacu ) )
        - dt * tz1 * ( r43*c34 - c1345 ) * tmp2_jacu * u[k+1][j][i][3];
      cu[j][i][4][4] = dt * tz2
        * ( C1 * ( u[k+1][j][i][3] * tmp1_jacu ) )
        - dt * tz1 * c1345 * tmp1_jacu
        - dt * tz1 * dz5;
    }
  }
}

