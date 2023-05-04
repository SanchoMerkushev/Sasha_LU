#include <stdio.h>
#include "applu.incl"
#include "timers.h"



//---------------------------------------------------------------------
// to perform pseudo-time stepping SSOR iterations
// for five nonlinear pde's.
//---------------------------------------------------------------------
void ssor(int niter)
{
  //---------------------------------------------------------------------
  // local variables
  //---------------------------------------------------------------------
  int i, j, k, m, n, t;
  int istep;
  double tmp, tmp2, tv[ISIZ2][ISIZ1][5];
  double delunm[5];

  //---------------------------------------------------------------------
  // begin pseudo-time stepping iterations
  //---------------------------------------------------------------------
  tmp = 1.0 / ( omega * ( 2.0 - omega ) );

  //---------------------------------------------------------------------
  // initialize a,b,c,d to zero (guarantees that page tables have been
  // formed, if applicable on given architecture, before timestepping).
  //---------------------------------------------------------------------
  //#pragma omp parallel default(shared) private(m,n,i,j)
  {
  //#pragma omp for nowait
  for (j = jst; j < jend; j++) {
    for (i = ist; i < iend; i++) {
      for (n = 0; n < 5; n++) {
        for (m = 0; m < 5; m++) {
          a[j][i][n][m] = 0.0;
          b[j][i][n][m] = 0.0;
          c[j][i][n][m] = 0.0;
          d[j][i][n][m] = 0.0;
        }
      }
    }
  }
  //#pragma omp for nowait
  for (j = jend - 1; j >= jst; j--) {
    for (i = iend - 1; i >= ist; i--) {
      for (n = 0; n < 5; n++) {
        for (m = 0; m < 5; m++) {
          au[j][i][n][m] = 0.0;
          bu[j][i][n][m] = 0.0;
          cu[j][i][n][m] = 0.0;
          du[j][i][n][m] = 0.0;
        }
      }
    }
  }
  } //end parallel

  //---------------------------------------------------------------------
  // compute the steady-state residuals
  //---------------------------------------------------------------------
  rhs();

  //---------------------------------------------------------------------
  // compute the L2 norms of newton iteration residuals
  //---------------------------------------------------------------------
  l2norm( ISIZ1, ISIZ2, ISIZ3, nx0, ny0, nz0,
          ist, iend, jst, jend, rsd, rsdnm );

  //---------------------------------------------------------------------
  // the timestep loop
  //---------------------------------------------------------------------
  for (istep = 1; istep <= niter; istep++) {
    if ((istep % 20) == 0 || istep == itmax || istep == 1) {
      if (niter > 1) printf(" Time step %4d\n", istep);
    }
    //#pragma omp parallel default(shared) private(i,j,k,m,tmp2) \
                shared(ist,iend,jst,jend,nx,ny,nz,nx0,ny0,omega)
    
    { // start parallel
    //#pragma omp master
    tmp2 = dt;
    //#pragma omp for nowait
    #pragma acc parallel loop private(k, j, i, m)
    for (k = 1; k < nz - 1; k++) {
      for (j = jst; j < jend; j++) {
        for (i = ist; i < iend; i++) {
          for (m = 0; m < 5; m++) {
            rsd[k][j][i][m] = tmp2 * rsd[k][j][i][m];
          }
        }
      }
    }
    for (k = 1; k < nz -1; k++) { // start k_first
      // start jacld(k);
	  double r43;
	  double c1345;
	  double c34;
	  double tmp1, tmp2_jacld, tmp3;

	  r43 = ( 4.0 / 3.0 );
	  c1345 = C1 * C3 * C4 * C5;
	  c34 = C3 * C4;
	  //#pragma omp for schedule(static) nowait
	  #pragma acc  parallel loop private(j, i, tmp1, tmp2, tmp3)
	  for (j = jst; j < jend; j++) {
	    for (i = ist; i < iend; i++) {
	      //---------------------------------------------------------------------
	      // form the block daigonal
	      //---------------------------------------------------------------------
	      tmp1 = rho_i[k][j][i];
	      tmp2_jacld = tmp1 * tmp1;
	      tmp3 = tmp1 * tmp2_jacld;

	      d[j][i][0][0] =  1.0 + dt * 2.0 * ( tx1 * dx1 + ty1 * dy1 + tz1 * dz1 );
	      d[j][i][1][0] =  0.0;
	      d[j][i][2][0] =  0.0;
	      d[j][i][3][0] =  0.0;
	      d[j][i][4][0] =  0.0;

	      d[j][i][0][1] = -dt * 2.0
		* ( tx1 * r43 + ty1 + tz1 ) * c34 * tmp2_jacld * u[k][j][i][1];
	      d[j][i][1][1] =  1.0
		+ dt * 2.0 * c34 * tmp1 * ( tx1 * r43 + ty1 + tz1 )
		+ dt * 2.0 * ( tx1 * dx2 + ty1 * dy2 + tz1 * dz2 );
	      d[j][i][2][1] = 0.0;
	      d[j][i][3][1] = 0.0;
	      d[j][i][4][1] = 0.0;

	      d[j][i][0][2] = -dt * 2.0
		* ( tx1 + ty1 * r43 + tz1 ) * c34 * tmp2_jacld * u[k][j][i][2];
	      d[j][i][1][2] = 0.0;
	      d[j][i][2][2] = 1.0
		+ dt * 2.0 * c34 * tmp1 * ( tx1 + ty1 * r43 + tz1 )
		+ dt * 2.0 * ( tx1 * dx3 + ty1 * dy3 + tz1 * dz3 );
	      d[j][i][3][2] = 0.0;
	      d[j][i][4][2] = 0.0;

	      d[j][i][0][3] = -dt * 2.0
		* ( tx1 + ty1 + tz1 * r43 ) * c34 * tmp2_jacld * u[k][j][i][3];
	      d[j][i][1][3] = 0.0;
	      d[j][i][2][3] = 0.0;
	      d[j][i][3][3] = 1.0
		+ dt * 2.0 * c34 * tmp1 * ( tx1 + ty1 + tz1 * r43 )
		+ dt * 2.0 * ( tx1 * dx4 + ty1 * dy4 + tz1 * dz4 );
	      d[j][i][4][3] = 0.0;

	      d[j][i][0][4] = -dt * 2.0
		* ( ( ( tx1 * ( r43*c34 - c1345 )
		        + ty1 * ( c34 - c1345 )
		        + tz1 * ( c34 - c1345 ) ) * ( u[k][j][i][1]*u[k][j][i][1] )
		      + ( tx1 * ( c34 - c1345 )
		        + ty1 * ( r43*c34 - c1345 )
		        + tz1 * ( c34 - c1345 ) ) * ( u[k][j][i][2]*u[k][j][i][2] )
		      + ( tx1 * ( c34 - c1345 )
		        + ty1 * ( c34 - c1345 )
		        + tz1 * ( r43*c34 - c1345 ) ) * (u[k][j][i][3]*u[k][j][i][3])
		    ) * tmp3
		    + ( tx1 + ty1 + tz1 ) * c1345 * tmp2_jacld * u[k][j][i][4] );

	      d[j][i][1][4] = dt * 2.0 * tmp2_jacld * u[k][j][i][1]
		* ( tx1 * ( r43*c34 - c1345 )
		  + ty1 * (     c34 - c1345 )
		  + tz1 * (     c34 - c1345 ) );
	      d[j][i][2][4] = dt * 2.0 * tmp2_jacld * u[k][j][i][2]
		* ( tx1 * ( c34 - c1345 )
		  + ty1 * ( r43*c34 -c1345 )
		  + tz1 * ( c34 - c1345 ) );
	      d[j][i][3][4] = dt * 2.0 * tmp2_jacld * u[k][j][i][3]
		* ( tx1 * ( c34 - c1345 )
		  + ty1 * ( c34 - c1345 )
		  + tz1 * ( r43*c34 - c1345 ) );
	      d[j][i][4][4] = 1.0
		+ dt * 2.0 * ( tx1  + ty1 + tz1 ) * c1345 * tmp1
		+ dt * 2.0 * ( tx1 * dx5 +  ty1 * dy5 +  tz1 * dz5 );

	      //---------------------------------------------------------------------
	      // form the first block sub-diagonal
	      //---------------------------------------------------------------------
	      tmp1 = rho_i[k-1][j][i];
	      tmp2_jacld = tmp1 * tmp1;
	      tmp3 = tmp1 * tmp2_jacld;

	      a[j][i][0][0] = - dt * tz1 * dz1;
	      a[j][i][1][0] =   0.0;
	      a[j][i][2][0] =   0.0;
	      a[j][i][3][0] = - dt * tz2;
	      a[j][i][4][0] =   0.0;

	      a[j][i][0][1] = - dt * tz2
		* ( - ( u[k-1][j][i][1]*u[k-1][j][i][3] ) * tmp2_jacld )
		- dt * tz1 * ( - c34 * tmp2_jacld * u[k-1][j][i][1] );
	      a[j][i][1][1] = - dt * tz2 * ( u[k-1][j][i][3] * tmp1 )
		- dt * tz1 * c34 * tmp1
		- dt * tz1 * dz2;
	      a[j][i][2][1] = 0.0;
	      a[j][i][3][1] = - dt * tz2 * ( u[k-1][j][i][1] * tmp1 );
	      a[j][i][4][1] = 0.0;

	      a[j][i][0][2] = - dt * tz2
		* ( - ( u[k-1][j][i][2]*u[k-1][j][i][3] ) * tmp2_jacld )
		- dt * tz1 * ( - c34 * tmp2_jacld * u[k-1][j][i][2] );
	      a[j][i][1][2] = 0.0;
	      a[j][i][2][2] = - dt * tz2 * ( u[k-1][j][i][3] * tmp1 )
		- dt * tz1 * ( c34 * tmp1 )
		- dt * tz1 * dz3;
	      a[j][i][3][2] = - dt * tz2 * ( u[k-1][j][i][2] * tmp1 );
	      a[j][i][4][2] = 0.0;

	      a[j][i][0][3] = - dt * tz2
		* ( - ( u[k-1][j][i][3] * tmp1 ) * ( u[k-1][j][i][3] * tmp1 )
		    + C2 * qs[k-1][j][i] * tmp1 )
		- dt * tz1 * ( - r43 * c34 * tmp2_jacld * u[k-1][j][i][3] );
	      a[j][i][1][3] = - dt * tz2
		* ( - C2 * ( u[k-1][j][i][1] * tmp1 ) );
	      a[j][i][2][3] = - dt * tz2
		* ( - C2 * ( u[k-1][j][i][2] * tmp1 ) );
	      a[j][i][3][3] = - dt * tz2 * ( 2.0 - C2 )
		* ( u[k-1][j][i][3] * tmp1 )
		- dt * tz1 * ( r43 * c34 * tmp1 )
		- dt * tz1 * dz4;
	      a[j][i][4][3] = - dt * tz2 * C2;

	      a[j][i][0][4] = - dt * tz2
		* ( ( C2 * 2.0 * qs[k-1][j][i] - C1 * u[k-1][j][i][4] )
		    * u[k-1][j][i][3] * tmp2_jacld )
		- dt * tz1
		* ( - ( c34 - c1345 ) * tmp3 * (u[k-1][j][i][1]*u[k-1][j][i][1])
		    - ( c34 - c1345 ) * tmp3 * (u[k-1][j][i][2]*u[k-1][j][i][2])
		    - ( r43*c34 - c1345 )* tmp3 * (u[k-1][j][i][3]*u[k-1][j][i][3])
		    - c1345 * tmp2_jacld * u[k-1][j][i][4] );
	      a[j][i][1][4] = - dt * tz2
		* ( - C2 * ( u[k-1][j][i][1]*u[k-1][j][i][3] ) * tmp2_jacld )
		- dt * tz1 * ( c34 - c1345 ) * tmp2_jacld * u[k-1][j][i][1];
	      a[j][i][2][4] = - dt * tz2
		* ( - C2 * ( u[k-1][j][i][2]*u[k-1][j][i][3] ) * tmp2_jacld )
		- dt * tz1 * ( c34 - c1345 ) * tmp2_jacld * u[k-1][j][i][2];
	      a[j][i][3][4] = - dt * tz2
		* ( C1 * ( u[k-1][j][i][4] * tmp1 )
		  - C2 * ( qs[k-1][j][i] * tmp1
		         + u[k-1][j][i][3]*u[k-1][j][i][3] * tmp2_jacld ) )
		- dt * tz1 * ( r43*c34 - c1345 ) * tmp2_jacld * u[k-1][j][i][3];
	      a[j][i][4][4] = - dt * tz2
		* ( C1 * ( u[k-1][j][i][3] * tmp1 ) )
		- dt * tz1 * c1345 * tmp1
		- dt * tz1 * dz5;

	      //---------------------------------------------------------------------
	      // form the second block sub-diagonal
	      //---------------------------------------------------------------------
	      tmp1 = rho_i[k][j-1][i];
	      tmp2_jacld = tmp1 * tmp1;
	      tmp3 = tmp1 * tmp2_jacld;

	      b[j][i][0][0] = - dt * ty1 * dy1;
	      b[j][i][1][0] =   0.0;
	      b[j][i][2][0] = - dt * ty2;
	      b[j][i][3][0] =   0.0;
	      b[j][i][4][0] =   0.0;

	      b[j][i][0][1] = - dt * ty2
		* ( - ( u[k][j-1][i][1]*u[k][j-1][i][2] ) * tmp2_jacld )
		- dt * ty1 * ( - c34 * tmp2_jacld * u[k][j-1][i][1] );
	      b[j][i][1][1] = - dt * ty2 * ( u[k][j-1][i][2] * tmp1 )
		- dt * ty1 * ( c34 * tmp1 )
		- dt * ty1 * dy2;
	      b[j][i][2][1] = - dt * ty2 * ( u[k][j-1][i][1] * tmp1 );
	      b[j][i][3][1] = 0.0;
	      b[j][i][4][1] = 0.0;

	      b[j][i][0][2] = - dt * ty2
		* ( - ( u[k][j-1][i][2] * tmp1 ) * ( u[k][j-1][i][2] * tmp1 )
		    + C2 * ( qs[k][j-1][i] * tmp1 ) )
		- dt * ty1 * ( - r43 * c34 * tmp2_jacld * u[k][j-1][i][2] );
	      b[j][i][1][2] = - dt * ty2
		* ( - C2 * ( u[k][j-1][i][1] * tmp1 ) );
	      b[j][i][2][2] = - dt * ty2 * ( (2.0 - C2) * (u[k][j-1][i][2] * tmp1) )
		- dt * ty1 * ( r43 * c34 * tmp1 )
		- dt * ty1 * dy3;
	      b[j][i][3][2] = - dt * ty2 * ( - C2 * ( u[k][j-1][i][3] * tmp1 ) );
	      b[j][i][4][2] = - dt * ty2 * C2;

	      b[j][i][0][3] = - dt * ty2
		* ( - ( u[k][j-1][i][2]*u[k][j-1][i][3] ) * tmp2_jacld )
		- dt * ty1 * ( - c34 * tmp2_jacld * u[k][j-1][i][3] );
	      b[j][i][1][3] = 0.0;
	      b[j][i][2][3] = - dt * ty2 * ( u[k][j-1][i][3] * tmp1 );
	      b[j][i][3][3] = - dt * ty2 * ( u[k][j-1][i][2] * tmp1 )
		- dt * ty1 * ( c34 * tmp1 )
		- dt * ty1 * dy4;
	      b[j][i][4][3] = 0.0;

	      b[j][i][0][4] = - dt * ty2
		* ( ( C2 * 2.0 * qs[k][j-1][i] - C1 * u[k][j-1][i][4] )
		    * ( u[k][j-1][i][2] * tmp2_jacld ) )
		- dt * ty1
		* ( - (     c34 - c1345 )*tmp3*(u[k][j-1][i][1]*u[k][j-1][i][1])
		    - ( r43*c34 - c1345 )*tmp3*(u[k][j-1][i][2]*u[k][j-1][i][2])
		    - (     c34 - c1345 )*tmp3*(u[k][j-1][i][3]*u[k][j-1][i][3])
		    - c1345*tmp2_jacld*u[k][j-1][i][4] );
	      b[j][i][1][4] = - dt * ty2
		* ( - C2 * ( u[k][j-1][i][1]*u[k][j-1][i][2] ) * tmp2_jacld )
		- dt * ty1 * ( c34 - c1345 ) * tmp2_jacld * u[k][j-1][i][1];
	      b[j][i][2][4] = - dt * ty2
		* ( C1 * ( u[k][j-1][i][4] * tmp1 )
		  - C2 * ( qs[k][j-1][i] * tmp1
		         + u[k][j-1][i][2]*u[k][j-1][i][2] * tmp2_jacld ) )
		- dt * ty1 * ( r43*c34 - c1345 ) * tmp2_jacld * u[k][j-1][i][2];
	      b[j][i][3][4] = - dt * ty2
		* ( - C2 * ( u[k][j-1][i][2]*u[k][j-1][i][3] ) * tmp2_jacld )
		- dt * ty1 * ( c34 - c1345 ) * tmp2_jacld * u[k][j-1][i][3];
	      b[j][i][4][4] = - dt * ty2
		* ( C1 * ( u[k][j-1][i][2] * tmp1 ) )
		- dt * ty1 * c1345 * tmp1
		- dt * ty1 * dy5;

	      //---------------------------------------------------------------------
	      // form the third block sub-diagonal
	      //---------------------------------------------------------------------
	      tmp1 = rho_i[k][j][i-1];
	      tmp2_jacld = tmp1 * tmp1;
	      tmp3 = tmp1 * tmp2_jacld;

	      c[j][i][0][0] = - dt * tx1 * dx1;
	      c[j][i][1][0] = - dt * tx2;
	      c[j][i][2][0] =   0.0;
	      c[j][i][3][0] =   0.0;
	      c[j][i][4][0] =   0.0;

	      c[j][i][0][1] = - dt * tx2
		* ( - ( u[k][j][i-1][1] * tmp1 ) * ( u[k][j][i-1][1] * tmp1 )
		    + C2 * qs[k][j][i-1] * tmp1 )
		- dt * tx1 * ( - r43 * c34 * tmp2_jacld * u[k][j][i-1][1] );
	      c[j][i][1][1] = - dt * tx2
		* ( ( 2.0 - C2 ) * ( u[k][j][i-1][1] * tmp1 ) )
		- dt * tx1 * ( r43 * c34 * tmp1 )
		- dt * tx1 * dx2;
	      c[j][i][2][1] = - dt * tx2
		* ( - C2 * ( u[k][j][i-1][2] * tmp1 ) );
	      c[j][i][3][1] = - dt * tx2
		* ( - C2 * ( u[k][j][i-1][3] * tmp1 ) );
	      c[j][i][4][1] = - dt * tx2 * C2;

	      c[j][i][0][2] = - dt * tx2
		* ( - ( u[k][j][i-1][1] * u[k][j][i-1][2] ) * tmp2_jacld )
		- dt * tx1 * ( - c34 * tmp2_jacld * u[k][j][i-1][2] );
	      c[j][i][1][2] = - dt * tx2 * ( u[k][j][i-1][2] * tmp1 );
	      c[j][i][2][2] = - dt * tx2 * ( u[k][j][i-1][1] * tmp1 )
		- dt * tx1 * ( c34 * tmp1 )
		- dt * tx1 * dx3;
	      c[j][i][3][2] = 0.0;
	      c[j][i][4][2] = 0.0;

	      c[j][i][0][3] = - dt * tx2
		* ( - ( u[k][j][i-1][1]*u[k][j][i-1][3] ) * tmp2_jacld )
		- dt * tx1 * ( - c34 * tmp2_jacld * u[k][j][i-1][3] );
	      c[j][i][1][3] = - dt * tx2 * ( u[k][j][i-1][3] * tmp1 );
	      c[j][i][2][3] = 0.0;
	      c[j][i][3][3] = - dt * tx2 * ( u[k][j][i-1][1] * tmp1 )
		- dt * tx1 * ( c34 * tmp1 ) - dt * tx1 * dx4;
	      c[j][i][4][3] = 0.0;

	      c[j][i][0][4] = - dt * tx2
		* ( ( C2 * 2.0 * qs[k][j][i-1] - C1 * u[k][j][i-1][4] )
		    * u[k][j][i-1][1] * tmp2_jacld )
		- dt * tx1
		* ( - ( r43*c34 - c1345 ) * tmp3 * ( u[k][j][i-1][1]*u[k][j][i-1][1] )
		    - (     c34 - c1345 ) * tmp3 * ( u[k][j][i-1][2]*u[k][j][i-1][2] )
		    - (     c34 - c1345 ) * tmp3 * ( u[k][j][i-1][3]*u[k][j][i-1][3] )
		    - c1345 * tmp2_jacld * u[k][j][i-1][4] );
	      c[j][i][1][4] = - dt * tx2
		* ( C1 * ( u[k][j][i-1][4] * tmp1 )
		  - C2 * ( u[k][j][i-1][1]*u[k][j][i-1][1] * tmp2_jacld
		         + qs[k][j][i-1] * tmp1 ) )
		- dt * tx1 * ( r43*c34 - c1345 ) * tmp2_jacld * u[k][j][i-1][1];
	      c[j][i][2][4] = - dt * tx2
		* ( - C2 * ( u[k][j][i-1][2]*u[k][j][i-1][1] ) * tmp2_jacld )
		- dt * tx1 * (  c34 - c1345 ) * tmp2_jacld * u[k][j][i-1][2];
	      c[j][i][3][4] = - dt * tx2
		* ( - C2 * ( u[k][j][i-1][3]*u[k][j][i-1][1] ) * tmp2_jacld )
		- dt * tx1 * (  c34 - c1345 ) * tmp2_jacld * u[k][j][i-1][3];
	      c[j][i][4][4] = - dt * tx2
		* ( C1 * ( u[k][j][i-1][1] * tmp1 ) )
		- dt * tx1 * c1345 * tmp1
		- dt * tx1 * dx5;
	      }
	    }
      // end jacld(k);
      // start blts( ISIZ1, ISIZ2, ISIZ3, nx, ny, nz, k, omega, rsd, a, b, c, d, ist, iend, jst, jend, nx0, ny0 );
	  int diag;
	  double tmp_blts, tmp1_blts;  
	  double tmat_blts[ISIZ1][5][5], tv_blts[ISIZ1][5];

	  //double (*vk)[ldmx/2*2+1][5] = rsd[k];
	  //double (*vkm1)[ldmx/2*2+1][5] = rsd[k-1];


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
	    //#pragma acc parallel loop independent private(t,i, j, m, tmp_blts, tmp1_blts)
	    //#pragma acc parallel loop independent gang
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
	    //#pragma acc parallel loop independent private(t, i, j, m, tmp_blts, tmp1_blts)
	    //#pragma acc parallel loop independent gang
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
      // end blts( ISIZ1, ISIZ2, ISIZ3, nx, ny, nz, k, omega, rsd, a, b, c, d, ist, iend, jst, jend, nx0, ny0 );
    } // end k_first
    for (k = nz - 2; k > 0; k--) { // start k_second
      // start jacu(k);
	  double r43;
	  double c1345;
	  double c34;
	  double tmp1_jacu, tmp2_jacu, tmp3_jacu;

	  r43 = ( 4.0 / 3.0 );
	  c1345 = C1 * C3 * C4 * C5;
	  c34 = C3 * C4;

	  //#pragma omp for schedule(static) nowait
	  #pragma acc parallel loop private(j, i, tmp1_jacu, tmp2_jacu, tmp3_jacu)
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
      // end jacu(k);
      // start buts( ISIZ1, ISIZ2, ISIZ3, nx, ny, nz, k, omega, rsd, tv, du, au, bu, cu, ist, iend, jst, jend, nx0, ny0 );
	  int diag;
	  double tmp_buts, tmp1_buts;
	  double tmat_buts[ISIZ1][5][5];
	  #pragma acc parallel loop private(i, j, m)
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
	    //#pragma acc parallel loop independent private(i, j, m, tmp_buts, tmp1_buts, t)
	    //#pragma acc parallel loop independent gang
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
	    //#pragma acc parallel loop independent private(i, j, m, tmp_buts, tmp1_buts, t)
	    //#pragma acc parallel loop independent gang
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
      // end buts( ISIZ1, ISIZ2, ISIZ3, nx, ny, nz, k, omega, rsd, tv, du, au, bu, cu, ist, iend, jst, jend, nx0, ny0 );
    } // end k_second
    tmp2 = tmp;
    for (k = 1; k < nz-1; k++) { // start k_third
      for (j = jst; j < jend; j++) {
        for (i = ist; i < iend; i++) {
          for (m = 0; m < 5; m++) {
            u[k][j][i][m] = u[k][j][i][m] + tmp2 * rsd[k][j][i][m];
          }
        }
      }
    } // end k_third
    } //end parallel

    //---------------------------------------------------------------------
    // compute the max-norms of newton iteration corrections
    //---------------------------------------------------------------------
    if ( (istep % inorm) == 0 ) {
      l2norm( ISIZ1, ISIZ2, ISIZ3, nx0, ny0, nz0,
              ist, iend, jst, jend,
              rsd, delunm );
      /*
      if ( ipr == 1 ) {
        printf(" \n RMS-norm of SSOR-iteration correction "
               "for first pde  = %12.5E\n"
               " RMS-norm of SSOR-iteration correction "
               "for second pde = %12.5E\n"
               " RMS-norm of SSOR-iteration correction "
               "for third pde  = %12.5E\n"
               " RMS-norm of SSOR-iteration correction "
               "for fourth pde = %12.5E\n",
               " RMS-norm of SSOR-iteration correction "
               "for fifth pde  = %12.5E\n", 
               delunm[0], delunm[1], delunm[2], delunm[3], delunm[4]); 
      } else if ( ipr == 2 ) {
        printf("(%5d,%15.6f)\n", istep, delunm[4]);
      }
      */
    }
 
    //---------------------------------------------------------------------
    // compute the steady-state residuals
    //---------------------------------------------------------------------
    rhs();
 
    //---------------------------------------------------------------------
    // compute the max-norms of newton iteration residuals
    //---------------------------------------------------------------------
    if ( ((istep % inorm ) == 0 ) || ( istep == itmax ) ) {
      l2norm( ISIZ1, ISIZ2, ISIZ3, nx0, ny0, nz0,
              ist, iend, jst, jend, rsd, rsdnm );
      /*
      if ( ipr == 1 ) {
        printf(" \n RMS-norm of steady-state residual for "
               "first pde  = %12.5E\n"
               " RMS-norm of steady-state residual for "
               "second pde = %12.5E\n"
               " RMS-norm of steady-state residual for "
               "third pde  = %12.5E\n"
               " RMS-norm of steady-state residual for "
               "fourth pde = %12.5E\n"
               " RMS-norm of steady-state residual for "
               "fifth pde  = %12.5E\n", 
               rsdnm[0], rsdnm[1], rsdnm[2], rsdnm[3], rsdnm[4]);
      }
      */
    }

    //---------------------------------------------------------------------
    // check the newton-iteration residuals against the tolerance levels
    //---------------------------------------------------------------------
    if ( ( rsdnm[0] < tolrsd[0] ) && ( rsdnm[1] < tolrsd[1] ) &&
         ( rsdnm[2] < tolrsd[2] ) && ( rsdnm[3] < tolrsd[3] ) &&
         ( rsdnm[4] < tolrsd[4] ) ) {
      //if (ipr == 1 ) {
      printf(" \n convergence was achieved after %4d pseudo-time steps\n",
          istep);
      //}
      break;
    }
  } // end iter
  timer_stop(1);
  maxtime = timer_read(1);
}

