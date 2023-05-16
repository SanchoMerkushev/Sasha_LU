#include <stdio.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "applu.incl"
#include "timers.h"
#include <math.h>

//---------------------------------------------------------------------
// Thread synchronization for pipeline operation
//---------------------------------------------------------------------

//---------------------------------------------------------------------
// to perform pseudo-time stepping SSOR iterations
// for five nonlinear pde's.
//---------------------------------------------------------------------
void ssor(int niter)
{
  //---------------------------------------------------------------------
  // local variables
  //---------------------------------------------------------------------
  int i, j, k, m, n, pl;
  int istep;
  double tmp, tmp2;
  int k_start, k_end, j_start, j_end;
  double delunm[5];
  double tmat_blts[ISIZ1][ISIZ1][5][5], tv_blts[ISIZ1][ISIZ1][5];
  double tmat_buts[ISIZ1][ISIZ1][5][5], tv_buts[ISIZ2][ISIZ1][5];
  double tmp_buts, tmp1_buts;
  double r43;
  double c1345;
  double c34;
  r43 = ( 4.0 / 3.0 );
  c1345 = C1 * C3 * C4 * C5;
  c34 = C3 * C4;
  double tmp1, tmp2_jacld, tmp3;
  double tmp1_jacu, tmp2_jacu, tmp3_jacu;
  //int diag, t;
  double tmp_blts, tmp1_blts;
  //---------------------------------------------------------------------
  // begin pseudo-time stepping iterations
  //---------------------------------------------------------------------
  tmp = 1.0 / ( omega * ( 2.0 - omega ) );
  //---------------------------------------------------------------------
  // initialize a,b,c,d to zero (guarantees that page tables have been
  // formed, if applicable on given architecture, before timestepping).
  //---------------------------------------------------------------------
  //#pragma omp parallel default(shared) private(m,n,i,j)
  //#pragma omp for nowait

  for (i = 1; i <= t_last; i++) {
    timer_clear(i);
  }

  //---------------------------------------------------------------------
  // compute the steady-state residuals
  //---------------------------------------------------------------------
  rhs();

  //---------------------------------------------------------------------
  // compute the L2 norms of newton iteration residuals
  //---------------------------------------------------------------------
  l2norm( ISIZ1, ISIZ2, ISIZ3, nx0, ny0, nz0,
          ist, iend, jst, jend, rsd, rsdnm );

  for (i = 1; i <= t_last; i++) {
    timer_clear(i);
  }
  timer_start(1);

  //---------------------------------------------------------------------
  // the timestep loop
  //---------------------------------------------------------------------
  #pragma acc data copyin(frct[:ISIZ3][:ISIZ2/2*2+1][:ISIZ1/2*2+1][:5], flux [:ISIZ1][:5], qs[:ISIZ3][:ISIZ2/2*2+1][:ISIZ1/2*2+1], rho_i[:ISIZ3][:ISIZ2/2*2+1][:ISIZ1/2*2+1], tmat_blts[:ISIZ1][:ISIZ1][:5][:5], tv_blts[:ISIZ1][:ISIZ1][:5], tmat_buts[:ISIZ1][:ISIZ1][:5][:5], tv_buts[:ISIZ2][:ISIZ1][:5], delunm[:5]) copy(rsd[:ISIZ3][:ISIZ2/2*2+1][:ISIZ1/2*2+1][:5], u[:ISIZ3][:ISIZ2/2*2+1][:ISIZ1/2*2+1][:5]) create(a[:ISIZ2][:ISIZ1/2*2+1][:5][:5], b[:ISIZ2][:ISIZ1/2*2+1][:5][:5], c[:ISIZ2][:ISIZ1/2*2+1][:5][:5], d[:ISIZ2][:ISIZ1/2*2+1][:5][:5], au[:ISIZ2][:ISIZ1/2*2+1][:5][:5], bu[:ISIZ2][:ISIZ1/2*2+1][:5][:5], cu[:ISIZ2][:ISIZ1/2*2+1][:5][:5], du[:ISIZ2][:ISIZ1/2*2+1][:5][:5])
  { // DATA START
    #pragma acc parallel private(k,m,n,i,j)
    for (j = 0; j < ISIZ1; j++) {
      #pragma acc loop
      for (i = 0; i < ISIZ1/2*2+1; i++) {
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
    #pragma acc parallel private(k,m,n,i,j)
    for (j = 0; j < ISIZ1; j++) {
      #pragma acc loop
      for (i = 0; i < ISIZ1/2*2+1; i++) {
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
  //#pragma acc parallel
  //{ // PARALLEL START
  for (istep = 1; istep <= niter; istep++) {
    if ((istep % 20) == 0 || istep == itmax || istep == 1) {
      if (niter > 1) printf(" Time step %4d\n", istep);
    }

    //---------------------------------------------------------------------
    // perform SSOR iteration
    //---------------------------------------------------------------------
    //#pragma omp parallel default(shared) private(i,j,k,m,tmp2) \
                shared(ist,iend,jst,jend,nx,ny,nz,nx0,ny0,omega)
    {
    //#pragma omp master
    tmp2 = dt;
    #pragma acc parallel loop private(k, j, i)
    for (k = 1; k < nz - 1; k++) {
      #pragma acc loop
      for (j = jst; j < jend; j++) {
        for (i = ist; i < iend; i++) {
          for (m = 0; m < 5; m++) {
            rsd[k][j][i][m] = tmp2 * rsd[k][j][i][m];
          }
        }
      }
    }
    //#pragma omp master
    //#pragma omp barrier
    for (pl = 3; pl <= (ISIZ1 - 2) * 3; pl++) {
      //---------------------------------------------------------------------
      // form the lower triangular part of the jacobian matrix
      //---------------------------------------------------------------------
      // start jacld(k);
          k_start = max(1, pl - ISIZ1 + 2 - ISIZ1 + 2);
          k_end = min(pl - 1, ISIZ1 - 1);
	  #pragma acc parallel loop private(k, j_start, j_end, k_start, k_end, j, i, tmp_blts, tmp1_blts, tmp1, tmp2_jacld, tmp3)
	  for (k = k_start; k < k_end; k++) {
	     j_start = max(1, pl - k - ISIZ1 + 2);
	     j_end = min(pl - k, ISIZ1 - 1);
	    #pragma acc loop private(k_start, k_end, j, i, tmp_blts, tmp1_blts, tmp1, tmp2_jacld, tmp3)
	    for (j = j_start; j < j_end; j++) {
	      i = pl - k - j;
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
		// end jacld(k);
		// start blts
		for (m = 0; m < 5; m++) {
		rsd[k][j][i][m] =  rsd[k][j][i][m]
		  - omega * (  a[j][i][0][m] * rsd[k-1][j][i][0]
		             + a[j][i][1][m] * rsd[k-1][j][i][1]
		             + a[j][i][2][m] * rsd[k-1][j][i][2]
		             + a[j][i][3][m] * rsd[k-1][j][i][3]
		             + a[j][i][4][m] * rsd[k-1][j][i][4] );
	      }
	      for (m = 0; m < 5; m++) {
		tv_blts[k][j][m] =  rsd[k][j][i][m]
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
		tmat_blts[k][j][m][0] = d[j][i][0][m];
		tmat_blts[k][j][m][1] = d[j][i][1][m];
		tmat_blts[k][j][m][2] = d[j][i][2][m];
		tmat_blts[k][j][m][3] = d[j][i][3][m];
		tmat_blts[k][j][m][4] = d[j][i][4][m];
	      }

	      tmp1_blts = 1.0 / tmat_blts[k][j][0][0];
	      tmp_blts = tmp1_blts * tmat_blts[k][j][1][0];
	      tmat_blts[k][j][1][1] =  tmat_blts[k][j][1][1] - tmp_blts * tmat_blts[k][j][0][1];
	      tmat_blts[k][j][1][2] =  tmat_blts[k][j][1][2] - tmp_blts * tmat_blts[k][j][0][2];
	      tmat_blts[k][j][1][3] =  tmat_blts[k][j][1][3] - tmp_blts * tmat_blts[k][j][0][3];
	      tmat_blts[k][j][1][4] =  tmat_blts[k][j][1][4] - tmp_blts * tmat_blts[k][j][0][4];
	      tv_blts[k][j][1] = tv_blts[k][j][1] - tv_blts[k][j][0] * tmp_blts;

	      tmp_blts = tmp1_blts * tmat_blts[k][j][2][0];
	      tmat_blts[k][j][2][1] =  tmat_blts[k][j][2][1] - tmp_blts * tmat_blts[k][j][0][1];
	      tmat_blts[k][j][2][2] =  tmat_blts[k][j][2][2] - tmp_blts * tmat_blts[k][j][0][2];
	      tmat_blts[k][j][2][3] =  tmat_blts[k][j][2][3] - tmp_blts * tmat_blts[k][j][0][3];
	      tmat_blts[k][j][2][4] =  tmat_blts[k][j][2][4] - tmp_blts * tmat_blts[k][j][0][4];
	      tv_blts[k][j][2] = tv_blts[k][j][2] - tv_blts[k][j][0] * tmp_blts;

	      tmp_blts = tmp1_blts * tmat_blts[k][j][3][0];
	      tmat_blts[k][j][3][1] =  tmat_blts[k][j][3][1] - tmp_blts * tmat_blts[k][j][0][1];
	      tmat_blts[k][j][3][2] =  tmat_blts[k][j][3][2] - tmp_blts * tmat_blts[k][j][0][2];
	      tmat_blts[k][j][3][3] =  tmat_blts[k][j][3][3] - tmp_blts * tmat_blts[k][j][0][3];
	      tmat_blts[k][j][3][4] =  tmat_blts[k][j][3][4] - tmp_blts * tmat_blts[k][j][0][4];
	      tv_blts[k][j][3] = tv_blts[k][j][3] - tv_blts[k][j][0] * tmp_blts;

	      tmp_blts = tmp1_blts * tmat_blts[k][j][4][0];
	      tmat_blts[k][j][4][1] =  tmat_blts[k][j][4][1] - tmp_blts * tmat_blts[k][j][0][1];
	      tmat_blts[k][j][4][2] =  tmat_blts[k][j][4][2] - tmp_blts * tmat_blts[k][j][0][2];
	      tmat_blts[k][j][4][3] =  tmat_blts[k][j][4][3] - tmp_blts * tmat_blts[k][j][0][3];
	      tmat_blts[k][j][4][4] =  tmat_blts[k][j][4][4] - tmp_blts * tmat_blts[k][j][0][4];
	      tv_blts[k][j][4] = tv_blts[k][j][4] - tv_blts[k][j][0] * tmp_blts;

	      tmp1_blts = 1.0 / tmat_blts[k][j][1][1];
	      tmp_blts = tmp1_blts * tmat_blts[k][j][2][1];
	      tmat_blts[k][j][2][2] =  tmat_blts[k][j][2][2] - tmp_blts * tmat_blts[k][j][1][2];
	      tmat_blts[k][j][2][3] =  tmat_blts[k][j][2][3] - tmp_blts * tmat_blts[k][j][1][3];
	      tmat_blts[k][j][2][4] =  tmat_blts[k][j][2][4] - tmp_blts * tmat_blts[k][j][1][4];
	      tv_blts[k][j][2] = tv_blts[k][j][2] - tv_blts[k][j][1] * tmp_blts;

	      tmp_blts = tmp1_blts * tmat_blts[k][j][3][1];
	      tmat_blts[k][j][3][2] =  tmat_blts[k][j][3][2] - tmp_blts * tmat_blts[k][j][1][2];
	      tmat_blts[k][j][3][3] =  tmat_blts[k][j][3][3] - tmp_blts * tmat_blts[k][j][1][3];
	      tmat_blts[k][j][3][4] =  tmat_blts[k][j][3][4] - tmp_blts * tmat_blts[k][j][1][4];
	      tv_blts[k][j][3] = tv_blts[k][j][3] - tv_blts[k][j][1] * tmp_blts;

	      tmp_blts = tmp1_blts * tmat_blts[k][j][4][1];
	      tmat_blts[k][j][4][2] =  tmat_blts[k][j][4][2] - tmp_blts * tmat_blts[k][j][1][2];
	      tmat_blts[k][j][4][3] =  tmat_blts[k][j][4][3] - tmp_blts * tmat_blts[k][j][1][3];
	      tmat_blts[k][j][4][4] =  tmat_blts[k][j][4][4] - tmp_blts * tmat_blts[k][j][1][4];
	      tv_blts[k][j][4] = tv_blts[k][j][4] - tv_blts[k][j][1] * tmp_blts;

	      tmp1_blts = 1.0 / tmat_blts[k][j][2][2];
	      tmp_blts = tmp1_blts * tmat_blts[k][j][3][2];
	      tmat_blts[k][j][3][3] =  tmat_blts[k][j][3][3] - tmp_blts * tmat_blts[k][j][2][3];
	      tmat_blts[k][j][3][4] =  tmat_blts[k][j][3][4] - tmp_blts * tmat_blts[k][j][2][4];
	      tv_blts[k][j][3] = tv_blts[k][j][3] - tv_blts[k][j][2] * tmp_blts;

	      tmp_blts = tmp1_blts * tmat_blts[k][j][4][2];
	      tmat_blts[k][j][4][3] =  tmat_blts[k][j][4][3] - tmp_blts * tmat_blts[k][j][2][3];
	      tmat_blts[k][j][4][4] =  tmat_blts[k][j][4][4] - tmp_blts * tmat_blts[k][j][2][4];
	      tv_blts[k][j][4] = tv_blts[k][j][4] - tv_blts[k][j][2] * tmp_blts;

	      tmp1_blts = 1.0 / tmat_blts[k][j][3][3];
	      tmp_blts = tmp1_blts * tmat_blts[k][j][4][3];
	      tmat_blts[k][j][4][4] =  tmat_blts[k][j][4][4] - tmp_blts * tmat_blts[k][j][3][4];
	      tv_blts[k][j][4] = tv_blts[k][j][4] - tv_blts[k][j][3] * tmp_blts;

	      //---------------------------------------------------------------------
	      // back substitution
	      //---------------------------------------------------------------------
	      rsd[k][j][i][4] = tv_blts[k][j][4] / tmat_blts[k][j][4][4];

	      tv_blts[k][j][3] = tv_blts[k][j][3] 
		- tmat_blts[k][j][3][4] * rsd[k][j][i][4];
	      rsd[k][j][i][3] = tv_blts[k][j][3] / tmat_blts[k][j][3][3];

	      tv_blts[k][j][2] = tv_blts[k][j][2]
		- tmat_blts[k][j][2][3] * rsd[k][j][i][3]
		- tmat_blts[k][j][2][4] * rsd[k][j][i][4];
	      rsd[k][j][i][2] = tv_blts[k][j][2] / tmat_blts[k][j][2][2];

	      tv_blts[k][j][1] = tv_blts[k][j][1]
		- tmat_blts[k][j][1][2] * rsd[k][j][i][2]
		- tmat_blts[k][j][1][3] * rsd[k][j][i][3]
		- tmat_blts[k][j][1][4] * rsd[k][j][i][4];
	      rsd[k][j][i][1] = tv_blts[k][j][1] / tmat_blts[k][j][1][1];

	      tv_blts[k][j][0] = tv_blts[k][j][0]
		- tmat_blts[k][j][0][1] * rsd[k][j][i][1]
		- tmat_blts[k][j][0][2] * rsd[k][j][i][2]
		- tmat_blts[k][j][0][3] * rsd[k][j][i][3]
		- tmat_blts[k][j][0][4] * rsd[k][j][i][4];
	      rsd[k][j][i][0] = tv_blts[k][j][0] / tmat_blts[k][j][0][0];
	      //if ((k == 1) && (j == 1) && (i == 1))
	           //printf("BBB %f %f %f\n", rsd[1][1][1][3], rsd[1][1][0][0], rsd[1][0][1][4]);
	      //end blts
	    } // end i
	  } // end j

    } // end pl first
    
    
    
    //#pragma omp barrier
    //#pragma acc parallel loop private(k)
    for (pl = 3 * (ISIZ1 - 2); pl >= 3; pl--) {
      //---------------------------------------------------------------------
      // form the strictly upper triangular part of the jacobian matrix
      //---------------------------------------------------------------------
      // start jacu(k)
	  //#pragma omp for schedule(static) nowait
	  //for (j = jend - 1; j >= jst; j--) {
	    //for (i = iend - 1; i >= ist; i--) {
	    k_start = min(ISIZ1 - 2, pl - 2);
	    k_end = max(1, pl - ISIZ1 + 2 - ISIZ1 + 2);
	    #pragma acc parallel loop private(k, j_start, j_end, k_start, k_end, j, i, tmp1_jacu, tmp2_jacu, tmp3_jacu, tmp_buts, tmp1_buts)
	    for (k = k_start; k >= k_end; k--) {
	      j_start = min(ISIZ1 - 2, pl - k - 1);
	      j_end = max(1, pl - k - ISIZ1 + 2);
	      #pragma acc loop private(k_start, k_end,  i, tmp1_jacu, tmp2_jacu, tmp3_jacu, tmp_buts, tmp1_buts)
	      for (j = j_start; j >= j_end; j--) {
	      i = pl - k - j;
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
	       // end jacu(k)
	       // start buts
	       for (m = 0; m < 5; m++) {
		tv_buts[j][i][m] = 
		  omega * (  cu[j][i][0][m] * rsd[k+1][j][i][0]
		           + cu[j][i][1][m] * rsd[k+1][j][i][1]
		           + cu[j][i][2][m] * rsd[k+1][j][i][2]
		           + cu[j][i][3][m] * rsd[k+1][j][i][3]
		           + cu[j][i][4][m] * rsd[k+1][j][i][4] );
	      }
	for (m = 0; m < 5; m++) {
		tv_buts[j][i][m] = tv_buts[j][i][m]
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
		tmat_buts[k][j][m][0] = du[j][i][0][m];
		tmat_buts[k][j][m][1] = du[j][i][1][m];
		tmat_buts[k][j][m][2] = du[j][i][2][m];
		tmat_buts[k][j][m][3] = du[j][i][3][m];
		tmat_buts[k][j][m][4] = du[j][i][4][m];
	      }

	      tmp1_buts = 1.0 / tmat_buts[k][j][0][0];
	      tmp_buts = tmp1_buts * tmat_buts[k][j][1][0];
	      tmat_buts[k][j][1][1] =  tmat_buts[k][j][1][1] - tmp_buts * tmat_buts[k][j][0][1];
	      tmat_buts[k][j][1][2] =  tmat_buts[k][j][1][2] - tmp_buts * tmat_buts[k][j][0][2];
	      tmat_buts[k][j][1][3] =  tmat_buts[k][j][1][3] - tmp_buts * tmat_buts[k][j][0][3];
	      tmat_buts[k][j][1][4] =  tmat_buts[k][j][1][4] - tmp_buts * tmat_buts[k][j][0][4];
	      tv_buts[j][i][1] = tv_buts[j][i][1] - tv_buts[j][i][0] * tmp_buts;

	      tmp_buts = tmp1_buts * tmat_buts[k][j][2][0];
	      tmat_buts[k][j][2][1] =  tmat_buts[k][j][2][1] - tmp_buts * tmat_buts[k][j][0][1];
	      tmat_buts[k][j][2][2] =  tmat_buts[k][j][2][2] - tmp_buts * tmat_buts[k][j][0][2];
	      tmat_buts[k][j][2][3] =  tmat_buts[k][j][2][3] - tmp_buts * tmat_buts[k][j][0][3];
	      tmat_buts[k][j][2][4] =  tmat_buts[k][j][2][4] - tmp_buts * tmat_buts[k][j][0][4];
	      tv_buts[j][i][2] = tv_buts[j][i][2] - tv_buts[j][i][0] * tmp_buts;

	      tmp_buts = tmp1_buts * tmat_buts[k][j][3][0];
	      tmat_buts[k][j][3][1] =  tmat_buts[k][j][3][1] - tmp_buts * tmat_buts[k][j][0][1];
	      tmat_buts[k][j][3][2] =  tmat_buts[k][j][3][2] - tmp_buts * tmat_buts[k][j][0][2];
	      tmat_buts[k][j][3][3] =  tmat_buts[k][j][3][3] - tmp_buts * tmat_buts[k][j][0][3];
	      tmat_buts[k][j][3][4] =  tmat_buts[k][j][3][4] - tmp_buts * tmat_buts[k][j][0][4];
	      tv_buts[j][i][3] = tv_buts[j][i][3] - tv_buts[j][i][0] * tmp_buts;

	      tmp_buts = tmp1_buts * tmat_buts[k][j][4][0];
	      tmat_buts[k][j][4][1] =  tmat_buts[k][j][4][1] - tmp_buts * tmat_buts[k][j][0][1];
	      tmat_buts[k][j][4][2] =  tmat_buts[k][j][4][2] - tmp_buts * tmat_buts[k][j][0][2];
	      tmat_buts[k][j][4][3] =  tmat_buts[k][j][4][3] - tmp_buts * tmat_buts[k][j][0][3];
	      tmat_buts[k][j][4][4] =  tmat_buts[k][j][4][4] - tmp_buts * tmat_buts[k][j][0][4];
	      tv_buts[j][i][4] = tv_buts[j][i][4] - tv_buts[j][i][0] * tmp_buts;

	      tmp1_buts = 1.0 / tmat_buts[k][j][1][1];
	      tmp_buts = tmp1_buts * tmat_buts[k][j][2][1];
	      tmat_buts[k][j][2][2] =  tmat_buts[k][j][2][2] - tmp_buts * tmat_buts[k][j][1][2];
	      tmat_buts[k][j][2][3] =  tmat_buts[k][j][2][3] - tmp_buts * tmat_buts[k][j][1][3];
	      tmat_buts[k][j][2][4] =  tmat_buts[k][j][2][4] - tmp_buts * tmat_buts[k][j][1][4];
	      tv_buts[j][i][2] = tv_buts[j][i][2] - tv_buts[j][i][1] * tmp_buts;

	      tmp_buts = tmp1_buts * tmat_buts[k][j][3][1];
	      tmat_buts[k][j][3][2] =  tmat_buts[k][j][3][2] - tmp_buts * tmat_buts[k][j][1][2];
	      tmat_buts[k][j][3][3] =  tmat_buts[k][j][3][3] - tmp_buts * tmat_buts[k][j][1][3];
	      tmat_buts[k][j][3][4] =  tmat_buts[k][j][3][4] - tmp_buts * tmat_buts[k][j][1][4];
	      tv_buts[j][i][3] = tv_buts[j][i][3] - tv_buts[j][i][1] * tmp_buts;

	      tmp_buts = tmp1_buts * tmat_buts[k][j][4][1];
	      tmat_buts[k][j][4][2] =  tmat_buts[k][j][4][2] - tmp_buts * tmat_buts[k][j][1][2];
	      tmat_buts[k][j][4][3] =  tmat_buts[k][j][4][3] - tmp_buts * tmat_buts[k][j][1][3];
	      tmat_buts[k][j][4][4] =  tmat_buts[k][j][4][4] - tmp_buts * tmat_buts[k][j][1][4];
	      tv_buts[j][i][4] = tv_buts[j][i][4] - tv_buts[j][i][1] * tmp_buts;

	      tmp1_buts = 1.0 / tmat_buts[k][j][2][2];
	      tmp_buts = tmp1_buts * tmat_buts[k][j][3][2];
	      tmat_buts[k][j][3][3] =  tmat_buts[k][j][3][3] - tmp_buts * tmat_buts[k][j][2][3];
	      tmat_buts[k][j][3][4] =  tmat_buts[k][j][3][4] - tmp_buts * tmat_buts[k][j][2][4];
	      tv_buts[j][i][3] = tv_buts[j][i][3] - tv_buts[j][i][2] * tmp_buts;

	      tmp_buts = tmp1_buts * tmat_buts[k][j][4][2];
	      tmat_buts[k][j][4][3] =  tmat_buts[k][j][4][3] - tmp_buts * tmat_buts[k][j][2][3];
	      tmat_buts[k][j][4][4] =  tmat_buts[k][j][4][4] - tmp_buts * tmat_buts[k][j][2][4];
	      tv_buts[j][i][4] = tv_buts[j][i][4] - tv_buts[j][i][2] * tmp_buts;

	      tmp1_buts = 1.0 / tmat_buts[k][j][3][3];
	      tmp_buts = tmp1_buts * tmat_buts[k][j][4][3];
	      tmat_buts[k][j][4][4] =  tmat_buts[k][j][4][4] - tmp_buts * tmat_buts[k][j][3][4];
	      tv_buts[j][i][4] = tv_buts[j][i][4] - tv_buts[j][i][3] * tmp_buts;

	      //---------------------------------------------------------------------
	      // back substitution
	      //---------------------------------------------------------------------
	      tv_buts[j][i][4] = tv_buts[j][i][4] / tmat_buts[k][j][4][4];

	      tv_buts[j][i][3] = tv_buts[j][i][3] - tmat_buts[k][j][3][4] * tv_buts[j][i][4];
	      tv_buts[j][i][3] = tv_buts[j][i][3] / tmat_buts[k][j][3][3];

	      tv_buts[j][i][2] = tv_buts[j][i][2]
		- tmat_buts[k][j][2][3] * tv_buts[j][i][3]
		- tmat_buts[k][j][2][4] * tv_buts[j][i][4];
	      tv_buts[j][i][2] = tv_buts[j][i][2] / tmat_buts[k][j][2][2];

	      tv_buts[j][i][1] = tv_buts[j][i][1]
		- tmat_buts[k][j][1][2] * tv_buts[j][i][2]
		- tmat_buts[k][j][1][3] * tv_buts[j][i][3]
		- tmat_buts[k][j][1][4] * tv_buts[j][i][4];
	      tv_buts[j][i][1] = tv_buts[j][i][1] / tmat_buts[k][j][1][1];

	      tv_buts[j][i][0] = tv_buts[j][i][0]
		- tmat_buts[k][j][0][1] * tv_buts[j][i][1]
		- tmat_buts[k][j][0][2] * tv_buts[j][i][2]
		- tmat_buts[k][j][0][3] * tv_buts[j][i][3]
		- tmat_buts[k][j][0][4] * tv_buts[j][i][4];
	      tv_buts[j][i][0] = tv_buts[j][i][0] / tmat_buts[k][j][0][0];

	      rsd[k][j][i][0] = rsd[k][j][i][0] - tv_buts[j][i][0];
	      rsd[k][j][i][1] = rsd[k][j][i][1] - tv_buts[j][i][1];
	      rsd[k][j][i][2] = rsd[k][j][i][2] - tv_buts[j][i][2];
	      rsd[k][j][i][3] = rsd[k][j][i][3] - tv_buts[j][i][3];
	      rsd[k][j][i][4] = rsd[k][j][i][4] - tv_buts[j][i][4];
	    } // end j
	  } // end i
      // end buts(k);
    } // END pl SECOND
    
    //#pragma omp barrier
    //---------------------------------------------------------------------
    // update the variables
    //---------------------------------------------------------------------
    //#pragma omp master
    tmp2 = tmp;
    //#pragma omp for nowait
    #pragma acc parallel loop private(k, j, i, m)
    for (k = 1; k < nz-1; k++) {
      #pragma acc loop
      for (j = jst; j < jend; j++) {
        for (i = ist; i < iend; i++) {
          for (m = 0; m < 5; m++) {
            u[k][j][i][m] = u[k][j][i][m] + tmp * rsd[k][j][i][m];
          }
        }
      }
      // CHANGE tmp2->tmp
    }
    } //end parallel

    //---------------------------------------------------------------------
    // compute the max-norms of newton iteration corrections
    //---------------------------------------------------------------------
    if ( (istep % inorm) == 0 ) {
    // start l2norm first
          //l2norm( ISIZ1, ISIZ2, ISIZ3, nx0, ny0, nz0, ist, iend, jst, jend, rsd, delunm );
	  //---------------------------------------------------------------------
	  // local variables
	  //---------------------------------------------------------------------
	  double sum_local[5];

	  for (m = 0; m < 5; m++) {
	    delunm[m] = 0.0;
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
		  sum_local[m] = sum_local[m] + rsd[k][j][i][m] * rsd[k][j][i][m];
		}
	      }
	    }
	  }
	  for (m = 0; m < 5; m++) {
	    //#pragma omp atomic
	    //#pragma acc atomic
	    delunm[m] += sum_local[m];
	  }
	  } //end parallel

	  for (m = 0; m < 5; m++) {
	    delunm[m] = sqrt (delunm[m] / ( (nx0-2)*(ny0-2)*(nz0-2) ) );
	  }
    // end l2norm first
    }

    //---------------------------------------------------------------------
    // compute the steady-state residuals
    //---------------------------------------------------------------------
        // begin rhs()
          //printf("%d rhs\n", k);
	  double q;
	  double tmp_rhs, u_rhs[ISIZ3][6], r_rhs[ISIZ3][5];
	  double u21, u31, u41;
	  double u21i, u31i, u41i, u51i;
	  double u21j, u31j, u41j, u51j;
	  double u21k, u31k, u41k, u51k;
	  double u21im1, u31im1, u41im1, u51im1;
	  double u21jm1, u31jm1, u41jm1, u51jm1;
	  double u21km1, u31km1, u41km1, u51km1;

	  //#pragma omp parallel default(shared) private(i,j,k,m,q,flux,tmp_rhs,u_rhs,r_rhs,\
		      u51im1,u41im1,u31im1,u21im1,u51i,u41i,u31i,u21i,u21, \
		      u51jm1,u41jm1,u31jm1,u21jm1,u51j,u41j,u31j,u21j,u31, \
		      u51km1,u41km1,u31km1,u21km1,u51k,u41k,u31k,u21k,u41)
	  {
	  //#pragma omp for schedule(static)
	  #pragma acc parallel loop private(i,j,k, m)
	  for (k = 0; k < nz; k++) {
	    #pragma acc loop
	    for (j = 0; j < ny; j++) {
	      for (i = 0; i < nx; i++) {
		for (m = 0; m < 5; m++) {
		  rsd[k][j][i][m] = - frct[k][j][i][m];
		}
		tmp_rhs = 1.0 / u[k][j][i][0];
		rho_i[k][j][i] = tmp_rhs;
		qs[k][j][i] = 0.50 * (  u[k][j][i][1] * u[k][j][i][1]
		                      + u[k][j][i][2] * u[k][j][i][2]
		                      + u[k][j][i][3] * u[k][j][i][3] )
		                   * tmp_rhs;
	      }
	    }
	  }

	  //---------------------------------------------------------------------
	  // xi-direction flux differences
	  //---------------------------------------------------------------------
	  //#pragma omp for schedule(static) nowait
	  #pragma acc parallel loop private(i,j,k,m,q,flux,tmp_rhs,u_rhs,r_rhs,\
		      u51im1,u41im1,u31im1,u21im1,u51i,u41i,u31i,u21i,u21, \
		      u51jm1,u41jm1,u31jm1,u21jm1,u51j,u41j,u31j,u21j,u31, \
		      u51km1,u41km1,u31km1,u21km1,u51k,u41k,u31k,u21k,u41)
	  for (k = 1; k < nz - 1; k++) {
	    #pragma acc loop
	    for (j = jst; j < jend; j++) {
	      for (i = 0; i < nx; i++) {
		flux[i][0] = u[k][j][i][1];
		u21 = u[k][j][i][1] * rho_i[k][j][i];

		q = qs[k][j][i];

		flux[i][1] = u[k][j][i][1] * u21 + C2 * ( u[k][j][i][4] - q );
		flux[i][2] = u[k][j][i][2] * u21;
		flux[i][3] = u[k][j][i][3] * u21;
		flux[i][4] = ( C1 * u[k][j][i][4] - C2 * q ) * u21;
	      }

	      for (i = ist; i < iend; i++) {
		for (m = 0; m < 5; m++) {
		  rsd[k][j][i][m] =  rsd[k][j][i][m]
		    - tx2 * ( flux[i+1][m] - flux[i-1][m] );
		}
	      }

	      for (i = ist; i < nx; i++) {
		tmp_rhs = rho_i[k][j][i];

		u21i = tmp_rhs * u[k][j][i][1];
		u31i = tmp_rhs * u[k][j][i][2];
		u41i = tmp_rhs * u[k][j][i][3];
		u51i = tmp_rhs * u[k][j][i][4];

		tmp_rhs = rho_i[k][j][i-1];

		u21im1 = tmp_rhs * u[k][j][i-1][1];
		u31im1 = tmp_rhs * u[k][j][i-1][2];
		u41im1 = tmp_rhs * u[k][j][i-1][3];
		u51im1 = tmp_rhs * u[k][j][i-1][4];

		flux[i][1] = (4.0/3.0) * tx3 * (u21i-u21im1);
		flux[i][2] = tx3 * ( u31i - u31im1 );
		flux[i][3] = tx3 * ( u41i - u41im1 );
		flux[i][4] = 0.50 * ( 1.0 - C1*C5 )
		  * tx3 * ( ( u21i*u21i     + u31i*u31i     + u41i*u41i )
		          - ( u21im1*u21im1 + u31im1*u31im1 + u41im1*u41im1 ) )
		  + (1.0/6.0)
		  * tx3 * ( u21i*u21i - u21im1*u21im1 )
		  + C1 * C5 * tx3 * ( u51i - u51im1 );
	      }

	      for (i = ist; i < iend; i++) {
		rsd[k][j][i][0] = rsd[k][j][i][0]
		  + dx1 * tx1 * (        u[k][j][i-1][0]
		                 - 2.0 * u[k][j][i][0]
		                 +       u[k][j][i+1][0] );
		rsd[k][j][i][1] = rsd[k][j][i][1]
		  + tx3 * C3 * C4 * ( flux[i+1][1] - flux[i][1] )
		  + dx2 * tx1 * (        u[k][j][i-1][1]
		                 - 2.0 * u[k][j][i][1]
		                 +       u[k][j][i+1][1] );
		rsd[k][j][i][2] = rsd[k][j][i][2]
		  + tx3 * C3 * C4 * ( flux[i+1][2] - flux[i][2] )
		  + dx3 * tx1 * (        u[k][j][i-1][2]
		                 - 2.0 * u[k][j][i][2]
		                 +       u[k][j][i+1][2] );
		rsd[k][j][i][3] = rsd[k][j][i][3]
		  + tx3 * C3 * C4 * ( flux[i+1][3] - flux[i][3] )
		  + dx4 * tx1 * (        u[k][j][i-1][3]
		                 - 2.0 * u[k][j][i][3]
		                 +       u[k][j][i+1][3] );
		rsd[k][j][i][4] = rsd[k][j][i][4]
		  + tx3 * C3 * C4 * ( flux[i+1][4] - flux[i][4] )
		  + dx5 * tx1 * (        u[k][j][i-1][4]
		                 - 2.0 * u[k][j][i][4]
		                 +       u[k][j][i+1][4] );
	      }

	      //---------------------------------------------------------------------
	      // Fourth-order dissipation
	      //---------------------------------------------------------------------
	      for (m = 0; m < 5; m++) {
		rsd[k][j][1][m] = rsd[k][j][1][m]
		  - dssp * ( + 5.0 * u[k][j][1][m]
		             - 4.0 * u[k][j][2][m]
		             +       u[k][j][3][m] );
		rsd[k][j][2][m] = rsd[k][j][2][m]
		  - dssp * ( - 4.0 * u[k][j][1][m]
		             + 6.0 * u[k][j][2][m]
		             - 4.0 * u[k][j][3][m]
		             +       u[k][j][4][m] );
	      }

	      for (i = 3; i < nx - 3; i++) {
		for (m = 0; m < 5; m++) {
		  rsd[k][j][i][m] = rsd[k][j][i][m]
		    - dssp * (         u[k][j][i-2][m]
		               - 4.0 * u[k][j][i-1][m]
		               + 6.0 * u[k][j][i][m]
		               - 4.0 * u[k][j][i+1][m]
		               +       u[k][j][i+2][m] );
		}
	      }


	      for (m = 0; m < 5; m++) {
		rsd[k][j][nx-3][m] = rsd[k][j][nx-3][m]
		  - dssp * (         u[k][j][nx-5][m]
		             - 4.0 * u[k][j][nx-4][m]
		             + 6.0 * u[k][j][nx-3][m]
		             - 4.0 * u[k][j][nx-2][m] );
		rsd[k][j][nx-2][m] = rsd[k][j][nx-2][m]
		  - dssp * (         u[k][j][nx-4][m]
		             - 4.0 * u[k][j][nx-3][m]
		             + 5.0 * u[k][j][nx-2][m] );
	      }

	    }
	  }

	  // eta-direction flux differences
	  //---------------------------------------------------------------------
	  //#pragma omp for schedule(static)
	  #pragma acc parallel loop private(i,j,k,m,q,flux,tmp_rhs,u_rhs,r_rhs,\
		      u51im1,u41im1,u31im1,u21im1,u51i,u41i,u31i,u21i,u21, \
		      u51jm1,u41jm1,u31jm1,u21jm1,u51j,u41j,u31j,u21j,u31, \
		      u51km1,u41km1,u31km1,u21km1,u51k,u41k,u31k,u21k,u41)
	  for (k = 1; k < nz - 1; k++) {
	    #pragma acc loop
	    for (i = ist; i < iend; i++) {
	      for (j = 0; j < ny; j++) {
		flux[j][0] = u[k][j][i][2];
		u31 = u[k][j][i][2] * rho_i[k][j][i];

		q = qs[k][j][i];

		flux[j][1] = u[k][j][i][1] * u31;
		flux[j][2] = u[k][j][i][2] * u31 + C2 * (u[k][j][i][4]-q);
		flux[j][3] = u[k][j][i][3] * u31;
		flux[j][4] = ( C1 * u[k][j][i][4] - C2 * q ) * u31;
	      }

	      for (j = jst; j < jend; j++) {
		for (m = 0; m < 5; m++) {
		  rsd[k][j][i][m] =  rsd[k][j][i][m]
		    - ty2 * ( flux[j+1][m] - flux[j-1][m] );
		}
	      }

	      for (j = jst; j < ny; j++) {
		tmp_rhs = rho_i[k][j][i];

		u21j = tmp_rhs * u[k][j][i][1];
		u31j = tmp_rhs * u[k][j][i][2];
		u41j = tmp_rhs * u[k][j][i][3];
		u51j = tmp_rhs * u[k][j][i][4];

		tmp_rhs = rho_i[k][j-1][i];
		u21jm1 = tmp_rhs * u[k][j-1][i][1];
		u31jm1 = tmp_rhs * u[k][j-1][i][2];
		u41jm1 = tmp_rhs * u[k][j-1][i][3];
		u51jm1 = tmp_rhs * u[k][j-1][i][4];

		flux[j][1] = ty3 * ( u21j - u21jm1 );
		flux[j][2] = (4.0/3.0) * ty3 * (u31j-u31jm1);
		flux[j][3] = ty3 * ( u41j - u41jm1 );
		flux[j][4] = 0.50 * ( 1.0 - C1*C5 )
		  * ty3 * ( ( u21j*u21j     + u31j*u31j     + u41j*u41j )
		          - ( u21jm1*u21jm1 + u31jm1*u31jm1 + u41jm1*u41jm1 ) )
		  + (1.0/6.0)
		  * ty3 * ( u31j*u31j - u31jm1*u31jm1 )
		  + C1 * C5 * ty3 * ( u51j - u51jm1 );
	      }

	      for (j = jst; j < jend; j++) {
		rsd[k][j][i][0] = rsd[k][j][i][0]
		  + dy1 * ty1 * (         u[k][j-1][i][0]
		                  - 2.0 * u[k][j][i][0]
		                  +       u[k][j+1][i][0] );

		rsd[k][j][i][1] = rsd[k][j][i][1]
		  + ty3 * C3 * C4 * ( flux[j+1][1] - flux[j][1] )
		  + dy2 * ty1 * (         u[k][j-1][i][1]
		                  - 2.0 * u[k][j][i][1]
		                  +       u[k][j+1][i][1] );

		rsd[k][j][i][2] = rsd[k][j][i][2]
		  + ty3 * C3 * C4 * ( flux[j+1][2] - flux[j][2] )
		  + dy3 * ty1 * (         u[k][j-1][i][2]
		                  - 2.0 * u[k][j][i][2]
		                  +       u[k][j+1][i][2] );

		rsd[k][j][i][3] = rsd[k][j][i][3]
		  + ty3 * C3 * C4 * ( flux[j+1][3] - flux[j][3] )
		  + dy4 * ty1 * (         u[k][j-1][i][3]
		                  - 2.0 * u[k][j][i][3]
		                  +       u[k][j+1][i][3] );

		rsd[k][j][i][4] = rsd[k][j][i][4]
		  + ty3 * C3 * C4 * ( flux[j+1][4] - flux[j][4] )
		  + dy5 * ty1 * (         u[k][j-1][i][4]
		                  - 2.0 * u[k][j][i][4]
		                  +       u[k][j+1][i][4] );
	      }
	    }

	    //---------------------------------------------------------------------
	    // fourth-order dissipation
	    //---------------------------------------------------------------------
	    for (i = ist; i < iend; i++) {
	      for (m = 0; m < 5; m++) {
		rsd[k][1][i][m] = rsd[k][1][i][m]
		  - dssp * ( + 5.0 * u[k][1][i][m]
		             - 4.0 * u[k][2][i][m]
		             +       u[k][3][i][m] );
		rsd[k][2][i][m] = rsd[k][2][i][m]
		  - dssp * ( - 4.0 * u[k][1][i][m]
		             + 6.0 * u[k][2][i][m]
		             - 4.0 * u[k][3][i][m]
		             +       u[k][4][i][m] );
	      }
	    }

	    for (j = 3; j < ny - 3; j++) {
	      for (i = ist; i < iend; i++) {
		for (m = 0; m < 5; m++) {
		  rsd[k][j][i][m] = rsd[k][j][i][m]
		    - dssp * (         u[k][j-2][i][m]
		               - 4.0 * u[k][j-1][i][m]
		               + 6.0 * u[k][j][i][m]
		               - 4.0 * u[k][j+1][i][m]
		               +       u[k][j+2][i][m] );
		}
	      }
	    }

	    for (i = ist; i < iend; i++) {
	      for (m = 0; m < 5; m++) {
		rsd[k][ny-3][i][m] = rsd[k][ny-3][i][m]
		  - dssp * (         u[k][ny-5][i][m]
		             - 4.0 * u[k][ny-4][i][m]
		             + 6.0 * u[k][ny-3][i][m]
		             - 4.0 * u[k][ny-2][i][m] );
		rsd[k][ny-2][i][m] = rsd[k][ny-2][i][m]
		  - dssp * (         u[k][ny-4][i][m]
		             - 4.0 * u[k][ny-3][i][m]
		             + 5.0 * u[k][ny-2][i][m] );
	      }
	    }

	  }

	  //---------------------------------------------------------------------
	  // zeta-direction flux differences
	  //---------------------------------------------------------------------
	  //#pragma omp for schedule(static) nowait
	  #pragma acc parallel loop private(i,j,k,m,q,flux,tmp_rhs,u_rhs,r_rhs,\
		      u51im1,u41im1,u31im1,u21im1,u51i,u41i,u31i,u21i,u21, \
		      u51jm1,u41jm1,u31jm1,u21jm1,u51j,u41j,u31j,u21j,u31, \
		      u51km1,u41km1,u31km1,u21km1,u51k,u41k,u31k,u21k,u41)
	  for (j = jst; j < jend; j++) {
	    #pragma acc loop
	    for (i = ist; i < iend; i++) {
	      for (k = 0; k < nz; k++) {
		u_rhs[k][0] = u[k][j][i][0];
		u_rhs[k][1] = u[k][j][i][1];
		u_rhs[k][2] = u[k][j][i][2];
		u_rhs[k][3] = u[k][j][i][3];
		u_rhs[k][4] = u[k][j][i][4];
		u_rhs[k][5] = rho_i[k][j][i];
	      }
	      for (k = 0; k < nz; k++) {
		flux[k][0] = u_rhs[k][3];
		u41 = u_rhs[k][3] * u_rhs[k][5];

		q = qs[k][j][i];

		flux[k][1] = u_rhs[k][1] * u41;
		flux[k][2] = u_rhs[k][2] * u41;
		flux[k][3] = u_rhs[k][3] * u41 + C2 * (u_rhs[k][4]-q);
		flux[k][4] = ( C1 * u_rhs[k][4] - C2 * q ) * u41;
	      }

	      for (k = 1; k < nz - 1; k++) {
		for (m = 0; m < 5; m++) {
		  r_rhs[k][m] =  rsd[k][j][i][m]
		    - tz2 * ( flux[k+1][m] - flux[k-1][m] );
		}
	      }

	      for (k = 1; k < nz; k++) {
		tmp_rhs = u_rhs[k][5];

		u21k = tmp_rhs * u_rhs[k][1];
		u31k = tmp_rhs * u_rhs[k][2];
		u41k = tmp_rhs * u_rhs[k][3];
		u51k = tmp_rhs * u_rhs[k][4];

		tmp_rhs = u_rhs[k-1][5];

		u21km1 = tmp_rhs * u_rhs[k-1][1];
		u31km1 = tmp_rhs * u_rhs[k-1][2];
		u41km1 = tmp_rhs * u_rhs[k-1][3];
		u51km1 = tmp_rhs * u_rhs[k-1][4];

		flux[k][1] = tz3 * ( u21k - u21km1 );
		flux[k][2] = tz3 * ( u31k - u31km1 );
		flux[k][3] = (4.0/3.0) * tz3 * (u41k-u41km1);
		flux[k][4] = 0.50 * ( 1.0 - C1*C5 )
		  * tz3 * ( ( u21k*u21k     + u31k*u31k     + u41k*u41k )
		          - ( u21km1*u21km1 + u31km1*u31km1 + u41km1*u41km1 ) )
		  + (1.0/6.0)
		  * tz3 * ( u41k*u41k - u41km1*u41km1 )
		  + C1 * C5 * tz3 * ( u51k - u51km1 );
	      }

	      for (k = 1; k < nz - 1; k++) {
		r_rhs[k][0] = r_rhs[k][0]
		  + dz1 * tz1 * (         u_rhs[k-1][0]
		                  - 2.0 * u_rhs[k][0]
		                  +       u_rhs[k+1][0] );
		r_rhs[k][1] = r_rhs[k][1]
		  + tz3 * C3 * C4 * ( flux[k+1][1] - flux[k][1] )
		  + dz2 * tz1 * (         u_rhs[k-1][1]
		                  - 2.0 * u_rhs[k][1]
		                  +       u_rhs[k+1][1] );
		r_rhs[k][2] = r_rhs[k][2]
		  + tz3 * C3 * C4 * ( flux[k+1][2] - flux[k][2] )
		  + dz3 * tz1 * (         u_rhs[k-1][2]
		                  - 2.0 * u_rhs[k][2]
		                  +       u_rhs[k+1][2] );
		r_rhs[k][3] = r_rhs[k][3]
		  + tz3 * C3 * C4 * ( flux[k+1][3] - flux[k][3] )
		  + dz4 * tz1 * (         u_rhs[k-1][3]
		                  - 2.0 * u_rhs[k][3]
		                  +       u_rhs[k+1][3] );
		r_rhs[k][4] = r_rhs[k][4]
		  + tz3 * C3 * C4 * ( flux[k+1][4] - flux[k][4] )
		  + dz5 * tz1 * (         u_rhs[k-1][4]
		                  - 2.0 * u_rhs[k][4]
		                  +       u_rhs[k+1][4] );
	      }

	      //---------------------------------------------------------------------
	      // fourth-order dissipation
	      //---------------------------------------------------------------------
	      for (m = 0; m < 5; m++) {
		rsd[1][j][i][m] = r_rhs[1][m]
		  - dssp * ( + 5.0 * u_rhs[1][m]
		             - 4.0 * u_rhs[2][m]
		             +       u_rhs[3][m] );
		rsd[2][j][i][m] = r_rhs[2][m]
		  - dssp * ( - 4.0 * u_rhs[1][m]
		             + 6.0 * u_rhs[2][m]
		             - 4.0 * u_rhs[3][m]
		             +       u_rhs[4][m] );
	      }

	      for (k = 3; k < nz - 3; k++) {
		for (m = 0; m < 5; m++) {
		  rsd[k][j][i][m] = r_rhs[k][m]
		    - dssp * (         u_rhs[k-2][m]
		               - 4.0 * u_rhs[k-1][m]
		               + 6.0 * u_rhs[k][m]
		               - 4.0 * u_rhs[k+1][m]
		               +       u_rhs[k+2][m] );
		}
	      }

	      for (m = 0; m < 5; m++) {
		rsd[nz-3][j][i][m] = r_rhs[nz-3][m]
		  - dssp * (         u_rhs[nz-5][m]
		             - 4.0 * u_rhs[nz-4][m]
		             + 6.0 * u_rhs[nz-3][m]
		             - 4.0 * u_rhs[nz-2][m] );
		rsd[nz-2][j][i][m] = r_rhs[nz-2][m]
		  - dssp * (         u_rhs[nz-4][m]
		             - 4.0 * u_rhs[nz-3][m]
		             + 5.0 * u_rhs[nz-2][m] );
	      }
	    }
	  }
	  } //end parallel
    // end rhs()

    //---------------------------------------------------------------------
    // compute the max-norms of newton iteration residuals
    //---------------------------------------------------------------------
  } 
  //} // PARALLEL END
  } // DATA END
        // start l2norm second
        //printf("%d l2norm second\n", k);
        //l2norm( ISIZ1, ISIZ2, ISIZ3, nx0, ny0, nz0, ist, iend, jst, jend, rsd, rsdnm );      
	//---------------------------------------------------------------------
	  // local variables
	  //---------------------------------------------------------------------
	  double sum_local[5];
	  for (m = 0; m < 5; m++) {
	    rsdnm[m] = 0.0;
	  }
	  //#pragma omp parallel default(shared) private(i,j,k,m,sum_local)
	  //{
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
		  sum_local[m] = sum_local[m] + rsd[k][j][i][m] * rsd[k][j][i][m];
		}
	      }
	    }
	  }
	  for (m = 0; m < 5; m++) {
	    //#pragma omp atomic
	    rsdnm[m] += sum_local[m];
	   }
	  //} //end parallel
	  for (m = 0; m < 5; m++) {
	    rsdnm[m] = sqrt ( rsdnm[m] / ( (nx0-2)*(ny0-2)*(nz0-2) ) );
	  }	  
    // end l2norm second
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
      //break;
    }
  timer_stop(1);
  maxtime = timer_read(1);
}
