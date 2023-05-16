#include <stdio.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "applu.incl"

void read_input()
{
  FILE *fp;
  int result;
  
  //---------------------------------------------------------------------
  // if input file does not exist, it uses defaults
  //    ipr = 1 for detailed progress output
  //    inorm = how often the norm is printed (once every inorm iterations)
  //    itmax = number of pseudo time steps
  //    dt = time step
  //    omega 1 over-relaxation factor for SSOR
  //    tolrsd = steady state residual tolerance levels
  //    nx, ny, nz = number of grid points in x, y, z directions
  //---------------------------------------------------------------------

  printf("\n\n NAS Parallel Benchmarks (NPB3.3-OMP-C) - LU Benchmark\n\n");

  if ((fp = fopen("inputlu.data", "r")) != NULL) {
    printf("Reading from input file inputlu.data\n");

    while (fgetc(fp) != '\n');
    while (fgetc(fp) != '\n');
    result = fscanf(fp, "%d%d", &ipr, &inorm); 
    while (fgetc(fp) != '\n');

    while (fgetc(fp) != '\n');
    while (fgetc(fp) != '\n');
    result = fscanf(fp, "%d", &itmax);
    while (fgetc(fp) != '\n');

    while (fgetc(fp) != '\n');
    while (fgetc(fp) != '\n');
    result = fscanf(fp, "%lf", &dt);
    while (fgetc(fp) != '\n');

    while (fgetc(fp) != '\n');
    while (fgetc(fp) != '\n');
    result = fscanf(fp, "%lf", &omega);
    while (fgetc(fp) != '\n');

    while (fgetc(fp) != '\n');
    while (fgetc(fp) != '\n');
    result = fscanf(fp, "%lf%lf%lf%lf%lf",
        &tolrsd[0], &tolrsd[1], &tolrsd[2], &tolrsd[3], &tolrsd[4]);
    while (fgetc(fp) != '\n');
    while (fgetc(fp) != '\n');
    result = fscanf(fp, "%d%d%d", &nx0, &ny0, &nz0);
    fclose(fp);
  } else {
    ipr = IPR_DEFAULT;
    inorm = INORM_DEFAULT;
    itmax = ITMAX_DEFAULT;
    dt = DT_DEFAULT;
    omega = OMEGA_DEFAULT;
    tolrsd[0] = TOLRSD1_DEF;
    tolrsd[1] = TOLRSD2_DEF;
    tolrsd[2] = TOLRSD3_DEF;
    tolrsd[3] = TOLRSD4_DEF;
    tolrsd[4] = TOLRSD5_DEF;
    nx0 = ISIZ1;
    ny0 = ISIZ2;
    nz0 = ISIZ3;
  }

  //---------------------------------------------------------------------
  // check problem size
  //---------------------------------------------------------------------
  if ( ( nx0 < 4 ) || ( ny0 < 4 ) || ( nz0 < 4 ) ) {
    printf("     PROBLEM SIZE IS TOO SMALL - \n"
           "     SET EACH OF NX, NY AND NZ AT LEAST EQUAL TO 5\n");
    exit(EXIT_FAILURE);
  }

  if ( ( nx0 > ISIZ1 ) || ( ny0 > ISIZ2 ) || ( nz0 > ISIZ3 ) ) {
    printf("     PROBLEM SIZE IS TOO LARGE - \n"
           "     NX, NY AND NZ SHOULD BE EQUAL TO \n"
           "     ISIZ1, ISIZ2 AND ISIZ3 RESPECTIVELY\n");
    exit(EXIT_FAILURE);
  }

  printf(" Size: %4dx%4dx%4d\n", nx0, ny0, nz0);
  printf(" Iterations:                  %5d\n", itmax);
  printf(" Number of available threads: %5d\n", omp_get_max_threads());
  printf("\n");
}
