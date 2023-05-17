#include <stdio.h>
#include <stdlib.h>
#include "applu.incl"

void domain()
{
  //---------------------------------------------------------------------
  // local variables
  //---------------------------------------------------------------------
  nx = nx0;
  ny = ny0;
  nz = nz0;

  //---------------------------------------------------------------------
  // check the sub-domain size
  //---------------------------------------------------------------------
  if ( ( nx < 4 ) || ( ny < 4 ) || ( nz < 4 ) ) {
    printf("     SUBDOMAIN SIZE IS TOO SMALL - \n"
           "     ADJUST PROBLEM SIZE OR NUMBER OF PROCESSORS\n"
           "     SO THAT NX, NY AND NZ ARE GREATER THAN OR EQUAL\n"
           "     TO 4 THEY ARE CURRENTLY%3d%3d%3d\n", nx, ny, nz);
    exit(EXIT_FAILURE);
  }

  if ( ( nx > ISIZ1 ) || ( ny > ISIZ2 ) || ( nz > ISIZ3 ) ) {
    printf("     SUBDOMAIN SIZE IS TOO LARGE - \n"
           "     ADJUST PROBLEM SIZE OR NUMBER OF PROCESSORS\n"
           "     SO THAT NX, NY AND NZ ARE LESS THAN OR EQUAL TO \n"
           "     ISIZ1, ISIZ2 AND ISIZ3 RESPECTIVELY.  THEY ARE\n"
           "     CURRENTLYi%4d%4d%4d\n", nx, ny, nz);
    exit(EXIT_FAILURE);
  }

  //---------------------------------------------------------------------
  // set up the start and end in i and j extents for all processors
  //---------------------------------------------------------------------
  ist = 1;
  iend = nx - 1;

  jst = 1;
  jend = ny - 1;

  ii1 = 1;
  ii2 = nx0 - 1;
  ji1 = 1;
  ji2 = ny0 - 2;
  ki1 = 2;
  ki2 = nz0 - 1;
}
