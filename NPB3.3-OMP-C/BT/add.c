#include "header.h"
#include "timers.h"

//---------------------------------------------------------------------
// addition of update to the vector u
//---------------------------------------------------------------------
void add()
{
  int i, j, k, m;

  if (timeron) timer_start(t_add);
  #pragma omp parallel for default(shared) private(i,j,k,m)
  for (k = 1; k <= grid_points[2]-2; k++) {
    for (j = 1; j <= grid_points[1]-2; j++) {
      for (i = 1; i <= grid_points[0]-2; i++) {
        for (m = 0; m < 5; m++) {
          u[k][j][i][m] = u[k][j][i][m] + rhs[k][j][i][m];
        }
      }
    }
  }
  if (timeron) timer_stop(t_add);
}
