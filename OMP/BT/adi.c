#include "header.h"

void adi()
{
  compute_rhs();

  x_solve();

  y_solve();

  z_solve();

  add();
}
