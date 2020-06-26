/* 
  This file was generated automatically with /nfs/data-012/marques/software/source/libxc/svn/scripts/maple2c.pl.
  Do not edit this file directly as it can be overwritten!!

  This Source Code Form is subject to the terms of the Mozilla Public
  License, v. 2.0. If a copy of the MPL was not distributed with this
  file, You can obtain one at http://mozilla.org/MPL/2.0/.

  Maple version     : Maple 2016 (X86 64 LINUX)
  Maple source      : ../maple/gga_x_2d_pbe.mpl
  Type of functional: work_gga_x
*/

void xc_gga_x_2d_pbe_enhance
  (const xc_func_type_cuda *p,  xc_gga_work_x_t *r)
{
  double t1, t3, t6, t7, t10, t14;


  t1 = r->x * r->x;
  t3 = 0.4604e0 + 0.70534859642542911404e-2 * t1;
  r->f = 0.14604e1 - 0.21196816e0 / t3;

  if(r->order < 1) return;

  t6 = t3 * t3;
  t7 = 0.1e1 / t6;
  r->dfdx = 0.29902288828576157303e-2 * t7 * r->x;

  if(r->order < 2) return;

  t10 = 0.1e1 / t6 / t3;
  r->d2fdx2 = -0.84366149820575925909e-4 * t10 * t1 + 0.29902288828576157303e-2 * t7;

  if(r->order < 3) return;

  t14 = t6 * t6;
  r->d3fdx3 = 0.35704527217056418575e-5 / t14 * t1 * r->x - 0.25309844946172777773e-3 * t10 * r->x;

  if(r->order < 4) return;


}

#define maple2c_order 3
#define maple2c_func  xc_gga_x_2d_pbe_enhance
