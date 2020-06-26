/* 
  This file was generated automatically with /nfs/data-012/marques/software/source/libxc/svn/scripts/maple2c.pl.
  Do not edit this file directly as it can be overwritten!!

  This Source Code Form is subject to the terms of the Mozilla Public
  License, v. 2.0. If a copy of the MPL was not distributed with this
  file, You can obtain one at http://mozilla.org/MPL/2.0/.

  Maple version     : Maple 2016 (X86 64 LINUX)
  Maple source      : ../maple/gga_x_rge2.mpl
  Type of functional: work_gga_x
*/

void xc_gga_x_rge2_enhance
  (const xc_func_type_cuda *p,  xc_gga_work_x_t *r)
{
  double t1, t2, t3, t4, t6, t7, t10, t12;
  double t13, t14, t17, t20, t21, t27, t30, t31;
  double t37, t40;


  t1 = M_CBRT6;
  t2 = 0.31415926535897932385e1 * 0.31415926535897932385e1;
  t3 = cbrt(t2);
  t4 = t3 * t3;
  t6 = t1 / t4;
  t7 = r->x * r->x;
  t10 = t1 * t1;
  t12 = 0.1e1 / t3 / t2;
  t13 = t10 * t12;
  t14 = t7 * t7;
  t17 = 0.8040e0 + 0.5e1 / 0.972e3 * t6 * t7 + 0.32911784453572541027e-4 * t13 * t14;
  r->f = 0.18040e1 - 0.64641600e0 / t17;

  if(r->order < 1) return;

  t20 = t17 * t17;
  t21 = 0.1e1 / t20;
  t27 = 0.5e1 / 0.486e3 * t6 * r->x + 0.13164713781429016411e-3 * t13 * t7 * r->x;
  r->dfdx = 0.64641600e0 * t21 * t27;

  if(r->order < 2) return;

  t30 = 0.1e1 / t20 / t17;
  t31 = t27 * t27;
  t37 = 0.5e1 / 0.486e3 * t6 + 0.39494141344287049233e-3 * t13 * t7;
  r->d2fdx2 = -0.129283200e1 * t30 * t31 + 0.64641600e0 * t21 * t37;

  if(r->order < 3) return;

  t40 = t20 * t20;
  r->d3fdx3 = 0.387849600e1 / t40 * t31 * t27 - 0.387849600e1 * t30 * t27 * t37 + 0.51059289742417314434e-3 * t21 * t10 * t12 * r->x;

  if(r->order < 4) return;


}

#define maple2c_order 3
#define maple2c_func  xc_gga_x_rge2_enhance
