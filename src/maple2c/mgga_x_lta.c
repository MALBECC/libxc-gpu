/* 
  This file was generated automatically with /nfs/data-012/marques/software/source/libxc/svn/scripts/maple2c.pl.
  Do not edit this file directly as it can be overwritten!!

  This Source Code Form is subject to the terms of the Mozilla Public
  License, v. 2.0. If a copy of the MPL was not distributed with this
  file, You can obtain one at http://mozilla.org/MPL/2.0/.

  Maple version     : Maple 2016 (X86 64 LINUX)
  Maple source      : ../maple/mgga_x_lta.mpl
  Type of functional: work_mgga_x
*/

static void 
xc_mgga_x_lta_enhance(const xc_func_type_cuda *pt, xc_mgga_work_x_t *r)
{
  double t1, t2, t3, t4, t5, t6, t7, t8;
  double t9, t12, t13, t14, t15, t16, t17;


  t1 = pow(0.5e1, 0.1e1 / 0.5e1);
  t2 = t1 * t1;
  t3 = t2 * t2;
  t4 = pow(0.9e1, 0.1e1 / 0.5e1);
  t5 = t3 * t4;
  t6 = M_CBRT6;
  t7 = 0.31415926535897932385e1 * 0.31415926535897932385e1;
  t8 = POW_1_3(t7);
  t9 = t8 * t8;
  t12 = pow(t6 / t9, 0.1e1 / 0.5e1);
  t13 = t12 * t12;
  t14 = t13 * t13;
  t15 = pow(r->t, 0.1e1 / 0.5e1);
  t16 = t15 * t15;
  t17 = t16 * t16;
  r->f = t5 * t14 * t17 / 0.9e1;

  if(r->order < 1) return;

  r->dfdrs = 0.0e0;
  r->dfdx = 0.0e0;
  r->dfdt = 0.4e1 / 0.45e2 * t5 * t14 / t15;
  r->dfdu = 0.0e0;

  if(r->order < 2) return;

  r->d2fdrs2 = 0.0e0;
  r->d2fdx2 = 0.0e0;
  r->d2fdt2 = -0.4e1 / 0.225e3 * t5 * t14 / t15 / r->t;
  r->d2fdu2 = 0.0e0;
  r->d2fdrsx = 0.0e0;
  r->d2fdrst = 0.0e0;
  r->d2fdrsu = 0.0e0;
  r->d2fdxt = 0.0e0;
  r->d2fdxu = 0.0e0;
  r->d2fdtu = 0.0e0;

  if(r->order < 3) return;


}

#define maple2c_order 3
#define maple2c_func  xc_mgga_x_lta_enhance
