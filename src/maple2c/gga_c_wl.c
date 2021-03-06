/* 
  This file was generated automatically with /nfs/data-012/marques/software/source/libxc/svn/scripts/maple2c.pl.
  Do not edit this file directly as it can be overwritten!!

  This Source Code Form is subject to the terms of the Mozilla Public
  License, v. 2.0. If a copy of the MPL was not distributed with this
  file, You can obtain one at http://mozilla.org/MPL/2.0/.

  Maple version     : Maple 2016 (X86 64 LINUX)
  Maple source      : ../maple/gga_c_wl.mpl
  Type of functional: work_gga_c
*/

void xc_gga_c_wl_func
  (const xc_func_type_cuda *p, xc_gga_work_c_t *r)
{
  double t1, t2, t3, t5, t6, t9, t10, t11;
  double t12, t13, t14, t15, t16, t21, t22, t24;
  double t27, t28, t32, t36, t38, t41, t44, t45;
  double t47, t49, t52;


  t1 = r->z * r->z;
  t2 = -t1 + 0.1e1;
  t3 = sqrt(t2);
  t5 = -0.74860e0 + 0.6001e-1 * r->xt;
  t6 = t3 * t5;
  t9 = 0.360073e1 + 0.90000e0 * r->xs[0] + 0.90000e0 * r->xs[1] + r->rs;
  t10 = 0.1e1 / t9;
  r->f = t6 * t10;

  if(r->order < 1) return;

  t11 = t9 * t9;
  t12 = 0.1e1 / t11;
  t13 = t6 * t12;
  r->dfdrs = -t13;
  t14 = 0.1e1 / t3;
  t15 = t14 * t5;
  t16 = t10 * r->z;
  r->dfdz = -t15 * t16;
  r->dfdxt = 0.6001e-1 * t3 * t10;
  r->dfdxs[0] = -0.90000e0 * t13;
  r->dfdxs[1] = r->dfdxs[0];

  if(r->order < 2) return;

  t21 = 0.1e1 / t11 / t9;
  t22 = t6 * t21;
  r->d2fdrs2 = 0.2e1 * t22;
  r->d2fdrsz = t15 * t12 * r->z;
  t24 = t3 * t12;
  r->d2fdrsxt = -0.6001e-1 * t24;
  r->d2fdrsxs[0] = 0.180000e1 * t22;
  r->d2fdrsxs[1] = r->d2fdrsxs[0];
  t27 = 0.1e1 / t3 / t2;
  t28 = t27 * t5;
  r->d2fdz2 = -t28 * t10 * t1 - t15 * t10;
  t32 = t14 * t10;
  r->d2fdzxt = -0.6001e-1 * t32 * r->z;
  r->d2fdzxs[0] = 0.90000e0 * r->d2fdrsz;
  r->d2fdzxs[1] = r->d2fdzxs[0];
  r->d2fdxt2 = 0.0e0;
  r->d2fdxtxs[0] = -0.540090000e-1 * t24;
  r->d2fdxtxs[1] = r->d2fdxtxs[0];
  r->d2fdxs2[0] = 0.16200000000e1 * t22;
  r->d2fdxs2[1] = r->d2fdxs2[0];
  r->d2fdxs2[2] = r->d2fdxs2[1];

  if(r->order < 3) return;

  t36 = t11 * t11;
  t38 = t6 / t36;
  r->d3fdrs3 = -0.6e1 * t38;
  t41 = t15 * t21 * r->z;
  r->d3fdrs2z = -0.2e1 * t41;
  t44 = t28 * t12 * t1;
  t45 = t15 * t12;
  r->d3fdrsz2 = t44 + t45;
  t47 = t14 * t12 * r->z;
  r->d3fdrszxt = 0.6001e-1 * t47;
  r->d3fdrszxs[0] = -0.180000e1 * t41;
  r->d3fdrszxs[1] = r->d3fdrszxs[0];
  t49 = t3 * t21;
  r->d3fdrs2xt = 0.12002e0 * t49;
  r->d3fdrsxt2 = 0.0e0;
  r->d3fdrsxtxs[0] = 0.1080180000e0 * t49;
  r->d3fdrsxtxs[1] = r->d3fdrsxtxs[0];
  r->d3fdrs2xs[0] = -0.540000e1 * t38;
  r->d3fdrs2xs[1] = r->d3fdrs2xs[0];
  r->d3fdrsxs2[0] = -0.48600000000e1 * t38;
  r->d3fdrsxs2[1] = r->d3fdrsxs2[0];
  r->d3fdrsxs2[2] = r->d3fdrsxs2[1];
  t52 = t2 * t2;
  r->d3fdz3 = -0.3e1 / t3 / t52 * t5 * t10 * t1 * r->z - 0.3e1 * t28 * t16;
  r->d3fdz2xt = -0.6001e-1 * t27 * t10 * t1 - 0.6001e-1 * t32;
  r->d3fdzxt2 = 0.0e0;
  r->d3fdzxtxs[0] = 0.540090000e-1 * t47;
  r->d3fdzxtxs[1] = r->d3fdzxtxs[0];
  r->d3fdz2xs[0] = 0.90000e0 * t44 + 0.90000e0 * t45;
  r->d3fdz2xs[1] = r->d3fdz2xs[0];
  r->d3fdzxs2[0] = -0.16200000000e1 * t41;
  r->d3fdzxs2[1] = r->d3fdzxs2[0];
  r->d3fdzxs2[2] = r->d3fdzxs2[1];
  r->d3fdxt3 = 0.0e0;
  r->d3fdxt2xs[0] = 0.0e0;
  r->d3fdxt2xs[1] = 0.0e0;
  r->d3fdxtxs2[0] = 0.97216200000000e-1 * t49;
  r->d3fdxtxs2[1] = r->d3fdxtxs2[0];
  r->d3fdxtxs2[2] = r->d3fdxtxs2[1];
  r->d3fdxs3[0] = -0.4374000000000000e1 * t38;
  r->d3fdxs3[1] = r->d3fdxs3[0];
  r->d3fdxs3[2] = r->d3fdxs3[1];
  r->d3fdxs3[3] = r->d3fdxs3[2];

  if(r->order < 4) return;


}

#define maple2c_order 3
#define maple2c_func  xc_gga_c_wl_func
