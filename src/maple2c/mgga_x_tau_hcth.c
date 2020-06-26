/* 
  This file was generated automatically with /nfs/data-012/marques/software/source/libxc/svn/scripts/maple2c.pl.
  Do not edit this file directly as it can be overwritten!!

  This Source Code Form is subject to the terms of the Mozilla Public
  License, v. 2.0. If a copy of the MPL was not distributed with this
  file, You can obtain one at http://mozilla.org/MPL/2.0/.

  Maple version     : Maple 2016 (X86 64 LINUX)
  Maple source      : ../maple/mgga_x_tau_hcth.mpl
  Type of functional: work_mgga_x
*/

static void 
xc_mgga_x_tau_hcth_enhance(const xc_func_type_cuda *pt, xc_mgga_work_x_t *r)
{
  double t2, t3, t4, t6, t7, t10, t11, t12;
  double t13, t14, t17, t18, t19, t21, t25, t26;
  double t29, t30, t33, t34, t37, t38, t39, t40;
  double t41, t42, t44, t45, t46, t47, t49, t50;
  double t51, t52, t53, t56, t57, t58, t60, t62;
  double t67, t74, t81, t83, t84, t105, t107, t111;
  double t117, t120, t141, t144;

  mgga_x_tau_hcth_params *params;

  assert(pt->params != NULL);
  params = (mgga_x_tau_hcth_params * ) (pt->params);

  t2 = params->cx_local[1];
  t3 = r->x * r->x;
  t4 = t2 * t3;
  t6 = 0.1e1 + 0.4e-2 * t3;
  t7 = 0.1e1 / t6;
  t10 = params->cx_local[2];
  t11 = t3 * t3;
  t12 = t10 * t11;
  t13 = t6 * t6;
  t14 = 0.1e1 / t13;
  t17 = params->cx_local[3];
  t18 = t11 * t3;
  t19 = t17 * t18;
  t21 = 0.1e1 / t13 / t6;
  t25 = params->cx_nlocal[1];
  t26 = t25 * t3;
  t29 = params->cx_nlocal[2];
  t30 = t29 * t11;
  t33 = params->cx_nlocal[3];
  t34 = t33 * t18;
  t37 = params->cx_nlocal[0] + 0.4e-2 * t26 * t7 + 0.16e-4 * t30 * t14 + 0.64e-7 * t34 * t21;
  t38 = M_CBRT6;
  t39 = t38 * t38;
  t40 = 0.31415926535897932385e1 * 0.31415926535897932385e1;
  t41 = POW_1_3(t40);
  t42 = t41 * t41;
  t44 = 0.3e1 / 0.10e2 * t39 * t42;
  t45 = t44 - r->t;
  t46 = t44 + r->t;
  t47 = 0.1e1 / t46;
  t49 = t45 * t45;
  t50 = t49 * t45;
  t51 = t46 * t46;
  t52 = t51 * t46;
  t53 = 0.1e1 / t52;
  t56 = t49 * t49;
  t57 = t56 * t45;
  t58 = t51 * t51;
  t60 = 0.1e1 / t58 / t46;
  t62 = t45 * t47 - 0.2e1 * t50 * t53 + t57 * t60;
  r->f = params->cx_local[0] + 0.4e-2 * t4 * t7 + 0.16e-4 * t12 * t14 + 0.64e-7 * t19 * t21 + t37 * t62;

  if(r->order < 1) return;

  r->dfdrs = 0.0e0;
  t67 = t3 * r->x;
  t74 = t11 * r->x;
  t81 = t11 * t67;
  t83 = t13 * t13;
  t84 = 0.1e1 / t83;
  t105 = 0.8e-2 * t25 * r->x * t7 - 0.32e-4 * t25 * t67 * t14 + 0.64e-4 * t29 * t67 * t14 - 0.256e-6 * t29 * t74 * t21 + 0.384e-6 * t33 * t74 * t21 - 0.1536e-8 * t33 * t81 * t84;
  r->dfdx = 0.8e-2 * t2 * r->x * t7 - 0.32e-4 * t2 * t67 * t14 + 0.64e-4 * t10 * t67 * t14 - 0.256e-6 * t10 * t74 * t21 + 0.384e-6 * t17 * t74 * t21 - 0.1536e-8 * t17 * t81 * t84 + t105 * t62;
  t107 = 0.1e1 / t51;
  t111 = 0.1e1 / t58;
  t117 = 0.1e1 / t58 / t51;
  t120 = -t45 * t107 + 0.6e1 * t50 * t111 - 0.5e1 * t57 * t117 + 0.6e1 * t49 * t53 - 0.5e1 * t56 * t60 - t47;
  r->dfdt = t37 * t120;
  r->dfdu = 0.0e0;

  if(r->order < 2) return;

  r->d2fdrs2 = 0.0e0;
  t141 = t11 * t11;
  t144 = 0.1e1 / t83 / t6;
  r->d2fdx2 = 0.8e-2 * t2 * t7 - 0.160e-3 * t4 * t14 + 0.512e-6 * t2 * t11 * t21 + 0.192e-3 * t10 * t3 * t14 - 0.2304e-5 * t12 * t21 + 0.6144e-8 * t10 * t18 * t84 + 0.1920e-5 * t17 * t11 * t21 - 0.19968e-7 * t19 * t84 + 0.49152e-10 * t17 * t141 * t144 + (0.8e-2 * t25 * t7 - 0.160e-3 * t26 * t14 + 0.512e-6 * t25 * t11 * t21 + 0.192e-3 * t29 * t3 * t14 - 0.2304e-5 * t30 * t21 + 0.6144e-8 * t29 * t18 * t84 + 0.1920e-5 * t33 * t11 * t21 - 0.19968e-7 * t34 * t84 + 0.49152e-10 * t33 * t141 * t144) * t62;
  r->d2fdt2 = t37 * (0.2e1 * t107 - 0.10e2 * t45 * t53 - 0.36e2 * t49 * t111 - 0.4e1 * t50 * t60 + 0.50e2 * t56 * t117 + 0.30e2 * t57 / t58 / t52);
  r->d2fdu2 = 0.0e0;
  r->d2fdrsx = 0.0e0;
  r->d2fdrst = 0.0e0;
  r->d2fdrsu = 0.0e0;
  r->d2fdxt = t105 * t120;
  r->d2fdxu = 0.0e0;
  r->d2fdtu = 0.0e0;

  if(r->order < 3) return;


}

#define maple2c_order 3
#define maple2c_func  xc_mgga_x_tau_hcth_enhance
