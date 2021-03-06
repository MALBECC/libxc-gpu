/* 
  This file was generated automatically with /nfs/data-012/marques/software/source/libxc/svn/scripts/maple2c.pl.
  Do not edit this file directly as it can be overwritten!!

  This Source Code Form is subject to the terms of the Mozilla Public
  License, v. 2.0. If a copy of the MPL was not distributed with this
  file, You can obtain one at http://mozilla.org/MPL/2.0/.

  Maple version     : Maple 2016 (X86 64 LINUX)
  Maple source      : ../maple/mgga_x_m08.mpl
  Type of functional: work_mgga_x
*/

static void 
xc_mgga_x_m08_enhance(const xc_func_type_cuda *pt, xc_mgga_work_x_t *r)
{
  double t1, t2, t3, t4, t5, t6, t7, t8;
  double t10, t13, t15, t16, t18, t19, t20, t21;
  double t22, t24, t25, t26, t27, t28, t30, t31;
  double t32, t33, t34, t36, t37, t38, t39, t40;
  double t42, t43, t44, t45, t46, t48, t49, t50;
  double t52, t54, t55, t56, t58, t60, t61, t62;
  double t63, t64, t66, t67, t68, t70, t72, t73;
  double t74, t76, t78, t79, t80, t82, t84, t87;
  double t89, t91, t92, t94, t95, t97, t98, t100;
  double t101, t103, t104, t106, t107, t109, t110, t112;
  double t113, t115, t116, t118, t119, t121, t122, t124;
  double t126, t128, t129, t133, t139, t144, t149, t154;
  double t159, t162, t165, t170, t175, t180, t185, t189;
  double t192, t193, t197, t202, t207, t212, t217, t220;
  double t223, t228, t233, t238, t243, t248, t249, t255;
  double t312, t340, t347, t391, t424;

  mgga_x_m08_params *params;

  assert(pt->params != NULL);
  params = (mgga_x_m08_params * ) (pt->params);

  t1 = M_CBRT6;
  t2 = 0.31415926535897932385e1 * 0.31415926535897932385e1;
  t3 = POW_1_3(t2);
  t4 = t3 * t3;
  t5 = 0.1e1 / t4;
  t6 = t1 * t5;
  t7 = r->x * r->x;
  t8 = t6 * t7;
  t10 = 0.8040e0 + 0.91462500000000000000e-2 * t8;
  t13 = 0.18040e1 - 0.64641600e0 / t10;
  t15 = params->a[1];
  t16 = t1 * t1;
  t18 = 0.3e1 / 0.10e2 * t16 * t4;
  t19 = t18 - r->t;
  t20 = t15 * t19;
  t21 = t18 + r->t;
  t22 = 0.1e1 / t21;
  t24 = params->a[2];
  t25 = t19 * t19;
  t26 = t24 * t25;
  t27 = t21 * t21;
  t28 = 0.1e1 / t27;
  t30 = params->a[3];
  t31 = t25 * t19;
  t32 = t30 * t31;
  t33 = t27 * t21;
  t34 = 0.1e1 / t33;
  t36 = params->a[4];
  t37 = t25 * t25;
  t38 = t36 * t37;
  t39 = t27 * t27;
  t40 = 0.1e1 / t39;
  t42 = params->a[5];
  t43 = t37 * t19;
  t44 = t42 * t43;
  t45 = t39 * t21;
  t46 = 0.1e1 / t45;
  t48 = params->a[6];
  t49 = t37 * t25;
  t50 = t48 * t49;
  t52 = 0.1e1 / t39 / t27;
  t54 = params->a[7];
  t55 = t37 * t31;
  t56 = t54 * t55;
  t58 = 0.1e1 / t39 / t33;
  t60 = params->a[8];
  t61 = t37 * t37;
  t62 = t60 * t61;
  t63 = t39 * t39;
  t64 = 0.1e1 / t63;
  t66 = params->a[9];
  t67 = t61 * t19;
  t68 = t66 * t67;
  t70 = 0.1e1 / t63 / t21;
  t72 = params->a[10];
  t73 = t61 * t25;
  t74 = t72 * t73;
  t76 = 0.1e1 / t63 / t27;
  t78 = params->a[11];
  t79 = t61 * t31;
  t80 = t78 * t79;
  t82 = 0.1e1 / t63 / t33;
  t84 = t20 * t22 + t26 * t28 + t32 * t34 + t38 * t40 + t44 * t46 + t50 * t52 + t56 * t58 + t62 * t64 + t68 * t70 + t74 * t76 + t80 * t82 + params->a[0];
  t87 = exp(-0.93189002206715572255e-2 * t8);
  t89 = 0.1552e1 - 0.552e0 * t87;
  t91 = params->b[1];
  t92 = t91 * t19;
  t94 = params->b[2];
  t95 = t94 * t25;
  t97 = params->b[3];
  t98 = t97 * t31;
  t100 = params->b[4];
  t101 = t100 * t37;
  t103 = params->b[5];
  t104 = t103 * t43;
  t106 = params->b[6];
  t107 = t106 * t49;
  t109 = params->b[7];
  t110 = t109 * t55;
  t112 = params->b[8];
  t113 = t112 * t61;
  t115 = params->b[9];
  t116 = t115 * t67;
  t118 = params->b[10];
  t119 = t118 * t73;
  t121 = params->b[11];
  t122 = t121 * t79;
  t124 = t101 * t40 + t104 * t46 + t107 * t52 + t110 * t58 + t113 * t64 + t116 * t70 + t119 * t76 + t122 * t82 + t92 * t22 + t95 * t28 + t98 * t34 + params->b[0];
  r->f = t89 * t124 + t13 * t84;

  if(r->order < 1) return;

  r->dfdrs = 0.0e0;
  t126 = t10 * t10;
  t128 = 0.1e1 / t126 * t1;
  t129 = t5 * r->x;
  t133 = r->x * t87;
  r->dfdx = 0.11824564680000000000e-1 * t128 * t129 * t84 + 0.10288065843621399177e-1 * t6 * t133 * t124;
  t139 = t24 * t19;
  t144 = t30 * t25;
  t149 = t36 * t31;
  t154 = t42 * t37;
  t159 = t48 * t43;
  t162 = -0.2e1 * t139 * t28 - 0.3e1 * t144 * t34 - 0.4e1 * t149 * t40 - t15 * t22 - 0.5e1 * t154 * t46 - 0.6e1 * t159 * t52 - t20 * t28 - 0.2e1 * t26 * t34 - 0.3e1 * t32 * t40 - 0.4e1 * t38 * t46 - 0.5e1 * t44 * t52;
  t165 = t54 * t49;
  t170 = t60 * t55;
  t175 = t66 * t61;
  t180 = t72 * t67;
  t185 = t78 * t73;
  t189 = 0.1e1 / t63 / t39;
  t192 = -0.7e1 * t165 * t58 - 0.8e1 * t170 * t64 - 0.9e1 * t175 * t70 - 0.10e2 * t180 * t76 - 0.11e2 * t185 * t82 - 0.11e2 * t80 * t189 - 0.6e1 * t50 * t58 - 0.7e1 * t56 * t64 - 0.8e1 * t62 * t70 - 0.9e1 * t68 * t76 - 0.10e2 * t74 * t82;
  t193 = t162 + t192;
  t197 = t94 * t19;
  t202 = t97 * t25;
  t207 = t100 * t31;
  t212 = t103 * t37;
  t217 = t106 * t43;
  t220 = -0.4e1 * t101 * t46 - 0.5e1 * t104 * t52 - 0.2e1 * t197 * t28 - 0.3e1 * t202 * t34 - 0.4e1 * t207 * t40 - 0.5e1 * t212 * t46 - 0.6e1 * t217 * t52 - t91 * t22 - t92 * t28 - 0.2e1 * t95 * t34 - 0.3e1 * t98 * t40;
  t223 = t109 * t49;
  t228 = t112 * t55;
  t233 = t115 * t61;
  t238 = t118 * t67;
  t243 = t121 * t73;
  t248 = -0.6e1 * t107 * t58 - 0.7e1 * t110 * t64 - 0.8e1 * t113 * t70 - 0.9e1 * t116 * t76 - 0.10e2 * t119 * t82 - 0.11e2 * t122 * t189 - 0.7e1 * t223 * t58 - 0.8e1 * t228 * t64 - 0.9e1 * t233 * t70 - 0.10e2 * t238 * t76 - 0.11e2 * t243 * t82;
  t249 = t220 + t248;
  r->dfdt = t13 * t193 + t89 * t249;
  r->dfdu = 0.0e0;

  if(r->order < 2) return;

  r->d2fdrs2 = 0.0e0;
  t255 = 0.1e1 / t3 / t2;
  r->d2fdx2 = -0.43260169881780000000e-3 / t126 / t10 * t16 * t255 * t7 * t84 + 0.11824564680000000000e-1 * t128 * t5 * t84 + 0.10288065843621399177e-1 * t6 * t87 * t124 - 0.19174691812081393468e-3 * t16 * t255 * t7 * t87 * t124;
  t312 = 0.6e1 * t30 * t19 * t34 + 0.12e2 * t36 * t25 * t40 + 0.20e2 * t42 * t31 * t46 + 0.30e2 * t48 * t37 * t52 + 0.42e2 * t54 * t43 * t58 + 0.56e2 * t60 * t49 * t64 + 0.72e2 * t66 * t55 * t70 + 0.90e2 * t72 * t61 * t76 + 0.110e3 * t78 * t67 * t82 + 0.2e1 * t20 * t34 + 0.8e1 * t139 * t34 + 0.6e1 * t26 * t40 + 0.18e2 * t144 * t40 + 0.12e2 * t32 * t46 + 0.32e2 * t149 * t46 + 0.20e2 * t38 * t52;
  t340 = 0.1e1 / t63 / t45;
  t347 = 0.2e1 * t15 * t28 + 0.50e2 * t154 * t52 + 0.72e2 * t159 * t58 + 0.98e2 * t165 * t64 + 0.128e3 * t170 * t70 + 0.162e3 * t175 * t76 + 0.200e3 * t180 * t82 + 0.242e3 * t185 * t189 + 0.110e3 * t74 * t189 + 0.2e1 * t24 * t28 + 0.132e3 * t80 * t340 + 0.30e2 * t44 * t58 + 0.42e2 * t50 * t64 + 0.56e2 * t56 * t70 + 0.72e2 * t62 * t76 + 0.90e2 * t68 * t82;
  t391 = 0.72e2 * t113 * t76 + 0.162e3 * t233 * t76 + 0.90e2 * t116 * t82 + 0.200e3 * t238 * t82 + 0.110e3 * t119 * t189 + 0.242e3 * t243 * t189 + 0.132e3 * t122 * t340 + 0.6e1 * t97 * t19 * t34 + 0.12e2 * t100 * t25 * t40 + 0.20e2 * t103 * t31 * t46 + 0.30e2 * t106 * t37 * t52 + 0.42e2 * t109 * t43 * t58 + 0.56e2 * t112 * t49 * t64 + 0.72e2 * t115 * t55 * t70 + 0.90e2 * t118 * t61 * t76 + 0.110e3 * t121 * t67 * t82;
  t424 = 0.20e2 * t101 * t52 + 0.30e2 * t104 * t58 + 0.42e2 * t107 * t64 + 0.56e2 * t110 * t70 + 0.8e1 * t197 * t34 + 0.18e2 * t202 * t40 + 0.32e2 * t207 * t46 + 0.50e2 * t212 * t52 + 0.72e2 * t217 * t58 + 0.98e2 * t223 * t64 + 0.128e3 * t228 * t70 + 0.2e1 * t91 * t28 + 0.2e1 * t94 * t28 + 0.2e1 * t92 * t34 + 0.6e1 * t95 * t40 + 0.12e2 * t98 * t46;
  r->d2fdt2 = t13 * (t312 + t347) + t89 * (t391 + t424);
  r->d2fdu2 = 0.0e0;
  r->d2fdrsx = 0.0e0;
  r->d2fdrst = 0.0e0;
  r->d2fdrsu = 0.0e0;
  r->d2fdxt = 0.11824564680000000000e-1 * t128 * t129 * t193 + 0.10288065843621399177e-1 * t6 * t133 * t249;
  r->d2fdxu = 0.0e0;
  r->d2fdtu = 0.0e0;

  if(r->order < 3) return;


}

#define maple2c_order 3
#define maple2c_func  xc_mgga_x_m08_enhance
