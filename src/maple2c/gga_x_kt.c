/* 
  This file was generated automatically with /nfs/data-012/marques/software/source/libxc/svn/scripts/maple2c.pl.
  Do not edit this file directly as it can be overwritten!!

  This Source Code Form is subject to the terms of the Mozilla Public
  License, v. 2.0. If a copy of the MPL was not distributed with this
  file, You can obtain one at http://mozilla.org/MPL/2.0/.

  Maple version     : Maple 2016 (X86 64 LINUX)
  Maple source      : ../maple/gga_x_kt.mpl
  Type of functional: work_gga_c
*/

void xc_gga_x_kt_func
  (const xc_func_type_cuda *p, xc_gga_work_c_t *r)
{
  double t1, t2, t3, t5, t6, t8, t9, t10;
  double t11, t12, t13, t14, t15, t16, t17, t18;
  double t19, t20, t21, t23, t26, t27, t32, t33;
  double t35, t36, t37, t38, t39, t40, t44, t45;
  double t50, t51, t55, t58, t59, t60, t61, t62;
  double t66, t67, t68, t69, t71, t72, t73, t74;
  double t79, t82, t83, t84, t85, t89, t90, t91;
  double t95, t96, t98, t99, t100, t101, t106, t109;
  double t110, t111, t115, t120, t128, t131, t132, t142;
  double t145, t146, t150, t152, t153, t154, t155, t156;
  double t157, t159, t161, t162, t163, t166, t167, t169;
  double t171, t172, t175, t177, t178, t179, t180, t185;
  double t189, t190, t191, t195, t196, t197, t198, t199;
  double t201, t204, t205, t212, t215, t218, t219, t220;
  double t223, t224, t227, t229, t230, t235, t239, t243;
  double t244, t245, t248, t249, t256, t259, t262, t263;
  double t266, t270, t277, t278, t279, t280, t290, t291;
  double t292, t294, t296, t303, t308, t311, t312, t313;
  double t316, t319, t320, t330, t331, t332, t334, t341;
  double t346, t349, t350, t353, t357, t362, t365, t370;
  double t376, t379, t383, t386, t391, t397, t403, t410;
  double t411, t413, t415, t419, t424, t425, t434, t435;
  double t437, t441, t446, t447, t451, t453, t460, t465;
  double t467, t474, t479, t481, t482, t485, t488, t490;
  double t492, t494, t500, t505, t506, t511, t519, t526;
  double t527, t530, t532, t536, t546, t554, t558, t562;
  double t566, t568, t570, t576, t580, t591, t598, t602;
  double t612, t629, t640, t641, t648, t662, t675, t685;
  double t698, t705, t729, t750, t760, t761, t777, t789;
  double t797, t805, t831, t845, t851, t857, t871, t874;
  double t887, t893, t907, t910, t923, t929, t955, t959;
  double t965, t995, t1016, t1036, t1047, t1091, t1112;

  gga_x_kt_params *params;
 
  assert(p->params != NULL);
  params = (gga_x_kt_params * )(p->params);

  t1 = M_CBRT3;
  t2 = M_CBRT4;
  t3 = t2 * t2;
  t5 = 0.1e1 / 0.31415926535897932385e1;
  t6 = cbrt(t5);
  t8 = 0.3e1 / 0.8e1 * t1 * t3 * t6;
  t9 = params->gamma * t1;
  t10 = cbrt(0.8e1);
  t11 = t10 * t10;
  t12 = t9 * t11;
  t13 = 0.1e1 + r->z;
  t14 = t13 * t5;
  t15 = r->rs * r->rs;
  t16 = t15 * r->rs;
  t17 = 0.1e1 / t16;
  t18 = t14 * t17;
  t19 = cbrt(t18);
  t20 = t19 * t18;
  t21 = r->xs[0] * r->xs[0];
  t23 = t1 * t11;
  t26 = 0.3e1 / 0.64e2 * t23 * t20 + params->delta;
  t27 = 0.1e1 / t26;
  t32 = (t8 - 0.3e1 / 0.64e2 * t12 * t20 * t21 * t27) * t1;
  t33 = t11 * t20;
  t35 = 0.1e1 - r->z;
  t36 = t35 * t5;
  t37 = t36 * t17;
  t38 = cbrt(t37);
  t39 = t38 * t37;
  t40 = r->xs[1] * r->xs[1];
  t44 = 0.3e1 / 0.64e2 * t23 * t39 + params->delta;
  t45 = 0.1e1 / t44;
  t50 = (t8 - 0.3e1 / 0.64e2 * t12 * t39 * t40 * t45) * t1;
  t51 = t11 * t39;
  t55 = (-0.3e1 / 0.64e2 * t32 * t33 - 0.3e1 / 0.64e2 * t50 * t51) * 0.31415926535897932385e1;
  r->f = 0.4e1 / 0.3e1 * t55 * t16;

  if(r->order < 1) return;

  t58 = t9 * t11 * t19;
  t59 = t21 * t27;
  t60 = t15 * t15;
  t61 = 0.1e1 / t60;
  t62 = t14 * t61;
  t66 = t1 * t1;
  t67 = params->gamma * t66;
  t68 = t19 * t19;
  t69 = t68 * t18;
  t71 = t67 * t10 * t69;
  t72 = t26 * t26;
  t73 = 0.1e1 / t72;
  t74 = t21 * t73;
  t79 = (0.3e1 / 0.16e2 * t58 * t59 * t62 - 0.9e1 / 0.128e3 * t71 * t74 * t62) * t1;
  t82 = t32 * t11;
  t83 = t19 * t13;
  t84 = t5 * t61;
  t85 = t83 * t84;
  t89 = t9 * t11 * t38;
  t90 = t40 * t45;
  t91 = t36 * t61;
  t95 = t38 * t38;
  t96 = t95 * t37;
  t98 = t67 * t10 * t96;
  t99 = t44 * t44;
  t100 = 0.1e1 / t99;
  t101 = t40 * t100;
  t106 = (0.3e1 / 0.16e2 * t89 * t90 * t91 - 0.9e1 / 0.128e3 * t98 * t101 * t91) * t1;
  t109 = t50 * t11;
  t110 = t38 * t35;
  t111 = t110 * t84;
  t115 = (-0.3e1 / 0.64e2 * t79 * t33 + 0.3e1 / 0.16e2 * t82 * t85 - 0.3e1 / 0.64e2 * t106 * t51 + 0.3e1 / 0.16e2 * t109 * t111) * 0.31415926535897932385e1;
  r->dfdrs = 0.4e1 / 0.3e1 * t115 * t16 + 0.4e1 * t55 * t15;
  t120 = t5 * t17;
  t128 = (-t58 * t59 * t120 / 0.16e2 + 0.3e1 / 0.128e3 * t71 * t74 * t120) * t1;
  t131 = t19 * t5;
  t132 = t131 * t17;
  t142 = (t89 * t90 * t120 / 0.16e2 - 0.3e1 / 0.128e3 * t98 * t101 * t120) * t1;
  t145 = t38 * t5;
  t146 = t145 * t17;
  t150 = (-0.3e1 / 0.64e2 * t128 * t33 - t82 * t132 / 0.16e2 - 0.3e1 / 0.64e2 * t142 * t51 + t109 * t146 / 0.16e2) * 0.31415926535897932385e1;
  r->dfdz = 0.4e1 / 0.3e1 * t150 * t16;
  r->dfdxt = 0.0e0;
  t152 = t13 * t13;
  t153 = 0.31415926535897932385e1 * 0.31415926535897932385e1;
  t154 = 0.1e1 / t153;
  t155 = t152 * t154;
  t156 = t60 * t15;
  t157 = 0.1e1 / t156;
  t159 = t68 * t155 * t157;
  t161 = t67 * t10 * t159;
  t162 = r->xs[0] * t27;
  t163 = 0.31415926535897932385e1 * t16;
  r->dfdxs[0] = 0.3e1 / 0.64e2 * t161 * t162 * t163;
  t166 = t35 * t35;
  t167 = t166 * t154;
  t169 = t95 * t167 * t157;
  t171 = t67 * t10 * t169;
  t172 = r->xs[1] * t45;
  r->dfdxs[1] = 0.3e1 / 0.64e2 * t171 * t172 * t163;

  if(r->order < 2) return;

  t175 = 0.1e1 / t68;
  t177 = t9 * t11 * t175;
  t178 = t60 * t60;
  t179 = 0.1e1 / t178;
  t180 = t155 * t179;
  t185 = t67 * t10 * t68;
  t189 = t60 * r->rs;
  t190 = 0.1e1 / t189;
  t191 = t14 * t190;
  t195 = t152 * t152;
  t196 = params->gamma * t195;
  t197 = t153 * t153;
  t198 = 0.1e1 / t197;
  t199 = t196 * t198;
  t201 = 0.1e1 / t178 / t156;
  t204 = 0.1e1 / t72 / t26;
  t205 = t201 * t21 * t204;
  t212 = (-0.3e1 / 0.16e2 * t177 * t59 * t180 + 0.81e2 / 0.128e3 * t185 * t74 * t180 - 0.3e1 / 0.4e1 * t58 * t59 * t191 - 0.81e2 / 0.128e3 * t199 * t205 + 0.9e1 / 0.32e2 * t71 * t74 * t191) * t1;
  t215 = t79 * t11;
  t218 = t175 * t152;
  t219 = t154 * t179;
  t220 = t218 * t219;
  t223 = t5 * t190;
  t224 = t83 * t223;
  t227 = 0.1e1 / t95;
  t229 = t9 * t11 * t227;
  t230 = t167 * t179;
  t235 = t67 * t10 * t95;
  t239 = t36 * t190;
  t243 = t166 * t166;
  t244 = params->gamma * t243;
  t245 = t244 * t198;
  t248 = 0.1e1 / t99 / t44;
  t249 = t201 * t40 * t248;
  t256 = (-0.3e1 / 0.16e2 * t229 * t90 * t230 + 0.81e2 / 0.128e3 * t235 * t101 * t230 - 0.3e1 / 0.4e1 * t89 * t90 * t239 - 0.81e2 / 0.128e3 * t245 * t249 + 0.9e1 / 0.32e2 * t98 * t101 * t239) * t1;
  t259 = t106 * t11;
  t262 = t227 * t166;
  t263 = t262 * t219;
  t266 = t110 * t223;
  t270 = (-0.3e1 / 0.64e2 * t212 * t33 + 0.3e1 / 0.8e1 * t215 * t85 - 0.3e1 / 0.16e2 * t82 * t220 - 0.3e1 / 0.4e1 * t82 * t224 - 0.3e1 / 0.64e2 * t256 * t51 + 0.3e1 / 0.8e1 * t259 * t111 - 0.3e1 / 0.16e2 * t109 * t263 - 0.3e1 / 0.4e1 * t109 * t266) * 0.31415926535897932385e1;
  r->d2fdrs2 = 0.4e1 / 0.3e1 * t270 * t16 + 0.8e1 * t115 * t15 + 0.8e1 * t55 * r->rs;
  t277 = t13 * t154;
  t278 = t60 * t16;
  t279 = 0.1e1 / t278;
  t280 = t277 * t279;
  t290 = t152 * t13;
  t291 = params->gamma * t290;
  t292 = t291 * t198;
  t294 = 0.1e1 / t178 / t189;
  t296 = t294 * t21 * t204;
  t303 = (t177 * t59 * t280 / 0.16e2 - 0.27e2 / 0.128e3 * t185 * t74 * t280 + 0.3e1 / 0.16e2 * t58 * t59 * t84 + 0.27e2 / 0.128e3 * t292 * t296 - 0.9e1 / 0.128e3 * t71 * t74 * t84) * t1;
  t308 = t128 * t11;
  t311 = t175 * t13;
  t312 = t154 * t279;
  t313 = t311 * t312;
  t316 = t131 * t61;
  t319 = t35 * t154;
  t320 = t319 * t279;
  t330 = t166 * t35;
  t331 = params->gamma * t330;
  t332 = t331 * t198;
  t334 = t294 * t40 * t248;
  t341 = (-t229 * t90 * t320 / 0.16e2 + 0.27e2 / 0.128e3 * t235 * t101 * t320 - 0.3e1 / 0.16e2 * t89 * t90 * t84 - 0.27e2 / 0.128e3 * t332 * t334 + 0.9e1 / 0.128e3 * t98 * t101 * t84) * t1;
  t346 = t142 * t11;
  t349 = t227 * t35;
  t350 = t349 * t312;
  t353 = t145 * t61;
  t357 = (-0.3e1 / 0.64e2 * t303 * t33 - t215 * t132 / 0.16e2 + 0.3e1 / 0.16e2 * t308 * t85 + t82 * t313 / 0.16e2 + 0.3e1 / 0.16e2 * t82 * t316 - 0.3e1 / 0.64e2 * t341 * t51 + t259 * t146 / 0.16e2 + 0.3e1 / 0.16e2 * t346 * t111 - t109 * t350 / 0.16e2 - 0.3e1 / 0.16e2 * t109 * t353) * 0.31415926535897932385e1;
  r->d2fdrsz = 0.4e1 / 0.3e1 * t357 * t16 + 0.4e1 * t150 * t15;
  r->d2fdrsxt = 0.0e0;
  t362 = t162 * t62;
  t365 = r->xs[0] * t73;
  t370 = (0.3e1 / 0.8e1 * t58 * t362 - 0.9e1 / 0.64e2 * t71 * t365 * t62) * t1;
  t376 = (-0.3e1 / 0.64e2 * t370 * t33 - 0.9e1 / 0.64e2 * t71 * t362) * 0.31415926535897932385e1;
  t379 = 0.31415926535897932385e1 * t15;
  r->d2fdrsxs[0] = 0.4e1 / 0.3e1 * t376 * t16 + 0.9e1 / 0.64e2 * t161 * t162 * t379;
  t383 = t172 * t91;
  t386 = r->xs[1] * t100;
  t391 = (0.3e1 / 0.8e1 * t89 * t383 - 0.9e1 / 0.64e2 * t98 * t386 * t91) * t1;
  t397 = (-0.3e1 / 0.64e2 * t391 * t51 - 0.9e1 / 0.64e2 * t98 * t383) * 0.31415926535897932385e1;
  r->d2fdrsxs[1] = 0.4e1 / 0.3e1 * t397 * t16 + 0.9e1 / 0.64e2 * t171 * t172 * t379;
  t403 = t154 * t157;
  t410 = params->gamma * t152;
  t411 = t410 * t198;
  t413 = 0.1e1 / t178 / t60;
  t415 = t413 * t21 * t204;
  t419 = (-t177 * t59 * t403 / 0.48e2 + 0.9e1 / 0.128e3 * t185 * t74 * t403 - 0.9e1 / 0.128e3 * t411 * t415) * t1;
  t424 = t175 * t154;
  t425 = t424 * t157;
  t434 = params->gamma * t166;
  t435 = t434 * t198;
  t437 = t413 * t40 * t248;
  t441 = (-t229 * t90 * t403 / 0.48e2 + 0.9e1 / 0.128e3 * t235 * t101 * t403 - 0.9e1 / 0.128e3 * t435 * t437) * t1;
  t446 = t227 * t154;
  t447 = t446 * t157;
  t451 = (-0.3e1 / 0.64e2 * t419 * t33 - t308 * t132 / 0.8e1 - t82 * t425 / 0.48e2 - 0.3e1 / 0.64e2 * t441 * t51 + t346 * t146 / 0.8e1 - t109 * t447 / 0.48e2) * 0.31415926535897932385e1;
  r->d2fdz2 = 0.4e1 / 0.3e1 * t451 * t16;
  r->d2fdzxt = 0.0e0;
  t453 = t162 * t120;
  t460 = (-t58 * t453 / 0.8e1 + 0.3e1 / 0.64e2 * t71 * t365 * t120) * t1;
  t465 = (-0.3e1 / 0.64e2 * t460 * t33 + 0.3e1 / 0.64e2 * t71 * t453) * 0.31415926535897932385e1;
  r->d2fdzxs[0] = 0.4e1 / 0.3e1 * t465 * t16;
  t467 = t172 * t120;
  t474 = (t89 * t467 / 0.8e1 - 0.3e1 / 0.64e2 * t98 * t386 * t120) * t1;
  t479 = (-0.3e1 / 0.64e2 * t98 * t467 - 0.3e1 / 0.64e2 * t474 * t51) * 0.31415926535897932385e1;
  r->d2fdzxs[1] = 0.4e1 / 0.3e1 * t479 * t16;
  r->d2fdxt2 = 0.0e0;
  r->d2fdxtxs[0] = 0.0e0;
  r->d2fdxtxs[1] = 0.0e0;
  t481 = t67 * t10;
  t482 = t159 * t27;
  r->d2fdxs2[0] = 0.3e1 / 0.64e2 * t481 * t482 * t163;
  r->d2fdxs2[1] = 0.0e0;
  t485 = t169 * t45;
  r->d2fdxs2[2] = 0.3e1 / 0.64e2 * t481 * t485 * t163;

  if(r->order < 3) return;

  t488 = 0.1e1 / t69;
  t490 = t9 * t11 * t488;
  t492 = 0.1e1 / t153 / 0.31415926535897932385e1;
  t494 = t290 * t492 * t413;
  t500 = t67 * t10 / t19;
  t505 = 0.1e1 / t178 / r->rs;
  t506 = t155 * t505;
  t511 = 0.1e1 / t178 / t278;
  t519 = t14 * t157;
  t526 = 0.1e1 / t197 / 0.31415926535897932385e1;
  t527 = t178 * t178;
  t530 = t526 / t527 / t15;
  t532 = t72 * t72;
  t536 = t21 / t532 * t23 * t19;
  t546 = t212 * t11;
  t554 = t492 * t413;
  t558 = t154 * t505;
  t562 = t5 * t157;
  t566 = 0.1e1 / t96;
  t568 = t9 * t11 * t566;
  t570 = t330 * t492 * t413;
  t576 = t67 * t10 / t38;
  t580 = t167 * t505;
  t591 = t36 * t157;
  t598 = t99 * t99;
  t602 = t40 / t598 * t23 * t38;
  t612 = t256 * t11;
  t629 = -0.3e1 / 0.64e2 * (-0.3e1 / 0.8e1 * t490 * t59 * t494 - 0.99e2 / 0.64e2 * t500 * t74 * t494 + 0.9e1 / 0.4e1 * t177 * t59 * t506 + 0.2187e4 / 0.128e3 * t199 * t511 * t21 * t204 - 0.243e3 / 0.32e2 * t185 * t74 * t506 + 0.15e2 / 0.4e1 * t58 * t59 * t519 - 0.729e3 / 0.2048e4 * params->gamma * t195 * t13 * t530 * t536 - 0.45e2 / 0.32e2 * t71 * t74 * t519) * t1 * t33 + 0.9e1 / 0.16e2 * t546 * t85 - 0.9e1 / 0.16e2 * t215 * t220 - 0.9e1 / 0.4e1 * t215 * t224 - 0.3e1 / 0.8e1 * t82 * t488 * t290 * t554 + 0.9e1 / 0.4e1 * t82 * t218 * t558 + 0.15e2 / 0.4e1 * t82 * t83 * t562 - 0.3e1 / 0.64e2 * (-0.3e1 / 0.8e1 * t568 * t90 * t570 - 0.99e2 / 0.64e2 * t576 * t101 * t570 + 0.9e1 / 0.4e1 * t229 * t90 * t580 + 0.2187e4 / 0.128e3 * t245 * t511 * t40 * t248 - 0.243e3 / 0.32e2 * t235 * t101 * t580 + 0.15e2 / 0.4e1 * t89 * t90 * t591 - 0.729e3 / 0.2048e4 * params->gamma * t243 * t35 * t530 * t602 - 0.45e2 / 0.32e2 * t98 * t101 * t591) * t1 * t51 + 0.9e1 / 0.16e2 * t612 * t111 - 0.9e1 / 0.16e2 * t259 * t263 - 0.9e1 / 0.4e1 * t259 * t266 - 0.3e1 / 0.8e1 * t109 * t566 * t330 * t554 + 0.9e1 / 0.4e1 * t109 * t262 * t558 + 0.15e2 / 0.4e1 * t109 * t110 * t562;
  r->d3fdrs3 = 0.4e1 / 0.3e1 * t629 * 0.31415926535897932385e1 * t16 + 0.12e2 * t270 * t15 + 0.24e2 * t115 * r->rs + 0.8e1 * t55;
  t640 = 0.1e1 / t178 / t16;
  t641 = t152 * t492 * t640;
  t648 = t277 * t179;
  t662 = t526 / t527 / r->rs;
  t675 = t303 * t11;
  t685 = t492 * t640;
  t698 = t166 * t492 * t640;
  t705 = t319 * t179;
  t729 = t341 * t11;
  t750 = -0.3e1 / 0.64e2 * (t490 * t59 * t641 / 0.8e1 + 0.33e2 / 0.64e2 * t500 * t74 * t641 - 0.5e1 / 0.8e1 * t177 * t59 * t648 - 0.675e3 / 0.128e3 * t292 * t205 + 0.135e3 / 0.64e2 * t185 * t74 * t648 - 0.3e1 / 0.4e1 * t58 * t59 * t223 + 0.243e3 / 0.2048e4 * t196 * t662 * t536 + 0.9e1 / 0.32e2 * t71 * t74 * t223) * t1 * t33 - t546 * t132 / 0.16e2 + 0.3e1 / 0.8e1 * t675 * t85 + t215 * t313 / 0.8e1 + 0.3e1 / 0.8e1 * t215 * t316 - 0.3e1 / 0.16e2 * t308 * t220 + t82 * t488 * t152 * t685 / 0.8e1 - 0.5e1 / 0.8e1 * t82 * t311 * t219 - 0.3e1 / 0.4e1 * t308 * t224 - 0.3e1 / 0.4e1 * t82 * t131 * t190 - 0.3e1 / 0.64e2 * (-t568 * t90 * t698 / 0.8e1 - 0.33e2 / 0.64e2 * t576 * t101 * t698 + 0.5e1 / 0.8e1 * t229 * t90 * t705 + 0.675e3 / 0.128e3 * t332 * t249 - 0.135e3 / 0.64e2 * t235 * t101 * t705 + 0.3e1 / 0.4e1 * t89 * t90 * t223 - 0.243e3 / 0.2048e4 * t244 * t662 * t602 - 0.9e1 / 0.32e2 * t98 * t101 * t223) * t1 * t51 + t612 * t146 / 0.16e2 + 0.3e1 / 0.8e1 * t729 * t111 - t259 * t350 / 0.8e1 - 0.3e1 / 0.8e1 * t259 * t353 - 0.3e1 / 0.16e2 * t346 * t263 - t109 * t566 * t166 * t685 / 0.8e1 + 0.5e1 / 0.8e1 * t109 * t349 * t219 - 0.3e1 / 0.4e1 * t346 * t266 + 0.3e1 / 0.4e1 * t109 * t145 * t190;
  r->d3fdrs2z = 0.4e1 / 0.3e1 * t750 * 0.31415926535897932385e1 * t16 + 0.8e1 * t357 * t15 + 0.8e1 * t150 * r->rs;
  t760 = 0.1e1 / t178 / t15;
  t761 = t13 * t492 * t760;
  t777 = t526 / t527;
  t789 = t419 * t11;
  t797 = t492 * t760;
  t805 = t35 * t492 * t760;
  t831 = t441 * t11;
  t845 = -0.3e1 / 0.64e2 * (-t490 * t59 * t761 / 0.24e2 - 0.11e2 / 0.64e2 * t500 * t74 * t761 + t177 * t59 * t312 / 0.8e1 + 0.189e3 / 0.128e3 * t411 * t296 - 0.27e2 / 0.64e2 * t185 * t74 * t312 - 0.81e2 / 0.2048e4 * t291 * t777 * t536) * t1 * t33 - t675 * t132 / 0.8e1 - t215 * t425 / 0.48e2 + 0.3e1 / 0.16e2 * t789 * t85 + t308 * t313 / 0.8e1 + 0.3e1 / 0.8e1 * t308 * t316 - t82 * t488 * t13 * t797 / 0.24e2 + t82 * t424 * t279 / 0.8e1 - 0.3e1 / 0.64e2 * (-t568 * t90 * t805 / 0.24e2 - 0.11e2 / 0.64e2 * t576 * t101 * t805 + t229 * t90 * t312 / 0.8e1 + 0.189e3 / 0.128e3 * t435 * t334 - 0.27e2 / 0.64e2 * t235 * t101 * t312 - 0.81e2 / 0.2048e4 * t331 * t777 * t602) * t1 * t51 + t729 * t146 / 0.8e1 - t259 * t447 / 0.48e2 + 0.3e1 / 0.16e2 * t831 * t111 - t346 * t350 / 0.8e1 - 0.3e1 / 0.8e1 * t346 * t353 - t109 * t566 * t35 * t797 / 0.24e2 + t109 * t446 * t279 / 0.8e1;
  r->d3fdrsz2 = 0.4e1 / 0.3e1 * t845 * 0.31415926535897932385e1 * t16 + 0.4e1 * t451 * t15;
  r->d3fdrszxt = 0.0e0;
  t851 = t162 * t280;
  t857 = t162 * t84;
  t871 = t370 * t11;
  t874 = t460 * t11;
  r->d3fdrszxs[0] = 0.4e1 / 0.3e1 * (-0.3e1 / 0.64e2 * (t177 * t851 / 0.8e1 - 0.27e2 / 0.64e2 * t185 * t365 * t280 + 0.3e1 / 0.8e1 * t58 * t857 + 0.27e2 / 0.64e2 * t292 * t294 * r->xs[0] * t204 - 0.9e1 / 0.64e2 * t71 * t365 * t84) * t1 * t33 - t871 * t132 / 0.16e2 + 0.3e1 / 0.16e2 * t874 * t85 - 0.3e1 / 0.64e2 * t185 * t851 - 0.9e1 / 0.64e2 * t71 * t857) * 0.31415926535897932385e1 * t16 + 0.4e1 * t465 * t15;
  t887 = t172 * t320;
  t893 = t172 * t84;
  t907 = t391 * t11;
  t910 = t474 * t11;
  r->d3fdrszxs[1] = 0.4e1 / 0.3e1 * (-0.3e1 / 0.64e2 * (-t229 * t887 / 0.8e1 + 0.27e2 / 0.64e2 * t235 * t386 * t320 - 0.3e1 / 0.8e1 * t89 * t893 - 0.27e2 / 0.64e2 * t332 * t294 * r->xs[1] * t248 + 0.9e1 / 0.64e2 * t98 * t386 * t84) * t1 * t51 + t907 * t146 / 0.16e2 + 0.3e1 / 0.16e2 * t910 * t111 + 0.3e1 / 0.64e2 * t235 * t887 + 0.9e1 / 0.64e2 * t98 * t893) * 0.31415926535897932385e1 * t16 + 0.4e1 * t479 * t15;
  r->d3fdrs2xt = 0.0e0;
  r->d3fdrsxt2 = 0.0e0;
  r->d3fdrsxtxs[0] = 0.0e0;
  r->d3fdrsxtxs[1] = 0.0e0;
  t923 = t162 * t180;
  t929 = t162 * t191;
  t955 = 0.31415926535897932385e1 * r->rs;
  r->d3fdrs2xs[0] = 0.4e1 / 0.3e1 * (-0.3e1 / 0.64e2 * (-0.3e1 / 0.8e1 * t177 * t923 + 0.81e2 / 0.64e2 * t185 * t365 * t180 - 0.3e1 / 0.2e1 * t58 * t929 - 0.81e2 / 0.64e2 * t199 * t201 * r->xs[0] * t204 + 0.9e1 / 0.16e2 * t71 * t365 * t191) * t1 * t33 + 0.3e1 / 0.8e1 * t871 * t85 + 0.9e1 / 0.64e2 * t185 * t923 + 0.9e1 / 0.16e2 * t71 * t929) * 0.31415926535897932385e1 * t16 + 0.8e1 * t376 * t15 + 0.9e1 / 0.32e2 * t161 * t162 * t955;
  t959 = t172 * t230;
  t965 = t172 * t239;
  r->d3fdrs2xs[1] = 0.4e1 / 0.3e1 * (-0.3e1 / 0.64e2 * (-0.3e1 / 0.8e1 * t229 * t959 + 0.81e2 / 0.64e2 * t235 * t386 * t230 - 0.3e1 / 0.2e1 * t89 * t965 - 0.81e2 / 0.64e2 * t245 * t201 * r->xs[1] * t248 + 0.9e1 / 0.16e2 * t98 * t386 * t239) * t1 * t51 + 0.3e1 / 0.8e1 * t907 * t111 + 0.9e1 / 0.64e2 * t235 * t959 + 0.9e1 / 0.16e2 * t98 * t965) * 0.31415926535897932385e1 * t16 + 0.8e1 * t397 * t15 + 0.9e1 / 0.32e2 * t171 * t172 * t955;
  t995 = t27 * t13 * t84;
  r->d3fdrsxs2[0] = 0.4e1 / 0.3e1 * (-0.3e1 / 0.64e2 * (0.3e1 / 0.8e1 * t58 * t995 - 0.9e1 / 0.64e2 * t71 * t73 * t13 * t84) * t1 * t33 - 0.9e1 / 0.64e2 * t71 * t995) * 0.31415926535897932385e1 * t16 + 0.9e1 / 0.64e2 * t481 * t482 * t379;
  r->d3fdrsxs2[1] = 0.0e0;
  t1016 = t45 * t35 * t84;
  r->d3fdrsxs2[2] = 0.4e1 / 0.3e1 * (-0.3e1 / 0.64e2 * (0.3e1 / 0.8e1 * t89 * t1016 - 0.9e1 / 0.64e2 * t98 * t100 * t35 * t84) * t1 * t51 - 0.9e1 / 0.64e2 * t98 * t1016) * 0.31415926535897932385e1 * t16 + 0.9e1 / 0.64e2 * t481 * t485 * t379;
  t1036 = t492 * t505;
  t1047 = t526 * t511;
  r->d3fdz3 = 0.4e1 / 0.3e1 * (-0.3e1 / 0.64e2 * (t490 * t59 * t1036 / 0.72e2 + 0.11e2 / 0.192e3 * t500 * t74 * t1036 - 0.45e2 / 0.128e3 * params->gamma * t13 * t198 * t415 + 0.27e2 / 0.2048e4 * t410 * t1047 * t536) * t1 * t33 - 0.3e1 / 0.16e2 * t789 * t132 - t308 * t425 / 0.16e2 + t82 * t488 * t492 * t505 / 0.72e2 - 0.3e1 / 0.64e2 * (-t568 * t90 * t1036 / 0.72e2 - 0.11e2 / 0.192e3 * t576 * t101 * t1036 + 0.45e2 / 0.128e3 * params->gamma * t35 * t198 * t437 - 0.27e2 / 0.2048e4 * t434 * t1047 * t602) * t1 * t51 + 0.3e1 / 0.16e2 * t831 * t146 - t346 * t447 / 0.16e2 - t109 * t566 * t492 * t505 / 0.72e2) * 0.31415926535897932385e1 * t16;
  r->d3fdz2xt = 0.0e0;
  r->d3fdzxt2 = 0.0e0;
  r->d3fdzxtxs[0] = 0.0e0;
  r->d3fdzxtxs[1] = 0.0e0;
  t1091 = t162 * t403;
  r->d3fdz2xs[0] = 0.4e1 / 0.3e1 * (-0.3e1 / 0.64e2 * (-t177 * t1091 / 0.24e2 + 0.9e1 / 0.64e2 * t185 * t365 * t403 - 0.9e1 / 0.64e2 * t411 * t413 * r->xs[0] * t204) * t1 * t33 - t874 * t132 / 0.8e1 + t185 * t1091 / 0.64e2) * 0.31415926535897932385e1 * t16;
  t1112 = t172 * t403;
  r->d3fdz2xs[1] = 0.4e1 / 0.3e1 * (-0.3e1 / 0.64e2 * (-t229 * t1112 / 0.24e2 + 0.9e1 / 0.64e2 * t235 * t386 * t403 - 0.9e1 / 0.64e2 * t435 * t413 * r->xs[1] * t248) * t1 * t51 + t910 * t146 / 0.8e1 + t235 * t1112 / 0.64e2) * 0.31415926535897932385e1 * t16;
  r->d3fdzxs2[0] = 0.4e1 / 0.3e1 * (-0.3e1 / 0.64e2 * (-t12 * t19 * t27 * t120 / 0.8e1 + 0.3e1 / 0.64e2 * t481 * t69 * t73 * t120) * t1 * t33 + 0.3e1 / 0.64e2 * t481 * t69 * t27 * t120) * 0.31415926535897932385e1 * t16;
  r->d3fdzxs2[1] = 0.0e0;
  r->d3fdzxs2[2] = 0.4e1 / 0.3e1 * (-0.3e1 / 0.64e2 * (t12 * t38 * t45 * t120 / 0.8e1 - 0.3e1 / 0.64e2 * t481 * t96 * t100 * t120) * t1 * t51 - 0.3e1 / 0.64e2 * t481 * t96 * t45 * t120) * 0.31415926535897932385e1 * t16;
  r->d3fdxt3 = 0.0e0;
  r->d3fdxt2xs[0] = 0.0e0;
  r->d3fdxt2xs[1] = 0.0e0;
  r->d3fdxtxs2[0] = 0.0e0;
  r->d3fdxtxs2[1] = 0.0e0;
  r->d3fdxtxs2[2] = 0.0e0;
  r->d3fdxs3[0] = 0.0e0;
  r->d3fdxs3[1] = 0.0e0;
  r->d3fdxs3[2] = 0.0e0;
  r->d3fdxs3[3] = 0.0e0;

  if(r->order < 4) return;


}

#define maple2c_order 3
#define maple2c_func  xc_gga_x_kt_func
