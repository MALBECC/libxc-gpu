/* 
  This file was generated automatically with /nfs/data-012/marques/software/source/libxc/svn/scripts/maple2c.pl.
  Do not edit this file directly as it can be overwritten!!

  This Source Code Form is subject to the terms of the Mozilla Public
  License, v. 2.0. If a copy of the MPL was not distributed with this
  file, You can obtain one at http://mozilla.org/MPL/2.0/.

  Maple version     : Maple 2016 (X86 64 LINUX)
  Maple source      : ../maple/mgga_c_cs.mpl
  Type of functional: work_mgga_c
*/

void xc_mgga_c_cs_func
  (const xc_func_type_cuda *p, xc_mgga_work_c_t *r)
{
  double t1, t2, t3, t4, t5, t6, t8, t9;
  double t11, t13, t14, t15, t17, t19, t20, t21;
  double t22, t23, t25, t28, t29, t30, t31, t32;
  double t34, t36, t38, t41, t44, t45, t47, t50;
  double t51, t52, t54, t57, t58, t59, t62, t73;
  double t74, t77, t79, t82, t86, t87, t91, t92;
  double t96, t97, t99, t101, t102, t103, t106, t107;
  double t108, t111, t114, t118, t120, t123, t126, t130;
  double t133, t137, t140, t144, t147, t151, t154, t158;
  double t161, t172, t175, t178, t179, t186, t191, t198;
  double t199, t206, t207, t211, t226, t234, t245, t315;
  double t329, t338, t347, t356, t361;


  t1 = r->z * r->z;
  t2 = -t1 + 0.1e1;
  t3 = POW_1_3(0.3e1);
  t4 = t3 * t3;
  t5 = POW_1_3(0.4e1);
  t6 = t4 * t5;
  t8 = POW_1_3(0.1e1 / 0.31415926535897932385e1);
  t9 = 0.1e1 / t8;
  t11 = t6 * t9 * r->rs;
  t13 = 0.1e1 + 0.11633333333333333333e0 * t11;
  t14 = 0.1e1 / t13;
  t15 = t2 * t14;
  t17 = exp(-0.84433333333333333333e-1 * t11);
  t19 = 0.1e1 / 0.2e1 + r->z / 0.2e1;
  t20 = t19 * t19;
  t21 = POW_1_3(t19);
  t22 = t21 * t21;
  t23 = t22 * t20;
  t25 = r->ts[0] - r->us[0] / 0.8e1;
  t28 = 0.1e1 / 0.2e1 - r->z / 0.2e1;
  t29 = t28 * t28;
  t30 = POW_1_3(t28);
  t31 = t30 * t30;
  t32 = t31 * t29;
  t34 = r->ts[1] - r->us[1] / 0.8e1;
  t36 = r->xt * r->xt;
  t38 = t22 * t19;
  t41 = t31 * t28;
  t44 = t23 * t25 + t32 * t34 - t36 / 0.8e1 + r->us[0] * t38 / 0.8e1 + r->us[1] * t41 / 0.8e1;
  t45 = t17 * t44;
  t47 = 0.1e1 + 0.264e0 * t45;
  r->f = -0.4918e-1 * t15 * t47;

  if(r->order < 1) return;

  t50 = t13 * t13;
  t51 = 0.1e1 / t50;
  t52 = t2 * t51;
  t54 = t6 * t9;
  t57 = t15 * t4;
  t58 = t5 * t9;
  t59 = t58 * t45;
  r->dfdrs = 0.57212733333333333332e-2 * t52 * t47 * t54 + 0.10962418720000000000e-2 * t57 * t59;
  t62 = r->z * t14;
  t73 = 0.4e1 / 0.3e1 * t38 * t25 - 0.4e1 / 0.3e1 * t41 * t34 + 0.5e1 / 0.48e2 * r->us[0] * t22 - 0.5e1 / 0.48e2 * r->us[1] * t31;
  t74 = t17 * t73;
  r->dfdz = 0.9836e-1 * t62 * t47 - 0.1298352e-1 * t15 * t74;
  t77 = t17 * r->xt;
  r->dfdxt = 0.32458800000000000000e-2 * t15 * t77;
  r->dfdxs[0] = 0.0e0;
  r->dfdxs[1] = 0.0e0;
  t79 = t17 * t23;
  r->dfdts[0] = -0.1298352e-1 * t15 * t79;
  t82 = t17 * t32;
  r->dfdts[1] = -0.1298352e-1 * t15 * t82;
  t86 = -t23 / 0.8e1 + t38 / 0.8e1;
  t87 = t17 * t86;
  r->dfdus[0] = -0.1298352e-1 * t15 * t87;
  t91 = -t32 / 0.8e1 + t41 / 0.8e1;
  t92 = t17 * t91;
  r->dfdus[1] = -0.1298352e-1 * t15 * t92;

  if(r->order < 2) return;

  t96 = 0.1e1 / t50 / t13;
  t97 = t2 * t96;
  t99 = t5 * t5;
  t101 = t8 * t8;
  t102 = 0.1e1 / t101;
  t103 = t3 * t99 * t102;
  t106 = t52 * t3;
  t107 = t99 * t102;
  t108 = t107 * t45;
  t111 = t15 * t3;
  r->d2fdrs2 = -0.39934487866666666665e-2 * t97 * t47 * t103 - 0.76517682665599999998e-3 * t106 * t108 - 0.27767806617760000000e-3 * t111 * t108;
  t114 = r->z * t51;
  t118 = t52 * t17;
  t120 = t73 * t4 * t58;
  t123 = t62 * t4;
  t126 = t58 * t74;
  r->d2fdrsz = -0.11442546666666666666e-1 * t114 * t47 * t54 + 0.15104161600000000000e-2 * t118 * t120 - 0.21924837440000000000e-2 * t123 * t59 + 0.10962418720000000000e-2 * t57 * t126;
  t130 = r->xt * t4 * t58;
  t133 = t58 * t77;
  r->d2fdrsxt = -0.37760403999999999999e-3 * t118 * t130 - 0.27406046800000000000e-3 * t57 * t133;
  r->d2fdrsxs[0] = 0.0e0;
  r->d2fdrsxs[1] = 0.0e0;
  t137 = t23 * t4 * t58;
  t140 = t58 * t79;
  r->d2fdrsts[0] = 0.15104161600000000000e-2 * t118 * t137 + 0.10962418720000000000e-2 * t57 * t140;
  t144 = t32 * t4 * t58;
  t147 = t58 * t82;
  r->d2fdrsts[1] = 0.15104161600000000000e-2 * t118 * t144 + 0.10962418720000000000e-2 * t57 * t147;
  t151 = t86 * t4 * t58;
  t154 = t58 * t87;
  r->d2fdrsus[0] = 0.15104161600000000000e-2 * t118 * t151 + 0.10962418720000000000e-2 * t57 * t154;
  t158 = t91 * t4 * t58;
  t161 = t58 * t92;
  r->d2fdrsus[1] = 0.15104161600000000000e-2 * t118 * t158 + 0.10962418720000000000e-2 * t57 * t161;
  t172 = 0.1e1 / t21;
  t175 = 0.1e1 / t30;
  t178 = 0.10e2 / 0.9e1 * t22 * t25 + 0.10e2 / 0.9e1 * t31 * t34 + 0.5e1 / 0.144e3 * r->us[0] * t172 + 0.5e1 / 0.144e3 * r->us[1] * t175;
  t179 = t17 * t178;
  r->d2fdz2 = 0.9836e-1 * t14 * t47 + 0.5193408e-1 * t62 * t74 - 0.1298352e-1 * t15 * t179;
  r->d2fdzxt = -0.64917600000000000000e-2 * t62 * t77;
  r->d2fdzxs[0] = 0.0e0;
  r->d2fdzxs[1] = 0.0e0;
  t186 = t17 * t38;
  r->d2fdzts[0] = 0.2596704e-1 * t62 * t79 - 0.17311360000000000000e-1 * t15 * t186;
  t191 = t17 * t41;
  r->d2fdzts[1] = 0.2596704e-1 * t62 * t82 + 0.17311360000000000000e-1 * t15 * t191;
  t198 = -t38 / 0.6e1 + 0.5e1 / 0.48e2 * t22;
  t199 = t17 * t198;
  r->d2fdzus[0] = 0.2596704e-1 * t62 * t87 - 0.1298352e-1 * t15 * t199;
  t206 = t41 / 0.6e1 - 0.5e1 / 0.48e2 * t31;
  t207 = t17 * t206;
  r->d2fdzus[1] = 0.2596704e-1 * t62 * t92 - 0.1298352e-1 * t15 * t207;
  r->d2fdxt2 = 0.32458800000000000000e-2 * t15 * t17;
  r->d2fdxtxs[0] = 0.0e0;
  r->d2fdxtxs[1] = 0.0e0;
  r->d2fdxtts[0] = 0.0e0;
  r->d2fdxtts[1] = 0.0e0;
  r->d2fdxtus[0] = 0.0e0;
  r->d2fdxtus[1] = 0.0e0;
  r->d2fdxs2[0] = 0.0e0;
  r->d2fdxs2[1] = 0.0e0;
  r->d2fdxs2[2] = 0.0e0;
  r->d2fdxsts[0] = 0.0e0;
  r->d2fdxsts[1] = 0.0e0;
  r->d2fdxsts[2] = 0.0e0;
  r->d2fdxsts[3] = 0.0e0;
  r->d2fdxsus[0] = 0.0e0;
  r->d2fdxsus[1] = 0.0e0;
  r->d2fdxsus[2] = 0.0e0;
  r->d2fdxsus[3] = 0.0e0;
  r->d2fdts2[0] = 0.0e0;
  r->d2fdts2[1] = 0.0e0;
  r->d2fdts2[2] = 0.0e0;
  r->d2fdtsus[0] = 0.0e0;
  r->d2fdtsus[1] = 0.0e0;
  r->d2fdtsus[2] = 0.0e0;
  r->d2fdtsus[3] = 0.0e0;
  r->d2fdus2[0] = 0.0e0;
  r->d2fdus2[1] = 0.0e0;
  r->d2fdus2[2] = 0.0e0;

  if(r->order < 3) return;

  t211 = t50 * t50;
  r->d3fdrs3 = 0.52541765884403959618e-1 * t2 / t211 * t47 + 0.10067423881974653480e-1 * t97 * t45 + 0.36534075491463892928e-2 * t52 * t45 + 0.88386641088708730458e-3 * t15 * t45;
  t226 = t97 * t17;
  t234 = t107 * t74;
  r->d3fdrs2z = 0.79868975733333333330e-2 * r->z * t96 * t47 * t103 - 0.10542704796800000000e-2 * t226 * t73 * t3 * t107 + 0.15303536533120000000e-2 * t114 * t3 * t108 - 0.76517682665599999998e-3 * t106 * t234 + 0.55535613235520000000e-3 * t62 * t3 * t108 - 0.27767806617760000000e-3 * t111 * t234;
  t245 = t114 * t17;
  r->d3fdrsz2 = -0.11442546666666666666e-1 * t51 * t47 * t54 - 0.60416646399999999998e-2 * t245 * t120 + 0.15104161600000000000e-2 * t118 * t178 * t4 * t58 - 0.21924837440000000000e-2 * t14 * t4 * t5 * t9 * t17 * t44 - 0.43849674880000000000e-2 * t123 * t126 + 0.10962418720000000000e-2 * t57 * t58 * t179;
  r->d3fdrszxt = 0.75520807999999999996e-3 * t245 * t130 + 0.54812093600000000000e-3 * t123 * t133;
  r->d3fdrszxs[0] = 0.0e0;
  r->d3fdrszxs[1] = 0.0e0;
  r->d3fdrszts[0] = -0.30208323199999999998e-2 * t245 * t137 + 0.20138882133333333333e-2 * t118 * t38 * t4 * t58 - 0.21924837440000000000e-2 * t123 * t140 + 0.14616558293333333333e-2 * t57 * t58 * t186;
  r->d3fdrszts[1] = -0.30208323199999999998e-2 * t245 * t144 - 0.20138882133333333333e-2 * t118 * t41 * t4 * t58 - 0.21924837440000000000e-2 * t123 * t147 - 0.14616558293333333333e-2 * t57 * t58 * t191;
  r->d3fdrszus[0] = -0.30208323199999999998e-2 * t245 * t151 + 0.15104161600000000000e-2 * t118 * t198 * t4 * t58 - 0.21924837440000000000e-2 * t123 * t154 + 0.10962418720000000000e-2 * t57 * t58 * t199;
  r->d3fdrszus[1] = -0.30208323199999999998e-2 * t245 * t158 + 0.15104161600000000000e-2 * t118 * t206 * t4 * t58 - 0.21924837440000000000e-2 * t123 * t161 + 0.10962418720000000000e-2 * t57 * t58 * t207;
  t315 = t107 * t77;
  r->d3fdrs2xt = 0.26356761991999999999e-3 * t226 * r->xt * t3 * t107 + 0.19129420666400000000e-3 * t106 * t315 + 0.69419516544400000000e-4 * t111 * t315;
  r->d3fdrsxt2 = -0.37760403999999999999e-3 * t118 * t54 - 0.27406046800000000000e-3 * t57 * t58 * t17;
  r->d3fdrsxtxs[0] = 0.0e0;
  r->d3fdrsxtxs[1] = 0.0e0;
  r->d3fdrsxtts[0] = 0.0e0;
  r->d3fdrsxtts[1] = 0.0e0;
  r->d3fdrsxtus[0] = 0.0e0;
  r->d3fdrsxtus[1] = 0.0e0;
  r->d3fdrs2xs[0] = 0.0e0;
  r->d3fdrs2xs[1] = 0.0e0;
  r->d3fdrsxs2[0] = 0.0e0;
  r->d3fdrsxs2[1] = 0.0e0;
  r->d3fdrsxs2[2] = 0.0e0;
  r->d3fdrsxsts[0] = 0.0e0;
  r->d3fdrsxsts[1] = 0.0e0;
  r->d3fdrsxsts[2] = 0.0e0;
  r->d3fdrsxsts[3] = 0.0e0;
  r->d3fdrsxsus[0] = 0.0e0;
  r->d3fdrsxsus[1] = 0.0e0;
  r->d3fdrsxsus[2] = 0.0e0;
  r->d3fdrsxsus[3] = 0.0e0;
  t329 = t107 * t79;
  r->d3fdrs2ts[0] = -0.10542704796800000000e-2 * t226 * t23 * t3 * t107 - 0.76517682665599999998e-3 * t106 * t329 - 0.27767806617760000000e-3 * t111 * t329;
  t338 = t107 * t82;
  r->d3fdrs2ts[1] = -0.10542704796800000000e-2 * t226 * t32 * t3 * t107 - 0.76517682665599999998e-3 * t106 * t338 - 0.27767806617760000000e-3 * t111 * t338;
  r->d3fdrsts2[0] = 0.0e0;
  r->d3fdrsts2[1] = 0.0e0;
  r->d3fdrsts2[2] = 0.0e0;
  r->d3fdrstsus[0] = 0.0e0;
  r->d3fdrstsus[1] = 0.0e0;
  r->d3fdrstsus[2] = 0.0e0;
  r->d3fdrstsus[3] = 0.0e0;
  t347 = t107 * t87;
  r->d3fdrs2us[0] = -0.10542704796800000000e-2 * t226 * t86 * t3 * t107 - 0.76517682665599999998e-3 * t106 * t347 - 0.27767806617760000000e-3 * t111 * t347;
  t356 = t107 * t92;
  r->d3fdrs2us[1] = -0.10542704796800000000e-2 * t226 * t91 * t3 * t107 - 0.76517682665599999998e-3 * t106 * t356 - 0.27767806617760000000e-3 * t111 * t356;
  r->d3fdrsus2[0] = 0.0e0;
  r->d3fdrsus2[1] = 0.0e0;
  r->d3fdrsus2[2] = 0.0e0;
  t361 = t14 * t17;
  r->d3fdz3 = 0.7790112e-1 * t361 * t73 + 0.7790112e-1 * t62 * t179 - 0.1298352e-1 * t15 * t17 * (0.10e2 / 0.27e2 * t172 * t25 - 0.10e2 / 0.27e2 * t175 * t34 - 0.5e1 / 0.864e3 * r->us[0] / t21 / t19 + 0.5e1 / 0.864e3 * r->us[1] / t30 / t28);
  r->d3fdz2xt = -0.64917600000000000000e-2 * t361 * r->xt;
  r->d3fdzxt2 = -0.64917600000000000000e-2 * t62 * t17;
  r->d3fdzxtxs[0] = 0.0e0;
  r->d3fdzxtxs[1] = 0.0e0;
  r->d3fdzxtts[0] = 0.0e0;
  r->d3fdzxtts[1] = 0.0e0;
  r->d3fdzxtus[0] = 0.0e0;
  r->d3fdzxtus[1] = 0.0e0;
  r->d3fdz2xs[0] = 0.0e0;
  r->d3fdz2xs[1] = 0.0e0;
  r->d3fdzxs2[0] = 0.0e0;
  r->d3fdzxs2[1] = 0.0e0;
  r->d3fdzxs2[2] = 0.0e0;
  r->d3fdzxsts[0] = 0.0e0;
  r->d3fdzxsts[1] = 0.0e0;
  r->d3fdzxsts[2] = 0.0e0;
  r->d3fdzxsts[3] = 0.0e0;
  r->d3fdzxsus[0] = 0.0e0;
  r->d3fdzxsus[1] = 0.0e0;
  r->d3fdzxsus[2] = 0.0e0;
  r->d3fdzxsus[3] = 0.0e0;
  r->d3fdz2ts[0] = 0.2596704e-1 * t361 * t23 + 0.69245440000000000000e-1 * t62 * t186 - 0.14426133333333333333e-1 * t15 * t17 * t22;
  r->d3fdz2ts[1] = 0.2596704e-1 * t361 * t32 - 0.69245440000000000000e-1 * t62 * t191 - 0.14426133333333333333e-1 * t15 * t17 * t31;
  r->d3fdzts2[0] = 0.0e0;
  r->d3fdzts2[1] = 0.0e0;
  r->d3fdzts2[2] = 0.0e0;
  r->d3fdztsus[0] = 0.0e0;
  r->d3fdztsus[1] = 0.0e0;
  r->d3fdztsus[2] = 0.0e0;
  r->d3fdztsus[3] = 0.0e0;
  r->d3fdz2us[0] = 0.2596704e-1 * t361 * t86 + 0.5193408e-1 * t62 * t199 - 0.1298352e-1 * t15 * t17 * (-0.5e1 / 0.36e2 * t22 + 0.5e1 / 0.144e3 * t172);
  r->d3fdz2us[1] = 0.2596704e-1 * t361 * t91 + 0.5193408e-1 * t62 * t207 - 0.1298352e-1 * t15 * t17 * (-0.5e1 / 0.36e2 * t31 + 0.5e1 / 0.144e3 * t175);
  r->d3fdzus2[0] = 0.0e0;
  r->d3fdzus2[1] = 0.0e0;
  r->d3fdzus2[2] = 0.0e0;
  r->d3fdxt3 = 0.0e0;
  r->d3fdxt2xs[0] = 0.0e0;
  r->d3fdxt2xs[1] = 0.0e0;
  r->d3fdxtxs2[0] = 0.0e0;
  r->d3fdxtxs2[1] = 0.0e0;
  r->d3fdxtxs2[2] = 0.0e0;
  r->d3fdxtxsts[0] = 0.0e0;
  r->d3fdxtxsts[1] = 0.0e0;
  r->d3fdxtxsts[2] = 0.0e0;
  r->d3fdxtxsts[3] = 0.0e0;
  r->d3fdxtxsus[0] = 0.0e0;
  r->d3fdxtxsus[1] = 0.0e0;
  r->d3fdxtxsus[2] = 0.0e0;
  r->d3fdxtxsus[3] = 0.0e0;
  r->d3fdxt2ts[0] = 0.0e0;
  r->d3fdxt2ts[1] = 0.0e0;
  r->d3fdxtts2[0] = 0.0e0;
  r->d3fdxtts2[1] = 0.0e0;
  r->d3fdxtts2[2] = 0.0e0;
  r->d3fdxttsus[0] = 0.0e0;
  r->d3fdxttsus[1] = 0.0e0;
  r->d3fdxttsus[2] = 0.0e0;
  r->d3fdxttsus[3] = 0.0e0;
  r->d3fdxt2us[0] = 0.0e0;
  r->d3fdxt2us[1] = 0.0e0;
  r->d3fdxtus2[0] = 0.0e0;
  r->d3fdxtus2[1] = 0.0e0;
  r->d3fdxtus2[2] = 0.0e0;
  r->d3fdxs3[0] = 0.0e0;
  r->d3fdxs3[1] = 0.0e0;
  r->d3fdxs3[2] = 0.0e0;
  r->d3fdxs3[3] = 0.0e0;
  r->d3fdxs2ts[0] = 0.0e0;
  r->d3fdxs2ts[1] = 0.0e0;
  r->d3fdxs2ts[2] = 0.0e0;
  r->d3fdxs2ts[3] = 0.0e0;
  r->d3fdxs2ts[4] = 0.0e0;
  r->d3fdxs2ts[5] = 0.0e0;
  r->d3fdxs2us[0] = 0.0e0;
  r->d3fdxs2us[1] = 0.0e0;
  r->d3fdxs2us[2] = 0.0e0;
  r->d3fdxs2us[3] = 0.0e0;
  r->d3fdxs2us[4] = 0.0e0;
  r->d3fdxs2us[5] = 0.0e0;
  r->d3fdxsts2[0] = 0.0e0;
  r->d3fdxsts2[1] = 0.0e0;
  r->d3fdxsts2[2] = 0.0e0;
  r->d3fdxsts2[3] = 0.0e0;
  r->d3fdxsts2[4] = 0.0e0;
  r->d3fdxsts2[5] = 0.0e0;
  r->d3fdxstsus[0] = 0.0e0;
  r->d3fdxstsus[1] = 0.0e0;
  r->d3fdxstsus[2] = 0.0e0;
  r->d3fdxstsus[3] = 0.0e0;
  r->d3fdxstsus[4] = 0.0e0;
  r->d3fdxstsus[5] = 0.0e0;
  r->d3fdxstsus[6] = 0.0e0;
  r->d3fdxstsus[7] = 0.0e0;
  r->d3fdxsus2[0] = 0.0e0;
  r->d3fdxsus2[1] = 0.0e0;
  r->d3fdxsus2[2] = 0.0e0;
  r->d3fdxsus2[3] = 0.0e0;
  r->d3fdxsus2[4] = 0.0e0;
  r->d3fdxsus2[5] = 0.0e0;
  r->d3fdts3[0] = 0.0e0;
  r->d3fdts3[1] = 0.0e0;
  r->d3fdts3[2] = 0.0e0;
  r->d3fdts3[3] = 0.0e0;
  r->d3fdts2us[0] = 0.0e0;
  r->d3fdts2us[1] = 0.0e0;
  r->d3fdts2us[2] = 0.0e0;
  r->d3fdts2us[3] = 0.0e0;
  r->d3fdts2us[4] = 0.0e0;
  r->d3fdts2us[5] = 0.0e0;
  r->d3fdtsus2[0] = 0.0e0;
  r->d3fdtsus2[1] = 0.0e0;
  r->d3fdtsus2[2] = 0.0e0;
  r->d3fdtsus2[3] = 0.0e0;
  r->d3fdtsus2[4] = 0.0e0;
  r->d3fdtsus2[5] = 0.0e0;
  r->d3fdus3[0] = 0.0e0;
  r->d3fdus3[1] = 0.0e0;
  r->d3fdus3[2] = 0.0e0;
  r->d3fdus3[3] = 0.0e0;

  if(r->order < 4) return;


}

#define maple2c_order 3
#define maple2c_func  xc_mgga_c_cs_func
