/* 
  This file was generated automatically with /nfs/data-012/marques/software/source/libxc/svn/scripts/maple2c.pl.
  Do not edit this file directly as it can be overwritten!!

  This Source Code Form is subject to the terms of the Mozilla Public
  License, v. 2.0. If a copy of the MPL was not distributed with this
  file, You can obtain one at http://mozilla.org/MPL/2.0/.

  Maple version     : Maple 2016 (X86 64 LINUX)
  Maple source      : ../maple/gga_c_pbe.mpl
  Type of functional: work_gga_c
*/

void xc_gga_c_pbe_func
  (const xc_func_type_cuda *p, xc_gga_work_c_t *r)
{
  double t2, t3, t6, t8, t10, t13, t14, t16;
  double t17, t18, t19, t20, t21, t22, t23, t24;
  double t25, t26, t27, t30, t32, t37, t40, t41;
  double t45, t50, t53, t54, t55, t58, t59, t60;
  double t62, t63, t64, t66, t67, t68, t69, t70;
  double t71, t72, t73, t74, t77, t78, t80, t81;
  double t83, t84, t85, t86, t87, t88, t89, t90;
  double t91, t92, t93, t94, t98, t99, t100, t103;
  double t104, t105, t107, t108, t110, t111, t112, t113;
  double t114, t116, t119, t120, t121, t123, t125, t126;
  double t127, t131, t132, t133, t137, t138, t139, t143;
  double t144, t145, t149, t150, t152, t153, t155, t157;
  double t158, t161, t162, t163, t164, t165, t167, t169;
  double t170, t171, t174, t178, t179, t183, t184, t186;
  double t187, t188, t189, t190, t192, t197, t198, t200;
  double t201, t202, t204, t205, t207, t209, t210, t211;
  double t212, t214, t215, t219, t220, t223, t224, t227;
  double t229, t230, t232, t234, t237, t240, t243, t244;
  double t245, t246, t250, t251, t253, t254, t259, t260;
  double t262, t263, t265, t268, t269, t273, t274, t276;
  double t277, t278, t279, t280, t282, t283, t286, t288;
  double t289, t290, t293, t295, t297, t299, t302, t303;
  double t304, t305, t306, t307, t310, t315, t316, t317;
  double t323, t327, t328, t329, t330, t331, t337, t338;
  double t339, t345, t346, t349, t350, t351, t352, t353;
  double t357, t358, t359, t361, t363, t365, t367, t369;
  double t371, t373, t374, t378, t380, t381, t383, t384;
  double t387, t388, t389, t390, t392, t397, t400, t403;
  double t406, t408, t410, t413, t414, t415, t419, t420;
  double t425, t426, t427, t428, t431, t432, t433, t435;
  double t436, t447, t448, t453, t454, t456, t457, t459;
  double t460, t461, t462, t465, t466, t468, t469, t471;
  double t475, t476, t479, t481, t482, t483, t484, t485;
  double t488, t489, t490, t491, t492, t493, t496, t499;
  double t503, t504, t509, t512, t513, t517, t518, t522;
  double t523, t524, t527, t528, t529, t530, t531, t540;
  double t541, t542, t543, t544, t547, t548, t555, t556;
  double t558, t559, t561, t567, t573, t574, t577, t580;
  double t581, t589, t590, t592, t593, t597, t599, t600;
  double t602, t606, t607, t608, t609, t611, t612, t613;
  double t623, t628, t631, t634, t636, t637, t638, t640;
  double t643, t644, t645, t646, t650, t654, t657, t660;
  double t663, t667, t671, t675, t679, t680, t684, t685;
  double t688, t689, t697, t704, t705, t707, t708, t710;
  double t711, t719, t725, t726, t729, t732, t733, t741;
  double t742, t744, t745, t747, t750, t753, t757, t758;
  double t760, t761, t766, t767, t768, t769, t770, t773;
  double t775, t776, t778, t779, t783, t787, t793, t797;
  double t800, t802, t804, t809, t812, t815, t817, t819;
  double t820, t824, t827, t831, t833, t836, t839, t840;
  double t843, t846, t850, t854, t858, t863, t871, t875;
  double t910, t937, t940, t944, t945, t946, t947, t953;
  double t954, t955, t956, t961, t966, t993, t995, t996;
  double t1020, t1029, t1030, t1031, t1039, t1041, t1043, t1048;
  double t1052, t1053, t1054, t1077, t1088, t1094, t1096, t1097;
  double t1099, t1101, t1103, t1105, t1124, t1125, t1138, t1163;
  double t1198, t1211, t1219, t1223, t1225, t1230, t1234, t1238;
  double t1259, t1265, t1283, t1297, t1301, t1312, t1317, t1319;
  double t1320, t1322, t1325, t1332, t1333, t1356, t1361, t1370;
  double t1378, t1382, t1445, t1446, t1460, t1464, t1478, t1482;
  double t1503, t1509, t1521, t1525, t1530, t1546, t1552, t1556;
  double t1559, t1563, t1572, t1592, t1615, t1637, t1649, t1654;
  double t1658, t1681, t1705, t1723, t1743, t1756, t1762, t1769;
  double t1781, t1794, t1797, t1800, t1806, t1808, t1811, t1812;
  double t1831, t1834, t1838, t1852, t1877, t1888, t1896, t1897;
  double t1902, t1944, t1961, t1965, t1996, t2002, t2042, t2068;
  double t2086, t2114, t2151, t2175, t2181;

  gga_c_pbe_params *params;

  assert(p->params != NULL);
  params = (gga_c_pbe_params * )(p->params);

  t2 = 0.1e1 + 0.21370e0 * r->rs;
  t3 = sqrt(r->rs);
  t6 = POW_3_2(r->rs);
  t8 = r->rs * r->rs;
  t10 = 0.75957e1 * t3 + 0.35876e1 * r->rs + 0.16382e1 * t6 + 0.49294e0 * t8;
  t13 = 0.1e1 + 0.16081979498692535067e2 / t10;
  t14 = log(t13);
  t16 = 0.621814e-1 * t2 * t14;
  t17 = r->z * r->z;
  t18 = t17 * t17;
  t19 = 0.1e1 + r->z;
  t20 = POW_1_3(t19);
  t21 = t20 * t19;
  t22 = 0.1e1 - r->z;
  t23 = POW_1_3(t22);
  t24 = t23 * t22;
  t25 = t21 + t24 - 0.2e1;
  t26 = t18 * t25;
  t27 = POW_1_3(0.2e1);
  t30 = 0.1e1 / (0.2e1 * t27 - 0.2e1);
  t32 = 0.1e1 + 0.20548e0 * r->rs;
  t37 = 0.141189e2 * t3 + 0.61977e1 * r->rs + 0.33662e1 * t6 + 0.62517e0 * t8;
  t40 = 0.1e1 + 0.32163958997385070134e2 / t37;
  t41 = log(t40);
  t45 = 0.1e1 + 0.11125e0 * r->rs;
  t50 = 0.10357e2 * t3 + 0.36231e1 * r->rs + 0.88026e0 * t6 + 0.49671e0 * t8;
  t53 = 0.1e1 + 0.29608749977793437516e2 / t50;
  t54 = log(t53);
  t55 = t45 * t54;
  t58 = t30 * (-0.3109070e-1 * t32 * t41 + t16 - 0.19751673498613801407e-1 * t55);
  t59 = t26 * t58;
  t60 = t25 * t30;
  t62 = 0.19751673498613801407e-1 * t60 * t55;
  t63 = t20 * t20;
  t64 = t23 * t23;
  t66 = t63 / 0.2e1 + t64 / 0.2e1;
  t67 = t66 * t66;
  t68 = t67 * t66;
  t69 = params->gamma * t68;
  t70 = r->xt * r->xt;
  t71 = t70 * t27;
  t72 = 0.1e1 / t67;
  t73 = 0.1e1 / r->rs;
  t74 = t72 * t73;
  t77 = params->BB * params->beta;
  t78 = 0.1e1 / params->gamma;
  t80 = (-t16 + t59 + t62) * t78;
  t81 = 0.1e1 / t68;
  t83 = exp(-t80 * t81);
  t84 = t83 - 0.1e1;
  t85 = 0.1e1 / t84;
  t86 = t78 * t85;
  t87 = t77 * t86;
  t88 = t70 * t70;
  t89 = t27 * t27;
  t90 = t88 * t89;
  t91 = t67 * t67;
  t92 = 0.1e1 / t91;
  t93 = 0.1e1 / t8;
  t94 = t92 * t93;
  t98 = t71 * t74 / 0.32e2 + t87 * t90 * t94 / 0.1024e4;
  t99 = params->beta * t98;
  t100 = params->beta * t78;
  t103 = t100 * t85 * t98 + 0.1e1;
  t104 = 0.1e1 / t103;
  t105 = t78 * t104;
  t107 = t99 * t105 + 0.1e1;
  t108 = log(t107);
  r->f = t69 * t108 - t16 + t59 + t62;

  if(r->order < 1) return;

  t110 = 0.13288165180e-1 * t14;
  t111 = t10 * t10;
  t112 = 0.1e1 / t111;
  t113 = t2 * t112;
  t114 = 0.1e1 / t3;
  t116 = sqrt(r->rs);
  t119 = 0.37978500000000000000e1 * t114 + 0.35876e1 + 0.245730e1 * t116 + 0.98588e0 * r->rs;
  t120 = 0.1e1 / t13;
  t121 = t119 * t120;
  t123 = 0.10000000000000000000e1 * t113 * t121;
  t125 = t37 * t37;
  t126 = 0.1e1 / t125;
  t127 = t32 * t126;
  t131 = 0.70594500000000000000e1 * t114 + 0.61977e1 + 0.504930e1 * t116 + 0.125034e1 * r->rs;
  t132 = 0.1e1 / t40;
  t133 = t131 * t132;
  t137 = t50 * t50;
  t138 = 0.1e1 / t137;
  t139 = t45 * t138;
  t143 = 0.51785000000000000000e1 * t114 + 0.36231e1 + 0.1320390e1 * t116 + 0.99342e0 * r->rs;
  t144 = 0.1e1 / t53;
  t145 = t143 * t144;
  t149 = t30 * (-0.63885170360e-2 * t41 + 0.10000000000000000000e1 * t127 * t133 + t110 - t123 - 0.21973736767207854065e-2 * t54 + 0.58482236226346462070e0 * t139 * t145);
  t150 = t26 * t149;
  t152 = 0.21973736767207854065e-2 * t60 * t54;
  t153 = t60 * t45;
  t155 = t138 * t143 * t144;
  t157 = 0.58482236226346462070e0 * t153 * t155;
  t158 = t72 * t93;
  t161 = params->gamma * params->gamma;
  t162 = 0.1e1 / t161;
  t163 = t84 * t84;
  t164 = 0.1e1 / t163;
  t165 = t162 * t164;
  t167 = t77 * t165 * t88;
  t169 = 0.1e1 / t91 / t68;
  t170 = t89 * t169;
  t171 = -t110 + t123 + t150 + t152 - t157;
  t174 = t170 * t93 * t171 * t83;
  t178 = 0.1e1 / t8 / r->rs;
  t179 = t92 * t178;
  t183 = -t71 * t158 / 0.32e2 + t167 * t174 / 0.1024e4 - t87 * t90 * t179 / 0.512e3;
  t184 = params->beta * t183;
  t186 = t103 * t103;
  t187 = 0.1e1 / t186;
  t188 = t78 * t187;
  t189 = params->beta * t162;
  t190 = t189 * t164;
  t192 = t81 * t83;
  t197 = t190 * t98 * t171 * t192 + t100 * t85 * t183;
  t198 = t188 * t197;
  t200 = t184 * t105 - t99 * t198;
  t201 = 0.1e1 / t107;
  t202 = t200 * t201;
  r->dfdrs = t69 * t202 - t110 + t123 + t150 + t152 - t157;
  t204 = t17 * r->z;
  t205 = t204 * t25;
  t207 = 0.4e1 * t205 * t58;
  t209 = 0.4e1 / 0.3e1 * t20 - 0.4e1 / 0.3e1 * t23;
  t210 = t18 * t209;
  t211 = t210 * t58;
  t212 = t209 * t30;
  t214 = 0.19751673498613801407e-1 * t212 * t55;
  t215 = params->gamma * t67;
  t219 = 0.1e1 / t20 / 0.3e1 - 0.1e1 / t23 / 0.3e1;
  t220 = t108 * t219;
  t223 = t81 * t73;
  t224 = t223 * t219;
  t227 = t78 * t164;
  t229 = t77 * t227 * t88;
  t230 = t89 * t92;
  t232 = (t207 + t211 + t214) * t78;
  t234 = t92 * t219;
  t237 = -t232 * t81 + 0.3e1 * t80 * t234;
  t240 = t230 * t93 * t237 * t83;
  t243 = t91 * t66;
  t244 = 0.1e1 / t243;
  t245 = t244 * t93;
  t246 = t245 * t219;
  t250 = -t71 * t224 / 0.16e2 - t229 * t240 / 0.1024e4 - t87 * t90 * t246 / 0.256e3;
  t251 = params->beta * t250;
  t253 = t100 * t164;
  t254 = t98 * t237;
  t259 = t100 * t85 * t250 - t253 * t254 * t83;
  t260 = t188 * t259;
  t262 = t251 * t105 - t99 * t260;
  t263 = t262 * t201;
  r->dfdz = 0.3e1 * t215 * t220 + t69 * t263 + t207 + t211 + t214;
  t265 = r->xt * t27;
  t268 = t70 * r->xt;
  t269 = t268 * t89;
  t273 = t265 * t74 / 0.16e2 + t87 * t269 * t94 / 0.256e3;
  t274 = params->beta * t273;
  t276 = params->beta * params->beta;
  t277 = t276 * t98;
  t278 = t277 * t162;
  t279 = t187 * t85;
  t280 = t279 * t273;
  t282 = t274 * t105 - t278 * t280;
  t283 = t282 * t201;
  r->dfdxt = t69 * t283;
  r->dfdxs[0] = 0.0e0;
  r->dfdxs[1] = 0.0e0;

  if(r->order < 2) return;

  t286 = 0.42740000000000000000e0 * t112 * t119 * t120;
  t288 = 0.1e1 / t111 / t10;
  t289 = t2 * t288;
  t290 = t119 * t119;
  t293 = 0.20000000000000000000e1 * t289 * t290 * t120;
  t295 = 0.1e1 / t3 / r->rs;
  t297 = 0.1e1/sqrt(r->rs);
  t299 = -0.18989250000000000000e1 * t295 + 0.1228650e1 * t297 + 0.98588e0;
  t302 = 0.10000000000000000000e1 * t113 * t299 * t120;
  t303 = t111 * t111;
  t304 = 0.1e1 / t303;
  t305 = t2 * t304;
  t306 = t13 * t13;
  t307 = 0.1e1 / t306;
  t310 = 0.16081979498692535067e2 * t305 * t290 * t307;
  t315 = 0.1e1 / t125 / t37;
  t316 = t32 * t315;
  t317 = t131 * t131;
  t323 = -0.35297250000000000000e1 * t295 + 0.2524650e1 * t297 + 0.125034e1;
  t327 = t125 * t125;
  t328 = 0.1e1 / t327;
  t329 = t32 * t328;
  t330 = t40 * t40;
  t331 = 0.1e1 / t330;
  t337 = 0.1e1 / t137 / t50;
  t338 = t45 * t337;
  t339 = t143 * t143;
  t345 = -0.25892500000000000000e1 * t295 + 0.6601950e0 * t297 + 0.99342e0;
  t346 = t345 * t144;
  t349 = t137 * t137;
  t350 = 0.1e1 / t349;
  t351 = t45 * t350;
  t352 = t53 * t53;
  t353 = 0.1e1 / t352;
  t357 = 0.41096000000000000000e0 * t126 * t131 * t132 - 0.20000000000000000000e1 * t316 * t317 * t132 + 0.10000000000000000000e1 * t127 * t323 * t132 + 0.32163958997385070134e2 * t329 * t317 * t331 - t286 + t293 - t302 - t310 + 0.13012297560362087810e0 * t155 - 0.11696447245269292414e1 * t338 * t339 * t144 + 0.58482236226346462070e0 * t139 * t346 + 0.17315859105681463759e2 * t351 * t339 * t353;
  t358 = t30 * t357;
  t359 = t26 * t358;
  t361 = 0.13012297560362087810e0 * t60 * t155;
  t363 = t337 * t339 * t144;
  t365 = 0.11696447245269292414e1 * t153 * t363;
  t367 = t138 * t345 * t144;
  t369 = 0.58482236226346462070e0 * t153 * t367;
  t371 = t350 * t339 * t353;
  t373 = 0.17315859105681463759e2 * t153 * t371;
  t374 = t72 * t178;
  t378 = 0.1e1 / t161 / params->gamma;
  t380 = 0.1e1 / t163 / t84;
  t381 = t378 * t380;
  t383 = t77 * t381 * t88;
  t384 = t91 * t91;
  t387 = t89 / t384 / t67;
  t388 = t171 * t171;
  t389 = t93 * t388;
  t390 = t83 * t83;
  t392 = t387 * t389 * t390;
  t397 = t170 * t178 * t171 * t83;
  t400 = t286 - t293 + t302 + t310 + t359 - t361 + t365 - t369 - t373;
  t403 = t170 * t93 * t400 * t83;
  t406 = t378 * t164;
  t408 = t77 * t406 * t88;
  t410 = t387 * t389 * t83;
  t413 = t8 * t8;
  t414 = 0.1e1 / t413;
  t415 = t92 * t414;
  t419 = t71 * t374 / 0.16e2 + t383 * t392 / 0.512e3 - t167 * t397 / 0.256e3 + t167 * t403 / 0.1024e4 - t408 * t410 / 0.1024e4 + 0.3e1 / 0.512e3 * t87 * t90 * t415;
  t420 = params->beta * t419;
  t425 = 0.1e1 / t186 / t103;
  t426 = t78 * t425;
  t427 = t197 * t197;
  t428 = t426 * t427;
  t431 = params->beta * t378;
  t432 = t431 * t380;
  t433 = t98 * t388;
  t435 = 0.1e1 / t91 / t67;
  t436 = t435 * t390;
  t447 = t431 * t164;
  t448 = t435 * t83;
  t453 = 0.2e1 * t190 * t183 * t171 * t192 + t190 * t98 * t400 * t192 + t100 * t85 * t419 + 0.2e1 * t432 * t433 * t436 - t447 * t433 * t448;
  t454 = t188 * t453;
  t456 = t420 * t105 - 0.2e1 * t184 * t198 + 0.2e1 * t99 * t428 - t99 * t454;
  t457 = t456 * t201;
  t459 = t200 * t200;
  t460 = t107 * t107;
  t461 = 0.1e1 / t460;
  t462 = t459 * t461;
  r->d2fdrs2 = t69 * t457 - t69 * t462 + t286 - t293 + t302 + t310 + t359 - t361 + t365 - t369 - t373;
  t465 = 0.4e1 * t205 * t149;
  t466 = t210 * t149;
  t468 = 0.21973736767207854065e-2 * t212 * t54;
  t469 = t212 * t45;
  t471 = 0.58482236226346462070e0 * t469 * t155;
  t475 = t81 * t93;
  t476 = t475 * t219;
  t479 = t162 * t380;
  t481 = t77 * t479 * t88;
  t482 = t170 * t93;
  t483 = t171 * t390;
  t484 = t483 * t237;
  t485 = t482 * t484;
  t488 = 0.1e1 / t384;
  t489 = t89 * t488;
  t490 = t489 * t93;
  t491 = t171 * t83;
  t492 = t491 * t219;
  t493 = t490 * t492;
  t496 = t465 + t466 + t468 - t471;
  t499 = t170 * t93 * t496 * t83;
  t503 = t171 * t237 * t83;
  t504 = t482 * t503;
  t509 = t230 * t178 * t237 * t83;
  t512 = t244 * t178;
  t513 = t512 * t219;
  t517 = t71 * t476 / 0.16e2 - t481 * t485 / 0.512e3 - 0.7e1 / 0.1024e4 * t167 * t493 + t167 * t499 / 0.1024e4 + t167 * t504 / 0.1024e4 + t229 * t509 / 0.512e3 + t87 * t90 * t513 / 0.128e3;
  t518 = params->beta * t517;
  t522 = t99 * t78;
  t523 = t425 * t197;
  t524 = t523 * t259;
  t527 = t380 * t98;
  t528 = t189 * t527;
  t529 = t171 * t81;
  t530 = t390 * t237;
  t531 = t529 * t530;
  t540 = t164 * t98;
  t541 = t189 * t540;
  t542 = t171 * t92;
  t543 = t83 * t219;
  t544 = t542 * t543;
  t547 = t237 * t83;
  t548 = t529 * t547;
  t555 = t190 * t250 * t171 * t192 - t253 * t183 * t237 * t83 + t190 * t98 * t496 * t192 + t100 * t85 * t517 - 0.2e1 * t528 * t531 - 0.3e1 * t541 * t544 + t541 * t548;
  t556 = t188 * t555;
  t558 = t518 * t105 - t184 * t260 - t251 * t198 + 0.2e1 * t522 * t524 - t99 * t556;
  t559 = t558 * t201;
  t561 = t200 * t461;
  r->d2fdrsz = 0.3e1 * t215 * t202 * t219 - t69 * t561 * t262 + t69 * t559 + t465 + t466 + t468 - t471;
  t567 = t77 * t165 * t268;
  t573 = -t265 * t158 / 0.16e2 + t567 * t174 / 0.256e3 - t87 * t269 * t179 / 0.128e3;
  t574 = params->beta * t573;
  t577 = t276 * t183 * t162;
  t580 = t85 * t273;
  t581 = t523 * t580;
  t589 = t190 * t273 * t171 * t192 + t100 * t85 * t573;
  t590 = t188 * t589;
  t592 = t574 * t105 - t274 * t198 + 0.2e1 * t278 * t581 - t577 * t280 - t99 * t590;
  t593 = t592 * t201;
  r->d2fdrsxt = -t69 * t561 * t282 + t69 * t593;
  r->d2fdrsxs[0] = 0.0e0;
  r->d2fdrsxs[1] = 0.0e0;
  t597 = t17 * t25;
  t599 = 0.12e2 * t597 * t58;
  t600 = t204 * t209;
  t602 = 0.8e1 * t600 * t58;
  t606 = 0.4e1 / 0.9e1 / t63 + 0.4e1 / 0.9e1 / t64;
  t607 = t18 * t606;
  t608 = t607 * t58;
  t609 = t606 * t30;
  t611 = 0.19751673498613801407e-1 * t609 * t55;
  t612 = params->gamma * t66;
  t613 = t219 * t219;
  t623 = -0.1e1 / t21 / 0.9e1 - 0.1e1 / t24 / 0.9e1;
  t628 = t92 * t73 * t613;
  t631 = t223 * t623;
  t634 = t78 * t380;
  t636 = t77 * t634 * t88;
  t637 = t237 * t237;
  t638 = t93 * t637;
  t640 = t230 * t638 * t390;
  t643 = t89 * t244;
  t644 = t643 * t93;
  t645 = t547 * t219;
  t646 = t644 * t645;
  t650 = (t599 + t602 + t608 + t611) * t78;
  t654 = t244 * t613;
  t657 = t92 * t623;
  t660 = 0.6e1 * t232 * t234 - t650 * t81 - 0.12e2 * t80 * t654 + 0.3e1 * t80 * t657;
  t663 = t230 * t93 * t660 * t83;
  t667 = t230 * t638 * t83;
  t671 = t435 * t93 * t613;
  t675 = t245 * t623;
  t679 = 0.3e1 / 0.16e2 * t71 * t628 - t71 * t631 / 0.16e2 + t636 * t640 / 0.512e3 + t229 * t646 / 0.128e3 - t229 * t663 / 0.1024e4 - t229 * t667 / 0.1024e4 + 0.5e1 / 0.256e3 * t87 * t90 * t671 - t87 * t90 * t675 / 0.256e3;
  t680 = params->beta * t679;
  t684 = t259 * t259;
  t685 = t426 * t684;
  t688 = t100 * t380;
  t689 = t98 * t637;
  t697 = t98 * t660;
  t704 = -0.2e1 * t253 * t250 * t237 * t83 + t100 * t85 * t679 - t253 * t689 * t83 - t253 * t697 * t83 + 0.2e1 * t688 * t689 * t390;
  t705 = t188 * t704;
  t707 = t680 * t105 - 0.2e1 * t251 * t260 + 0.2e1 * t99 * t685 - t99 * t705;
  t708 = t707 * t201;
  t710 = t262 * t262;
  t711 = t710 * t461;
  r->d2fdz2 = 0.3e1 * t215 * t108 * t623 + 0.6e1 * t612 * t108 * t613 + 0.6e1 * t215 * t263 * t219 + t69 * t708 - t69 * t711 + t599 + t602 + t608 + t611;
  t719 = t77 * t227 * t268;
  t725 = -t265 * t224 / 0.8e1 - t719 * t240 / 0.256e3 - t87 * t269 * t246 / 0.64e2;
  t726 = params->beta * t725;
  t729 = t276 * t250 * t162;
  t732 = t425 * t259;
  t733 = t732 * t580;
  t741 = -t253 * t273 * t237 * t83 + t100 * t85 * t725;
  t742 = t188 * t741;
  t744 = t726 * t105 - t274 * t260 + 0.2e1 * t278 * t733 - t729 * t280 - t99 * t742;
  t745 = t744 * t201;
  t747 = t262 * t461;
  r->d2fdzxt = 0.3e1 * t215 * t283 * t219 - t69 * t747 * t282 + t69 * t745;
  r->d2fdzxs[0] = 0.0e0;
  r->d2fdzxs[1] = 0.0e0;
  t750 = t27 * t72;
  t753 = t70 * t89;
  t757 = t750 * t73 / 0.16e2 + 0.3e1 / 0.256e3 * t87 * t753 * t94;
  t758 = params->beta * t757;
  t760 = t273 * t273;
  t761 = t276 * t760;
  t766 = t276 * params->beta;
  t767 = t766 * t98;
  t768 = t767 * t378;
  t769 = t425 * t164;
  t770 = t769 * t760;
  t773 = t279 * t757;
  t775 = -0.2e1 * t761 * t162 * t187 * t85 + t758 * t105 - t278 * t773 + 0.2e1 * t768 * t770;
  t776 = t775 * t201;
  t778 = t282 * t282;
  t779 = t778 * t461;
  r->d2fdxt2 = t69 * t776 - t69 * t779;
  r->d2fdxtxs[0] = 0.0e0;
  r->d2fdxtxs[1] = 0.0e0;
  r->d2fdxs2[0] = 0.0e0;
  r->d2fdxs2[1] = 0.0e0;
  r->d2fdxs2[2] = 0.0e0;

  if(r->order < 3) return;

  t783 = 0.1e1 / t460 / t107;
  t787 = t456 * t461;
  t793 = 0.12822000000000000000e1 * t288 * t290 * t120;
  t797 = t290 * t119;
  t800 = 0.96491876992155210402e2 * t2 / t303 / t10 * t797 * t307;
  t802 = 0.1e1 / t3 / t8;
  t804 = 0.1e1/POW_3_2(r->rs);
  t809 = 0.10000000000000000000e1 * t113 * (0.28483875000000000000e1 * t802 - 0.6143250e0 * t804) * t120;
  t812 = 0.64110000000000000000e0 * t112 * t299 * t120;
  t815 = 0.10310157056611784231e2 * t304 * t290 * t307;
  t817 = 0.39036892681086263431e0 * t60 * t363;
  t819 = 0.1e1 / t349 / t50;
  t820 = t339 * t143;
  t824 = 0.10389515463408878255e3 * t153 * t819 * t820 * t353;
  t827 = 0.38838750000000000000e1 * t802 - 0.33009750e0 * t804;
  t831 = 0.58482236226346462070e0 * t153 * t138 * t827 * t144;
  t833 = 0.1e1 / t349 / t137;
  t836 = 0.1e1 / t352 / t53;
  t839 = 0.10254018858216406658e4 * t153 * t833 * t820 * t836;
  t840 = 0.2e1 * t69 * t459 * t200 * t783 - 0.3e1 * t69 * t787 * t200 - t793 - t800 + t809 + t812 + t815 + t817 + t824 - t831 - t839;
  t843 = 0.60000000000000000000e1 * t305 * t797 * t120;
  t846 = 0.60000000000000000000e1 * t289 * t121 * t299;
  t850 = 0.48245938496077605201e2 * t305 * t299 * t307 * t119;
  t854 = 0.35089341735807877242e1 * t153 * t350 * t820 * t144;
  t858 = 0.35089341735807877242e1 * t153 * t337 * t143 * t346;
  t863 = 0.51947577317044391277e2 * t153 * t350 * t345 * t353 * t143;
  t871 = 0.51726012919273400301e3 * t2 / t303 / t111 * t797 / t306 / t13;
  t875 = t317 * t131;
  t910 = -0.19298375398431042081e3 * t32 / t327 / t37 * t875 * t331 + 0.10000000000000000000e1 * t127 * (0.52945875000000000000e1 * t802 - 0.12623250e1 * t804) * t132 + 0.20690405167709360120e4 * t32 / t327 / t125 * t875 / t330 / t40 - 0.10389515463408878255e3 * t45 * t819 * t820 * t353 + 0.58482236226346462070e0 * t139 * t827 * t144 + 0.10254018858216406658e4 * t45 * t833 * t820 * t836 + 0.35089341735807877242e1 * t351 * t820 * t144 - t871 + 0.60000000000000000000e1 * t329 * t875 * t132 + t800 - t809 - t843 + t793;
  t937 = -t812 - t815 + 0.61644000000000000000e0 * t126 * t323 * t132 + 0.19827150884348052633e2 * t328 * t317 * t331 + 0.19518446340543131715e0 * t367 + 0.57791679765211885293e1 * t371 - 0.12328800000000000000e1 * t315 * t317 * t132 - 0.39036892681086263431e0 * t363 + t846 - t850 - 0.60000000000000000000e1 * t316 * t133 * t323 + 0.96491876992155210402e2 * t329 * t323 * t331 * t131 - 0.35089341735807877242e1 * t338 * t145 * t345 + 0.51947577317044391277e2 * t351 * t345 * t353 * t143;
  t940 = t26 * t30 * (t910 + t937);
  t944 = t161 * t161;
  t945 = 0.1e1 / t944;
  t946 = t163 * t163;
  t947 = 0.1e1 / t946;
  t953 = t89 / t384 / t243;
  t954 = t388 * t171;
  t955 = t93 * t954;
  t956 = t390 * t83;
  t961 = t178 * t388;
  t966 = t387 * t93;
  t993 = 0.19518446340543131715e0 * t60 * t367;
  t995 = 0.57791679765211885293e1 * t60 * t371;
  t996 = -t793 - t800 + t809 + t812 + t815 + t817 + t824 - t831 - t839 + t843 - t846 + t850 - t854 + t858 - t863 + t871 + t940 - t993 - t995;
  t1020 = -0.3e1 / 0.16e2 * t71 * t72 * t414 + 0.3e1 / 0.512e3 * t77 * t945 * t947 * t88 * t953 * t955 * t956 - 0.3e1 / 0.256e3 * t383 * t387 * t961 * t390 + 0.3e1 / 0.512e3 * t383 * t966 * t483 * t400 - 0.3e1 / 0.512e3 * t77 * t945 * t380 * t88 * t953 * t955 * t390 + 0.9e1 / 0.512e3 * t167 * t170 * t414 * t171 * t83 - 0.3e1 / 0.512e3 * t167 * t170 * t178 * t400 * t83 + 0.3e1 / 0.512e3 * t408 * t387 * t961 * t83 + t167 * t170 * t93 * t996 * t83 / 0.1024e4 - 0.3e1 / 0.1024e4 * t408 * t966 * t400 * t171 * t83 + t77 * t945 * t164 * t88 * t953 * t955 * t83 / 0.1024e4 - 0.3e1 / 0.128e3 * t87 * t90 * t92 / t413 / r->rs;
  t1029 = t186 * t186;
  t1030 = 0.1e1 / t1029;
  t1031 = t78 * t1030;
  t1039 = params->beta * t945;
  t1041 = t98 * t954;
  t1043 = 0.1e1 / t384 / t66;
  t1048 = t183 * t388;
  t1052 = t431 * t527;
  t1053 = t171 * t435;
  t1054 = t390 * t400;
  t1077 = t431 * t540;
  t1088 = 0.6e1 * t1039 * t947 * t1041 * t1043 * t956 + 0.6e1 * t432 * t1048 * t436 + 0.6e1 * t1052 * t1053 * t1054 - 0.6e1 * t1039 * t380 * t1041 * t1043 * t390 + 0.3e1 * t190 * t419 * t171 * t192 + 0.3e1 * t190 * t183 * t400 * t192 - 0.3e1 * t447 * t1048 * t448 + t190 * t98 * t996 * t192 - 0.3e1 * t1077 * t400 * t435 * t491 + t1039 * t164 * t1041 * t1043 * t83 + t100 * t85 * t1020;
  t1094 = t843 - t846 + t850 - t854 + t858 - t863 + t871 + t940 + t69 * (-0.6e1 * t99 * t1031 * t427 * t197 + params->beta * t1020 * t105 - t99 * t188 * t1088 + 0.6e1 * t522 * t523 * t453 + 0.6e1 * t184 * t428 - 0.3e1 * t184 * t454 - 0.3e1 * t420 * t198) * t201 - t993 - t995;
  r->d3fdrs3 = t840 + t1094;
  t1096 = 0.4e1 * t205 * t358;
  t1097 = t210 * t358;
  t1099 = 0.13012297560362087810e0 * t212 * t155;
  t1101 = 0.11696447245269292414e1 * t469 * t363;
  t1103 = 0.58482236226346462070e0 * t469 * t367;
  t1105 = 0.17315859105681463759e2 * t469 * t371;
  t1124 = t89 / t384 / t68 * t93;
  t1125 = t388 * t390;
  t1138 = t170 * t178;
  t1163 = t1096 + t1097 - t1099 + t1101 - t1103 - t1105;
  t1198 = -t71 * t81 * t178 * t219 / 0.8e1 - 0.3e1 / 0.512e3 * t77 * t378 * t947 * t88 * t966 * t388 * t956 * t237 - 0.5e1 / 0.256e3 * t383 * t1124 * t1125 * t219 + t383 * t966 * t483 * t496 / 0.256e3 + 0.3e1 / 0.512e3 * t383 * t966 * t1125 * t237 + t481 * t1138 * t484 / 0.128e3 + 0.7e1 / 0.256e3 * t167 * t489 * t178 * t492 - t167 * t170 * t178 * t496 * t83 / 0.256e3 - t167 * t1138 * t503 / 0.256e3 - t481 * t482 * t1054 * t237 / 0.512e3 - 0.7e1 / 0.1024e4 * t167 * t490 * t400 * t83 * t219 + t167 * t170 * t93 * t1163 * t83 / 0.1024e4 + t167 * t482 * t400 * t237 * t83 / 0.1024e4 + 0.5e1 / 0.512e3 * t408 * t1124 * t388 * t83 * t219 - t408 * t966 * t491 * t496 / 0.512e3 - t408 * t966 * t388 * t237 * t83 / 0.1024e4 - 0.3e1 / 0.512e3 * t229 * t230 * t414 * t237 * t83 - 0.3e1 / 0.128e3 * t87 * t90 * t244 * t414 * t219;
  t1211 = t1030 * t427;
  t1219 = t425 * t453;
  t1223 = t947 * t98;
  t1225 = t388 * t435;
  t1230 = t250 * t388;
  t1234 = t390 * t496;
  t1238 = t388 * t169;
  t1259 = t189 * t164 * t183;
  t1265 = t400 * t81;
  t1283 = t83 * t496;
  t1297 = t190 * t98 * t1163 * t192 + t190 * t250 * t400 * t192 - t253 * t419 * t237 * t83 - 0.3e1 * t541 * t400 * t92 * t543 + t100 * t85 * t1198 - 0.2e1 * t1077 * t1053 * t1283 - t1077 * t1225 * t547 + 0.6e1 * t1077 * t1238 * t543 - t447 * t1230 * t448 - 0.2e1 * t528 * t1265 * t530 + t541 * t1265 * t547;
  t1301 = params->beta * t1198 * t105 - t420 * t260 - 0.2e1 * t518 * t198 + 0.4e1 * t184 * t78 * t524 - 0.2e1 * t184 * t556 + 0.2e1 * t251 * t428 - 0.6e1 * t522 * t1211 * t259 + 0.4e1 * t522 * t523 * t555 - t251 * t454 + 0.2e1 * t522 * t1219 * t259 - t99 * t188 * (-0.6e1 * t431 * t1223 * t1225 * t956 * t237 - 0.12e2 * t1052 * t1238 * t390 * t219 + 0.2e1 * t190 * t517 * t171 * t192 - 0.4e1 * t189 * t380 * t183 * t531 + 0.2e1 * t190 * t183 * t496 * t192 + 0.4e1 * t1052 * t1053 * t1234 + 0.6e1 * t1052 * t1225 * t530 + 0.2e1 * t432 * t1230 * t436 - 0.6e1 * t1259 * t544 + 0.2e1 * t1259 * t548 + t1297);
  t1312 = t459 * t783;
  r->d3fdrs2z = t69 * t1301 * t201 + 0.2e1 * t69 * t1312 * t262 + 0.3e1 * t215 * t457 * t219 - 0.3e1 * t215 * t462 * t219 - t69 * t787 * t262 - 0.2e1 * t69 * t561 * t558 + t1096 + t1097 - t1099 + t1101 - t1103 - t1105;
  t1317 = 0.12e2 * t597 * t149;
  t1319 = 0.8e1 * t600 * t149;
  t1320 = t607 * t149;
  t1322 = 0.21973736767207854065e-2 * t609 * t54;
  t1325 = 0.58482236226346462070e0 * t609 * t45 * t155;
  t1332 = t215 * t200;
  t1333 = t461 * t219;
  t1356 = t77 * t162;
  t1361 = t488 * t93 * t171;
  t1370 = t178 * t637;
  t1378 = t530 * t219;
  t1382 = t1317 + t1319 + t1320 + t1322 - t1325;
  t1445 = -0.5e1 / 0.128e3 * t87 * t90 * t435 * t178 * t613 - t481 * t482 * t1234 * t237 / 0.256e3 - t481 * t482 * t483 * t660 / 0.512e3 - 0.7e1 / 0.512e3 * t167 * t490 * t1283 * t219 - 0.7e1 / 0.1024e4 * t167 * t490 * t491 * t623 + t167 * t482 * t496 * t237 * t83 / 0.512e3 + t167 * t482 * t171 * t660 * t83 / 0.1024e4 - t229 * t643 * t178 * t645 / 0.64e2 + t229 * t230 * t1370 * t83 / 0.512e3 + 0.3e1 / 0.512e3 * t77 * t162 * t947 * t88 * t482 * t171 * t956 * t637 - 0.3e1 / 0.512e3 * t481 * t482 * t483 * t637;
  t1446 = t71 * t475 * t623 / 0.16e2 + t87 * t90 * t512 * t623 / 0.128e3 - 0.3e1 / 0.16e2 * t71 * t94 * t613 + 0.7e1 / 0.128e3 * t167 * t89 * t1043 * t93 * t491 * t613 - 0.7e1 / 0.512e3 * t1356 * t164 * t88 * t89 * t1361 * t645 + t167 * t482 * t171 * t637 * t83 / 0.1024e4 - t636 * t230 * t1370 * t390 / 0.256e3 + 0.7e1 / 0.256e3 * t1356 * t380 * t88 * t89 * t1361 * t1378 + t167 * t170 * t93 * t1382 * t83 / 0.1024e4 + t229 * t230 * t178 * t660 * t83 / 0.512e3 + t1445;
  t1460 = t1030 * t197;
  t1464 = t425 * t555;
  t1478 = t390 * t637;
  t1482 = t183 * t637;
  t1503 = t637 * t83;
  t1509 = 0.6e1 * t189 * t1223 * t529 * t956 * t637 + 0.12e2 * t541 * t171 * t244 * t83 * t613 + t190 * t98 * t1382 * t192 + t190 * t679 * t171 * t192 + 0.2e1 * t190 * t250 * t496 * t192 + t100 * t85 * t1446 + 0.12e2 * t528 * t542 * t1378 - 0.6e1 * t528 * t529 * t1478 - t253 * t1482 * t83 + t541 * t529 * t1503 - 0.6e1 * t541 * t542 * t645;
  t1521 = t496 * t81;
  t1525 = t390 * t660;
  t1530 = t189 * t164 * t250;
  t1546 = t660 * t83;
  t1552 = -t253 * t183 * t660 * t83 - 0.4e1 * t189 * t380 * t250 * t531 - 0.2e1 * t253 * t517 * t237 * t83 - 0.6e1 * t541 * t496 * t92 * t543 - 0.3e1 * t541 * t542 * t83 * t623 + 0.2e1 * t688 * t1482 * t390 - 0.4e1 * t528 * t1521 * t530 + 0.2e1 * t541 * t1521 * t547 - 0.2e1 * t528 * t529 * t1525 + t541 * t529 * t1546 - 0.6e1 * t1530 * t544 + 0.2e1 * t1530 * t548;
  t1556 = params->beta * t1446 * t105 - 0.2e1 * t518 * t260 + 0.2e1 * t184 * t685 - t184 * t705 - t680 * t198 + 0.4e1 * t251 * t78 * t524 - 0.2e1 * t251 * t556 - 0.6e1 * t522 * t1460 * t684 + 0.4e1 * t522 * t1464 * t259 + 0.2e1 * t522 * t523 * t704 - t99 * t188 * (t1509 + t1552);
  t1559 = t558 * t461;
  t1563 = t200 * t783;
  r->d3fdrsz2 = -0.6e1 * t1332 * t1333 * t262 + t69 * t1556 * t201 - 0.2e1 * t69 * t1559 * t262 + 0.2e1 * t69 * t1563 * t710 + 0.3e1 * t215 * t202 * t623 + 0.6e1 * t612 * t202 * t613 + 0.6e1 * t215 * t559 * t219 - t69 * t561 * t707 + t1317 + t1319 + t1320 + t1322 - t1325;
  t1572 = t1333 * t282;
  t1592 = t265 * t476 / 0.8e1 - t77 * t479 * t268 * t485 / 0.128e3 - 0.7e1 / 0.256e3 * t567 * t493 + t567 * t499 / 0.256e3 + t567 * t504 / 0.256e3 + t719 * t509 / 0.128e3 + t87 * t269 * t513 / 0.32e2;
  t1615 = t425 * t589;
  t1637 = t189 * t164 * t273;
  t1649 = params->beta * t1592 * t105 - t276 * t517 * t162 * t280 - t574 * t260 + 0.2e1 * t577 * t733 - t184 * t742 - t726 * t198 + 0.2e1 * t729 * t581 - t251 * t590 + 0.2e1 * t274 * t78 * t524 - 0.6e1 * t277 * t162 * t1030 * t197 * t259 * t580 + 0.2e1 * t522 * t1615 * t259 + 0.2e1 * t522 * t523 * t741 - t274 * t556 + 0.2e1 * t278 * t1464 * t580 - t99 * t188 * (t190 * t725 * t171 * t192 - 0.2e1 * t189 * t380 * t273 * t531 + t190 * t273 * t496 * t192 - t253 * t573 * t237 * t83 + t100 * t85 * t1592 - 0.3e1 * t1637 * t544 + t1637 * t548);
  t1654 = t592 * t461;
  t1658 = t783 * t262;
  r->d3fdrszxt = 0.2e1 * t69 * t200 * t1658 * t282 - t69 * t1559 * t282 + t69 * t1649 * t201 - t69 * t1654 * t262 + 0.3e1 * t215 * t593 * t219 - t69 * t561 * t744 - 0.3e1 * t1332 * t1572;
  r->d3fdrszxs[0] = 0.0e0;
  r->d3fdrszxs[1] = 0.0e0;
  t1681 = t265 * t374 / 0.8e1 + t77 * t381 * t268 * t392 / 0.128e3 - t567 * t397 / 0.64e2 + t567 * t403 / 0.256e3 - t77 * t406 * t268 * t410 / 0.256e3 + 0.3e1 / 0.128e3 * t87 * t269 * t415;
  t1705 = t273 * t388;
  t1723 = params->beta * t1681 * t105 - t276 * t419 * t162 * t280 - 0.2e1 * t574 * t198 + 0.4e1 * t577 * t581 - 0.2e1 * t184 * t590 + 0.2e1 * t274 * t428 - 0.6e1 * t278 * t1211 * t580 + 0.4e1 * t522 * t523 * t589 - t274 * t454 + 0.2e1 * t278 * t1219 * t580 - t99 * t188 * (0.2e1 * t190 * t573 * t171 * t192 + t190 * t273 * t400 * t192 + t100 * t85 * t1681 + 0.2e1 * t432 * t1705 * t436 - t447 * t1705 * t448);
  r->d3fdrs2xt = 0.2e1 * t69 * t1312 * t282 + t69 * t1723 * t201 - t69 * t787 * t282 - 0.2e1 * t69 * t561 * t592;
  t1743 = -t750 * t93 / 0.16e2 + 0.3e1 / 0.256e3 * t77 * t165 * t70 * t174 - 0.3e1 / 0.128e3 * t87 * t753 * t179;
  t1756 = t761 * t162;
  t1762 = t164 * t760;
  t1769 = t85 * t757;
  t1781 = params->beta * t1743 * t105 - 0.2e1 * t276 * t573 * t162 * t280 + 0.2e1 * t766 * t183 * t378 * t770 - t577 * t773 - t758 * t198 + 0.4e1 * t1756 * t523 * t85 - 0.2e1 * t274 * t590 - 0.6e1 * t768 * t1460 * t1762 + 0.4e1 * t278 * t1615 * t580 + 0.2e1 * t278 * t523 * t1769 - t99 * t188 * (t190 * t757 * t171 * t192 + t100 * t85 * t1743);
  r->d3fdrsxt2 = 0.2e1 * t69 * t1563 * t778 - 0.2e1 * t69 * t1654 * t282 + t69 * t1781 * t201 - t69 * t561 * t775;
  r->d3fdrsxtxs[0] = 0.0e0;
  r->d3fdrsxtxs[1] = 0.0e0;
  r->d3fdrs2xs[0] = 0.0e0;
  r->d3fdrs2xs[1] = 0.0e0;
  r->d3fdrsxs2[0] = 0.0e0;
  r->d3fdrsxs2[1] = 0.0e0;
  r->d3fdrsxs2[2] = 0.0e0;
  t1794 = 0.24e2 * r->z * t25 * t58;
  t1797 = 0.36e2 * t17 * t209 * t58;
  t1800 = 0.12e2 * t204 * t606 * t58;
  t1806 = -0.8e1 / 0.27e2 / t63 / t19 + 0.8e1 / 0.27e2 / t64 / t22;
  t1808 = t18 * t1806 * t58;
  t1811 = 0.19751673498613801407e-1 * t1806 * t30 * t55;
  t1812 = t613 * t219;
  t1831 = t19 * t19;
  t1834 = t22 * t22;
  t1838 = 0.4e1 / 0.27e2 / t20 / t1831 - 0.4e1 / 0.27e2 / t23 / t1834;
  t1852 = t89 * t435;
  t1877 = -(t1794 + t1797 + t1800 + t1808 + t1811) * t78 * t81 + 0.9e1 * t650 * t234 - 0.36e2 * t232 * t654 + 0.9e1 * t232 * t657 + 0.60e2 * t80 * t435 * t1812 - 0.36e2 * t80 * t244 * t219 * t623 + 0.3e1 * t80 * t92 * t1838;
  t1888 = t73 * t219;
  t1896 = t637 * t237;
  t1897 = t93 * t1896;
  t1902 = t230 * t93;
  t1944 = -t71 * t223 * t1838 / 0.16e2 - 0.15e2 / 0.128e3 * t87 * t90 * t169 * t93 * t1812 + 0.15e2 / 0.256e3 * t77 * t86 * t88 * t1852 * t93 * t219 * t623 - t229 * t230 * t93 * t1877 * t83 / 0.1024e4 - 0.3e1 / 0.4e1 * t71 * t244 * t73 * t1812 + 0.9e1 / 0.16e2 * t71 * t92 * t1888 * t623 - t87 * t90 * t245 * t1838 / 0.256e3 - t229 * t230 * t1897 * t83 / 0.1024e4 + 0.3e1 / 0.512e3 * t636 * t1902 * t530 * t660 + 0.3e1 / 0.256e3 * t229 * t644 * t1546 * t219 + 0.3e1 / 0.256e3 * t229 * t644 * t547 * t623 - 0.3e1 / 0.1024e4 * t229 * t1902 * t660 * t237 * t83 - 0.3e1 / 0.128e3 * t636 * t644 * t1478 * t219 - 0.3e1 / 0.512e3 * t77 * t78 * t947 * t88 * t230 * t1897 * t956 + 0.3e1 / 0.512e3 * t636 * t230 * t1897 * t390 - 0.15e2 / 0.256e3 * t229 * t1852 * t93 * t547 * t613 + 0.3e1 / 0.256e3 * t229 * t644 * t1503 * t219;
  t1961 = t98 * t1896;
  t1965 = t250 * t637;
  t1996 = -0.6e1 * t100 * t947 * t1961 * t956 - t253 * t98 * t1877 * t83 - 0.3e1 * t253 * t679 * t237 * t83 - 0.3e1 * t253 * t250 * t660 * t83 + t100 * t85 * t1944 + 0.6e1 * t688 * t254 * t1525 - t253 * t1961 * t83 + 0.6e1 * t688 * t1961 * t390 - 0.3e1 * t253 * t1965 * t83 + 0.6e1 * t688 * t1965 * t390 - 0.3e1 * t253 * t697 * t547;
  t2002 = t707 * t461;
  r->d3fdz3 = t1794 + t1797 + t1800 + t1808 + t1811 + 0.6e1 * params->gamma * t1812 * t108 + 0.18e2 * t612 * t263 * t613 + 0.18e2 * t612 * t220 * t623 + 0.9e1 * t215 * t708 * t219 - 0.9e1 * t215 * t711 * t219 + 0.9e1 * t215 * t263 * t623 + 0.3e1 * t215 * t108 * t1838 + t69 * (-0.6e1 * t99 * t1031 * t684 * t259 + params->beta * t1944 * t105 - t99 * t188 * t1996 + 0.6e1 * t522 * t732 * t704 + 0.6e1 * t251 * t685 - 0.3e1 * t251 * t705 - 0.3e1 * t680 * t260) * t201 - 0.3e1 * t69 * t2002 * t262 + 0.2e1 * t69 * t710 * t262 * t783;
  t2042 = 0.3e1 / 0.8e1 * t265 * t628 - t265 * t631 / 0.8e1 + t77 * t634 * t268 * t640 / 0.128e3 + t719 * t646 / 0.32e2 - t719 * t663 / 0.256e3 - t719 * t667 / 0.256e3 + 0.5e1 / 0.64e2 * t87 * t269 * t671 - t87 * t269 * t675 / 0.64e2;
  t2068 = t273 * t637;
  t2086 = params->beta * t2042 * t105 - t276 * t679 * t162 * t280 - 0.2e1 * t726 * t260 + 0.4e1 * t729 * t733 - 0.2e1 * t251 * t742 + 0.2e1 * t274 * t685 - 0.6e1 * t278 * t1030 * t684 * t580 + 0.4e1 * t522 * t732 * t741 - t274 * t705 + 0.2e1 * t278 * t425 * t704 * t580 - t99 * t188 * (-0.2e1 * t253 * t725 * t237 * t83 - t253 * t273 * t660 * t83 + t100 * t85 * t2042 - t253 * t2068 * t83 + 0.2e1 * t688 * t2068 * t390);
  r->d3fdz2xt = 0.2e1 * t69 * t710 * t783 * t282 - 0.6e1 * t215 * t262 * t1572 - t69 * t2002 * t282 + t69 * t2086 * t201 + 0.6e1 * t215 * t745 * t219 + 0.3e1 * t215 * t283 * t623 + 0.6e1 * t612 * t283 * t613 - 0.2e1 * t69 * t747 * t744;
  t2114 = -t27 * t81 * t1888 / 0.8e1 - 0.3e1 / 0.256e3 * t77 * t227 * t70 * t240 - 0.3e1 / 0.64e2 * t87 * t753 * t246;
  t2151 = params->beta * t2114 * t105 - 0.2e1 * t276 * t725 * t162 * t280 + 0.2e1 * t766 * t250 * t378 * t770 - t729 * t773 - t758 * t260 + 0.4e1 * t1756 * t732 * t85 - 0.2e1 * t274 * t742 - 0.6e1 * t768 * t1030 * t259 * t1762 + 0.4e1 * t278 * t425 * t741 * t580 + 0.2e1 * t278 * t732 * t1769 - t99 * t188 * (-t253 * t757 * t237 * t83 + t100 * t85 * t2114);
  r->d3fdzxt2 = -0.2e1 * t69 * t744 * t461 * t282 + 0.2e1 * t69 * t1658 * t778 + t69 * t2151 * t201 + 0.3e1 * t215 * t776 * t219 - 0.3e1 * t215 * t779 * t219 - t69 * t747 * t775;
  r->d3fdzxtxs[0] = 0.0e0;
  r->d3fdzxtxs[1] = 0.0e0;
  r->d3fdz2xs[0] = 0.0e0;
  r->d3fdz2xs[1] = 0.0e0;
  r->d3fdzxs2[0] = 0.0e0;
  r->d3fdzxs2[1] = 0.0e0;
  r->d3fdzxs2[2] = 0.0e0;
  t2175 = t760 * t273;
  t2181 = t276 * t276;
  r->d3fdxt3 = t69 * (0.3e1 / 0.128e3 * t276 * params->BB * t162 * t85 * r->xt * t89 * t94 * t104 - 0.6e1 * t276 * t757 * t162 * t280 + 0.6e1 * t766 * t2175 * t378 * t425 * t164 - 0.6e1 * t2181 * t98 * t945 * t1030 * t380 * t2175 + 0.6e1 * t768 * t769 * t273 * t757 - 0.3e1 / 0.128e3 * t767 * t378 * t187 * t164 * params->BB * r->xt * t1902) * t201 - 0.3e1 * t69 * t775 * t461 * t282 + 0.2e1 * t69 * t778 * t282 * t783;
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
#define maple2c_func  xc_gga_c_pbe_func
