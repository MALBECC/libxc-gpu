/* 
  This file was generated automatically with /nfs/data-012/marques/software/source/libxc/svn/scripts/maple2c.pl.
  Do not edit this file directly as it can be overwritten!!

  This Source Code Form is subject to the terms of the Mozilla Public
  License, v. 2.0. If a copy of the MPL was not distributed with this
  file, You can obtain one at http://mozilla.org/MPL/2.0/.

  Maple version     : Maple 2016 (X86 64 LINUX)
  Maple source      : ../maple/mgga_x_mbeefvdw.mpl
  Type of functional: work_mgga_x
*/

static void 
xc_mgga_x_mbeefvdw_enhance(const xc_func_type_cuda *pt, xc_mgga_work_x_t *r)
{
  double t1, t2, t3, t4, t6, t7, t10, t11;
  double t13, t15, t16, t17, t19, t24, t26, t27;
  double t28, t31, t34, t35, t36, t37, t39, t40;
  double t41, t43, t46, t47, t48, t49, t51, t54;
  double t55, t57, t58, t60, t61, t62, t64, t67;
  double t68, t69, t70, t73, t79, t83, t84, t87;
  double t90, t91, t94, t117, t118, t119, t120, t123;
  double t125, t126, t127, t130, t131, t133, t134, t136;
  double t137, t140, t141, t142, t145, t147, t149, t150;
  double t152, t154, t155, t162, t165, t169, t170, t173;
  double t176, t177, t182, t183, t185, t186, t189, t192;
  double t195, t197, t200, t202, t205, t210, t211, t213;
  double t218, t235, t248, t259, t274, t275, t276, t278;
  double t281, t282, t285, t286, t289, t290, t293, t295;
  double t298, t299, t301, t303, t305, t307, t308, t311;
  double t314, t317, t320, t323, t326, t328, t330, t332;
  double t336, t345, t362, t371, t372, t375, t377, t381;
  double t383, t384, t387, t398, t399, t401, t402, t403;
  double t404, t406, t407, t409, t410, t413, t414, t419;
  double t420, t421, t423, t427, t432, t433, t435, t436;
  double t437, t439, t440, t441, t443, t446, t447, t449;
  double t451, t457, t458, t459, t461, t462, t463, t465;
  double t468, t469, t471, t473, t474, t475, t477, t478;
  double t479, t481, t484, t485, t487, t489, t492, t494;
  double t498, t509, t516, t525, t527, t532, t537, t554;
  double t582, t602, t605, t629, t647, t664, t668, t672;
  double t676, t677, t679, t682, t684, t688, t692, t717;
  double t720, t721, t723, t724, t726, t728, t729, t734;
  double t735, t737, t739, t741, t743, t745, t747, t750;
  double t752, t754, t756, t758, t760, t762, t764, t766;
  double t768, t770, t780, t791, t801, t804, t820, t837;
  double t844, t847, t849, t852, t855, t858, t873, t876;
  double t888, t903, t912, t913, t916, t917, t919, t922;
  double t924, t925, t931, t932, t934, t936, t938, t941;
  double t943, t945, t947, t954, t956, t958, t961, t963;
  double t965, t967, t972, t974, t976, t979, t981, t983;
  double t991, t994, t1003, t1034, t1063, t1087, t1100;


  t1 = M_CBRT6;
  t2 = 0.31415926535897932385e1 * 0.31415926535897932385e1;
  t3 = POW_1_3(t2);
  t4 = t3 * t3;
  t6 = t1 / t4;
  t7 = r->x * r->x;
  t10 = 0.65124e1 + t6 * t7 / 0.24e2;
  t11 = 0.1e1 / t10;
  t13 = t6 * t7 * t11;
  t15 = t13 / 0.12e2 - 0.1e1;
  t16 = t15 * t15;
  t17 = t16 * t16;
  t19 = t16 * t15;
  t24 = 0.3e1 / 0.8e1 + 0.35e2 / 0.8e1 * t17 - 0.15e2 / 0.4e1 * t16;
  t26 = r->t - t7 / 0.8e1;
  t27 = t26 * t26;
  t28 = t1 * t1;
  t31 = 0.1e1 / t3 / t2;
  t34 = 0.1e1 - 0.25e2 / 0.81e2 * t27 * t28 * t31;
  t35 = t34 * t34;
  t36 = t35 * t35;
  t37 = t36 * t35;
  t39 = t2 * t2;
  t40 = 0.1e1 / t39;
  t41 = t27 * t26 * t40;
  t43 = 0.1e1 + 0.250e3 / 0.243e3 * t41;
  t46 = 0.1e1 + 0.250e3 / 0.243e3 * t41 * t43;
  t47 = t46 * t46;
  t48 = 0.1e1 / t47;
  t49 = t37 * t48;
  t51 = -0.1e1 / 0.2e1 + 0.3e1 / 0.2e1 * t49;
  t54 = t36 * t36;
  t55 = t54 * t34;
  t57 = 0.1e1 / t47 / t46;
  t58 = t55 * t57;
  t60 = t35 * t34;
  t61 = 0.1e1 / t46;
  t62 = t60 * t61;
  t64 = -0.5e1 / 0.2e1 * t58 + 0.3e1 / 0.2e1 * t62;
  t67 = t54 * t36;
  t68 = t47 * t47;
  t69 = 0.1e1 / t68;
  t70 = t67 * t69;
  t73 = 0.3e1 / 0.8e1 + 0.35e2 / 0.8e1 * t70 - 0.15e2 / 0.4e1 * t49;
  t79 = t15 * t60;
  t83 = -0.1e1 / 0.2e1 + 0.3e1 / 0.2e1 * t16;
  t84 = t83 * t60;
  t87 = 0.10451438955835000000e1 + 0.61869984312500000000e-2 * t17 - 0.50282912000000000000e-1 * t19 - 0.85128253912500000000e-1 * t16 - 0.500749348e-6 * t24 * t51 + 0.574317889e-7 * t24 * t64 - 0.340722258e-8 * t24 * t73 + 0.69727705930000000000e-1 * t62 - 0.35198535500000000000e-2 * t58 + 0.12147009850000000000e-1 * t13 - 0.453837246e-1 * t79 * t61 + 0.222650139e-1 * t84 * t61;
  t90 = 0.5e1 / 0.2e1 * t19 - t13 / 0.8e1 + 0.3e1 / 0.2e1;
  t91 = t90 * t60;
  t94 = t24 * t60;
  t117 = -0.192374554e-1 * t91 * t61 - 0.919317034e-6 * t94 * t61 + 0.21768185977500000000e-1 * t49 + 0.318024096e-1 * t15 * t51 - 0.608338264e-2 * t15 * t64 + 0.61919587625000000000e-3 * t70 - 0.100478906e-6 * t15 * t73 - 0.521818079e-2 * t83 * t51 - 0.657949254e-6 * t83 * t64 + 0.201895739e-6 * t83 * t73 + 0.133707403e-6 * t90 * t51 - 0.549909413e-7 * t90 * t64 + 0.397324768e-8 * t90 * t73;
  r->f = t87 + t117;

  if(r->order < 1) return;

  r->dfdrs = 0.0e0;
  t118 = t37 * t57;
  t119 = t27 * t40;
  t120 = t43 * r->x;
  t123 = t27 * t27;
  t125 = t39 * t39;
  t126 = 0.1e1 / t125;
  t127 = t123 * t26 * t126;
  t130 = -0.125e3 / 0.162e3 * t119 * t120 - 0.15625e5 / 0.19683e5 * t127 * r->x;
  t131 = t118 * t130;
  t133 = t55 * t69;
  t134 = t133 * t130;
  t136 = t60 * t48;
  t137 = t136 * t130;
  t140 = 0.1e1 / t68 / t46;
  t141 = t67 * t140;
  t142 = t141 * t130;
  t145 = t6 * r->x * t11;
  t147 = t28 * t31;
  t149 = t10 * t10;
  t150 = 0.1e1 / t149;
  t152 = t147 * t7 * r->x * t150;
  t154 = t145 / 0.6e1 - t152 / 0.144e3;
  t155 = t15 * t154;
  t162 = t154 * t60;
  t165 = t16 * t154;
  t169 = 0.15e2 / 0.2e1 * t165 - t145 / 0.4e1 + t152 / 0.96e2;
  t170 = t169 * t60;
  t173 = t19 * t154;
  t176 = 0.35e2 / 0.2e1 * t173 - 0.15e2 / 0.2e1 * t155;
  t177 = t176 * t60;
  t182 = t36 * t34;
  t183 = t182 * t48;
  t185 = t147 * r->x;
  t186 = t183 * t26 * t185;
  t189 = 0.25e2 / 0.18e2 * t186 - 0.3e1 * t131;
  t192 = -0.43536371955000000000e-1 * t131 + 0.10559560650000000000e-1 * t134 - 0.69727705930000000000e-1 * t137 - 0.24767835050000000000e-2 * t142 - 0.1565454237e-1 * t155 * t51 - 0.1973847762e-5 * t155 * t64 + 0.605687217e-6 * t155 * t73 - 0.453837246e-1 * t162 * t61 - 0.192374554e-1 * t170 * t61 - 0.919317034e-6 * t177 * t61 + 0.318024096e-1 * t154 * t51 + 0.318024096e-1 * t15 * t189;
  t195 = t54 * t57;
  t197 = t195 * t26 * t185;
  t200 = t35 * t61;
  t202 = t200 * t26 * t185;
  t205 = -0.125e3 / 0.36e2 * t197 + 0.15e2 / 0.2e1 * t134 + 0.25e2 / 0.36e2 * t202 - 0.3e1 / 0.2e1 * t137;
  t210 = t54 * t60;
  t211 = t210 * t69;
  t213 = t211 * t26 * t185;
  t218 = 0.875e3 / 0.108e3 * t213 - 0.35e2 / 0.2e1 * t142 - 0.125e3 / 0.36e2 * t186 + 0.15e2 / 0.2e1 * t131;
  t235 = -0.608338264e-2 * t154 * t64 - 0.608338264e-2 * t15 * t205 - 0.100478906e-6 * t154 * t73 - 0.100478906e-6 * t15 * t218 - 0.17025650782500000000e0 * t155 - 0.521818079e-2 * t83 * t189 - 0.657949254e-6 * t83 * t205 + 0.201895739e-6 * t83 * t218 - 0.15084873600000000000e0 * t165 + 0.133707403e-6 * t169 * t51 + 0.133707403e-6 * t90 * t189 - 0.549909413e-7 * t169 * t64;
  t248 = t48 * t130;
  t259 = -0.549909413e-7 * t90 * t205 + 0.397324768e-8 * t169 * t73 + 0.397324768e-8 * t90 * t218 + 0.24747993725000000000e-1 * t173 - 0.500749348e-6 * t176 * t51 + 0.24294019700000000000e-1 * t145 - 0.10122508208333333333e-2 * t152 + 0.453837246e-1 * t79 * t248 - 0.222650139e-1 * t84 * t248 + 0.667950417e-1 * t155 * t62 + 0.192374554e-1 * t91 * t248 + 0.919317034e-6 * t94 * t248;
  t274 = t15 * t35;
  t275 = t274 * t61;
  t276 = t26 * t28;
  t278 = t276 * t31 * r->x;
  t281 = t83 * t35;
  t282 = t281 * t61;
  t285 = t90 * t35;
  t286 = t285 * t61;
  t289 = t24 * t35;
  t290 = t289 * t61;
  t293 = -0.500749348e-6 * t24 * t189 + 0.574317889e-7 * t176 * t64 + 0.574317889e-7 * t24 * t205 - 0.340722258e-8 * t176 * t73 - 0.340722258e-8 * t24 * t218 + 0.32281345337962962963e-1 * t202 - 0.48886854861111111111e-2 * t197 + 0.20155727756944444444e-1 * t186 + 0.11466590300925925926e-2 * t213 - 0.21010983611111111111e-1 * t275 * t278 + 0.10307876805555555556e-1 * t282 * t278 - 0.89062293518518518519e-2 * t286 * t278 - 0.42560973796296296296e-6 * t290 * t278;
  r->dfdx = t192 + t235 + t259 + t293;
  t295 = t119 * t43;
  t298 = 0.250e3 / 0.81e2 * t295 + 0.62500e5 / 0.19683e5 * t127;
  t299 = t136 * t298;
  t301 = t133 * t298;
  t303 = t118 * t298;
  t305 = t141 * t298;
  t307 = t276 * t31;
  t308 = t211 * t307;
  t311 = t183 * t307;
  t314 = -0.875e3 / 0.27e2 * t308 - 0.35e2 / 0.2e1 * t305 + 0.125e3 / 0.9e1 * t311 + 0.15e2 / 0.2e1 * t303;
  t317 = t195 * t307;
  t320 = t200 * t307;
  t323 = 0.125e3 / 0.9e1 * t317 + 0.15e2 / 0.2e1 * t301 - 0.25e2 / 0.9e1 * t320 - 0.3e1 / 0.2e1 * t299;
  t326 = t275 * t307;
  t328 = t282 * t307;
  t330 = t286 * t307;
  t332 = t290 * t307;
  t336 = -0.50e2 / 0.9e1 * t311 - 0.3e1 * t303;
  t345 = -0.69727705930000000000e-1 * t299 + 0.10559560650000000000e-1 * t301 - 0.43536371955000000000e-1 * t303 - 0.24767835050000000000e-2 * t305 - 0.340722258e-8 * t24 * t314 + 0.574317889e-7 * t24 * t323 + 0.84043934444444444444e-1 * t326 - 0.41231507222222222222e-1 * t328 + 0.35624917407407407407e-1 * t330 + 0.17024389518518518519e-5 * t332 + 0.318024096e-1 * t15 * t336 - 0.608338264e-2 * t15 * t323 - 0.100478906e-6 * t15 * t314 - 0.521818079e-2 * t83 * t336;
  t362 = t48 * t298;
  t371 = -0.12912538135185185185e0 * t320 + 0.19554741944444444444e-1 * t317 - 0.80622911027777777778e-1 * t311 - 0.45866361203703703704e-2 * t308 - 0.657949254e-6 * t83 * t323 + 0.201895739e-6 * t83 * t314 + 0.133707403e-6 * t90 * t336 - 0.549909413e-7 * t90 * t323 + 0.397324768e-8 * t90 * t314 - 0.500749348e-6 * t24 * t336 + 0.453837246e-1 * t79 * t362 - 0.222650139e-1 * t84 * t362 + 0.192374554e-1 * t91 * t362 + 0.919317034e-6 * t94 * t362;
  r->dfdt = t345 + t371;
  r->dfdu = 0.0e0;

  if(r->order < 2) return;

  r->d2fdrs2 = 0.0e0;
  t372 = t6 * t11;
  t375 = t147 * t7 * t150;
  t377 = t7 * t7;
  t381 = t40 * t377 / t149 / t10;
  t383 = t372 / 0.6e1 - 0.5e1 / 0.144e3 * t375 + t381 / 0.144e3;
  t384 = t15 * t383;
  t387 = t154 * t154;
  t398 = t54 * t35 * t69;
  t399 = t398 * t27;
  t401 = 0.1e1 / t4 / t39;
  t402 = t1 * t401;
  t403 = t402 * t7;
  t404 = t399 * t403;
  t406 = t210 * t140;
  t407 = t406 * t26;
  t409 = t147 * r->x * t130;
  t410 = t407 * t409;
  t413 = t7 * t28 * t31;
  t414 = t211 * t413;
  t419 = t67 / t68 / t47;
  t420 = t130 * t130;
  t421 = t419 * t420;
  t423 = t26 * t40;
  t427 = t123 * t126;
  t432 = 0.125e3 / 0.324e3 * t423 * t43 * t7 + 0.31250e5 / 0.19683e5 * t427 * t7 - 0.125e3 / 0.162e3 * t295 - 0.15625e5 / 0.19683e5 * t127;
  t433 = t141 * t432;
  t435 = t36 * t48;
  t436 = t435 * t27;
  t437 = t436 * t403;
  t439 = t182 * t57;
  t440 = t439 * t26;
  t441 = t440 * t409;
  t443 = t183 * t413;
  t446 = t37 * t69;
  t447 = t446 * t420;
  t449 = t118 * t432;
  t451 = 0.240625e6 / 0.2916e4 * t404 - 0.1750e4 / 0.27e2 * t410 - 0.875e3 / 0.432e3 * t414 + 0.875e3 / 0.108e3 * t308 + 0.175e3 / 0.2e1 * t421 - 0.35e2 / 0.2e1 * t433 - 0.15625e5 / 0.972e3 * t437 + 0.125e3 / 0.9e1 * t441 + 0.125e3 / 0.144e3 * t443 - 0.125e3 / 0.36e2 * t311 - 0.45e2 / 0.2e1 * t447 + 0.15e2 / 0.2e1 * t449;
  t457 = t36 * t60 * t57;
  t458 = t457 * t27;
  t459 = t458 * t403;
  t461 = t54 * t69;
  t462 = t461 * t26;
  t463 = t462 * t409;
  t465 = t195 * t413;
  t468 = t55 * t140;
  t469 = t468 * t420;
  t471 = t133 * t432;
  t473 = t34 * t61;
  t474 = t473 * t27;
  t475 = t474 * t403;
  t477 = t35 * t48;
  t478 = t477 * t26;
  t479 = t478 * t409;
  t481 = t200 * t413;
  t484 = t60 * t57;
  t485 = t484 * t420;
  t487 = t136 * t432;
  t489 = -0.6250e4 / 0.243e3 * t459 + 0.125e3 / 0.6e1 * t463 + 0.125e3 / 0.144e3 * t465 - 0.125e3 / 0.36e2 * t317 - 0.30e2 * t469 + 0.15e2 / 0.2e1 * t471 + 0.625e3 / 0.486e3 * t475 - 0.25e2 / 0.18e2 * t479 - 0.25e2 / 0.144e3 * t481 + 0.25e2 / 0.36e2 * t320 + 0.3e1 * t485 - 0.3e1 / 0.2e1 * t487;
  t492 = t16 * t387;
  t494 = t19 * t383;
  t498 = 0.105e3 / 0.2e1 * t492 + 0.35e2 / 0.2e1 * t494 - 0.15e2 / 0.2e1 * t387 - 0.15e2 / 0.2e1 * t384;
  t509 = 0.3125e4 / 0.486e3 * t437 - 0.50e2 / 0.9e1 * t441 - 0.25e2 / 0.72e2 * t443 + 0.25e2 / 0.18e2 * t311 + 0.9e1 * t447 - 0.3e1 * t449;
  t516 = 0.605687217e-6 * t384 * t73 + 0.667950417e-1 * t387 * t60 * t61 - 0.1973847762e-5 * t387 * t64 + 0.605687217e-6 * t387 * t73 - 0.1565454237e-1 * t387 * t51 - 0.340722258e-8 * t24 * t451 - 0.681444516e-8 * t176 * t218 + 0.574317889e-7 * t24 * t489 - 0.340722258e-8 * t498 * t73 + 0.1148635778e-6 * t176 * t205 - 0.500749348e-6 * t24 * t509 + 0.574317889e-7 * t498 * t64 - 0.1001498696e-5 * t176 * t189;
  t525 = t15 * t387;
  t527 = t16 * t383;
  t532 = 0.15e2 * t525 + 0.15e2 / 0.2e1 * t527 - t372 / 0.4e1 + 0.5e1 / 0.96e2 * t375 - t381 / 0.96e2;
  t537 = t48 * t432;
  t554 = 0.397324768e-8 * t90 * t451 - 0.500749348e-6 * t498 * t51 + 0.794649536e-8 * t169 * t218 - 0.549909413e-7 * t90 * t489 + 0.397324768e-8 * t532 * t73 + 0.384749108e-1 * t170 * t248 + 0.192374554e-1 * t91 * t537 + 0.1838634068e-5 * t177 * t248 + 0.919317034e-6 * t94 * t537 + 0.907674492e-1 * t162 * t248 + 0.453837246e-1 * t79 * t537 - 0.222650139e-1 * t84 * t537 + 0.667950417e-1 * t384 * t62 - 0.1099818826e-6 * t169 * t205;
  t582 = -0.549909413e-7 * t532 * t64 + 0.133707403e-6 * t90 * t509 + 0.201895739e-6 * t83 * t451 + 0.133707403e-6 * t532 * t51 + 0.267414806e-6 * t169 * t189 - 0.657949254e-6 * t83 * t489 - 0.521818079e-2 * t83 * t509 - 0.100478906e-6 * t15 * t451 - 0.608338264e-2 * t15 * t489 - 0.100478906e-6 * t383 * t73 - 0.200957812e-6 * t154 * t218 - 0.1216676528e-1 * t154 * t205 + 0.318024096e-1 * t15 * t509;
  t602 = -0.608338264e-2 * t383 * t64 + 0.636048192e-1 * t154 * t189 - 0.1335900834e0 * t155 * t137 - 0.80703363344907407408e-2 * t481 + 0.12221713715277777778e-2 * t465 - 0.50389319392361111110e-2 * t443 - 0.28666475752314814815e-3 * t414 - 0.17025650782500000000e0 * t387 + 0.22265573379629629630e-2 * t286 * t413 + 0.10640243449074074074e-6 * t290 * t413 + 0.59780269144375857339e-1 * t475 - 0.36212485082304526749e-1 * t459 + 0.93313554430298353907e-1 * t437 + 0.11678934565757887517e-1 * t404;
  t605 = t57 * t420;
  t629 = -0.907674492e-1 * t79 * t605 + 0.445300278e-1 * t84 * t605 - 0.384749108e-1 * t91 * t605 - 0.1838634068e-5 * t94 * t605 + 0.24294019700000000000e-1 * t372 + 0.10122508208333333333e-2 * t381 - 0.453837246e-1 * t383 * t60 * t61 - 0.192374554e-1 * t532 * t60 * t61 - 0.919317034e-6 * t498 * t60 * t61 + 0.13060911586500000000e0 * t447 - 0.43536371955000000000e-1 * t449 - 0.42238242600000000000e-1 * t469 + 0.10559560650000000000e-1 * t471;
  t647 = 0.13945541186000000000e0 * t485 - 0.69727705930000000000e-1 * t487 + 0.12383917525000000000e-1 * t421 - 0.24767835050000000000e-2 * t433 - 0.17025650782500000000e0 * t384 - 0.30169747200000000000e0 * t525 - 0.15084873600000000000e0 * t527 + 0.74243981175000000000e-1 * t492 + 0.24747993725000000000e-1 * t494 - 0.3130908474e-1 * t155 * t189 - 0.3947695524e-5 * t155 * t205 + 0.1211374434e-5 * t155 * t218 - 0.21010983611111111111e-1 * t326 + 0.10307876805555555556e-1 * t328;
  t664 = t154 * t35 * t61;
  t668 = t169 * t35 * t61;
  t672 = t176 * t35 * t61;
  t676 = t15 * t34 * t61;
  t677 = t27 * t1;
  t679 = t677 * t401 * t7;
  t682 = -0.89062293518518518519e-2 * t330 - 0.42560973796296296296e-6 * t332 + 0.52527459027777777778e-2 * t275 * t413 - 0.25769692013888888890e-2 * t282 * t413 - 0.1565454237e-1 * t384 * t51 - 0.1973847762e-5 * t384 * t64 - 0.64562690675925925926e-1 * t479 + 0.29332112916666666666e-1 * t463 - 0.80622911027777777777e-1 * t441 - 0.91732722407407407408e-2 * t410 - 0.42021967222222222222e-1 * t664 * t278 - 0.17812458703703703704e-1 * t668 * t278 - 0.85121947592592592592e-6 * t672 * t278 - 0.38909228909465020576e-1 * t676 * t679;
  t684 = t83 * t34 * t61;
  t688 = t90 * t34 * t61;
  t692 = t24 * t34 * t61;
  t717 = 0.19088660751028806585e-1 * t684 * t679 - 0.16493017318244170096e-1 * t688 * t679 - 0.78816618141289437585e-6 * t692 * t679 + 0.42021967222222222222e-1 * t274 * t248 * t278 - 0.20615753611111111112e-1 * t281 * t248 * t278 + 0.61847260833333333335e-1 * t155 * t200 * t278 + 0.17812458703703703704e-1 * t285 * t248 * t278 + 0.85121947592592592592e-6 * t289 * t248 * t278 + 0.32281345337962962963e-1 * t320 - 0.48886854861111111111e-2 * t317 + 0.20155727756944444444e-1 * t311 + 0.11466590300925925926e-2 * t308 + 0.318024096e-1 * t383 * t51 - 0.50612541041666666666e-2 * t375;
  r->d2fdx2 = t516 + t554 + t582 + t602 + t629 + t647 + t682 + t717;
  t720 = t677 * t401;
  t721 = t398 * t720;
  t723 = t147 * t298;
  t724 = t407 * t723;
  t726 = t211 * t147;
  t728 = t298 * t298;
  t729 = t419 * t728;
  t734 = 0.500e3 / 0.81e2 * t423 * t43 + 0.500000e6 / 0.19683e5 * t427;
  t735 = t141 * t734;
  t737 = t435 * t720;
  t739 = t440 * t723;
  t741 = t183 * t147;
  t743 = t446 * t728;
  t745 = t118 * t734;
  t747 = 0.962500e6 / 0.729e3 * t721 + 0.7000e4 / 0.27e2 * t724 - 0.875e3 / 0.27e2 * t726 + 0.175e3 / 0.2e1 * t729 - 0.35e2 / 0.2e1 * t735 - 0.62500e5 / 0.243e3 * t737 - 0.500e3 / 0.9e1 * t739 + 0.125e3 / 0.9e1 * t741 - 0.45e2 / 0.2e1 * t743 + 0.15e2 / 0.2e1 * t745;
  t750 = t457 * t720;
  t752 = t462 * t723;
  t754 = t195 * t147;
  t756 = t468 * t728;
  t758 = t133 * t734;
  t760 = t473 * t720;
  t762 = t478 * t723;
  t764 = t200 * t147;
  t766 = t484 * t728;
  t768 = t136 * t734;
  t770 = -0.100000e6 / 0.243e3 * t750 - 0.250e3 / 0.3e1 * t752 + 0.125e3 / 0.9e1 * t754 - 0.30e2 * t756 + 0.15e2 / 0.2e1 * t758 + 0.5000e4 / 0.243e3 * t760 + 0.50e2 / 0.9e1 * t762 - 0.25e2 / 0.9e1 * t764 + 0.3e1 * t766 - 0.3e1 / 0.2e1 * t768;
  t780 = 0.25000e5 / 0.243e3 * t737 + 0.200e3 / 0.9e1 * t739 - 0.50e2 / 0.9e1 * t741 + 0.9e1 * t743 - 0.3e1 * t745;
  t791 = t48 * t734;
  t801 = -0.340722258e-8 * t24 * t747 + 0.574317889e-7 * t24 * t770 + 0.397324768e-8 * t90 * t747 - 0.500749348e-6 * t24 * t780 - 0.549909413e-7 * t90 * t770 + 0.133707403e-6 * t90 * t780 + 0.201895739e-6 * t83 * t747 - 0.657949254e-6 * t83 * t770 + 0.453837246e-1 * t79 * t791 - 0.222650139e-1 * t84 * t791 + 0.192374554e-1 * t91 * t791 + 0.919317034e-6 * t94 * t791 - 0.12912538135185185185e0 * t764;
  t804 = t57 * t728;
  t820 = 0.19554741944444444444e-1 * t754 - 0.45866361203703703704e-2 * t726 - 0.907674492e-1 * t79 * t804 + 0.445300278e-1 * t84 * t804 - 0.384749108e-1 * t91 * t804 - 0.1838634068e-5 * t94 * t804 + 0.12383917525000000000e-1 * t729 - 0.69727705930000000000e-1 * t768 + 0.10559560650000000000e-1 * t758 - 0.43536371955000000000e-1 * t745 - 0.24767835050000000000e-2 * t735 + 0.13945541186000000000e0 * t766 - 0.42238242600000000000e-1 * t756;
  t837 = t61 * t28 * t31;
  t844 = 0.13060911586500000000e0 * t743 - 0.521818079e-2 * t83 * t780 - 0.100478906e-6 * t15 * t747 - 0.608338264e-2 * t15 * t770 + 0.318024096e-1 * t15 * t780 - 0.80622911027777777778e-1 * t741 + 0.14930168708847736626e1 * t737 + 0.95648430631001371741e0 * t760 - 0.57939976131687242797e0 * t750 + 0.18686295305212620028e0 * t721 + 0.84043934444444444444e-1 * t274 * t837 - 0.41231507222222222222e-1 * t281 * t837 + 0.35624917407407407407e-1 * t285 * t837;
  t847 = t274 * t48;
  t849 = t276 * t31 * t298;
  t852 = t281 * t48;
  t855 = t285 * t48;
  t858 = t289 * t48;
  t873 = 0.17024389518518518519e-5 * t289 * t837 - 0.16808786888888888889e0 * t847 * t849 + 0.82463014444444444444e-1 * t852 * t849 - 0.71249834814814814814e-1 * t855 * t849 - 0.34048779037037037038e-5 * t858 * t849 + 0.32249164411111111112e0 * t739 + 0.25825076270370370370e0 * t762 - 0.11732845166666666666e0 * t752 + 0.36693088962962962963e-1 * t724 - 0.62254766255144032922e0 * t676 * t720 + 0.30541857201646090535e0 * t684 * t720 - 0.26388827709190672153e0 * t688 * t720 - 0.12610658902606310014e-4 * t692 * t720;
  r->d2fdt2 = t801 + t820 + t844 + t873;
  r->d2fdu2 = 0.0e0;
  r->d2fdrsx = 0.0e0;
  r->d2fdrst = 0.0e0;
  r->d2fdrsu = 0.0e0;
  t876 = t57 * t130 * t298;
  t888 = t130 * t26 * t147;
  t903 = t677 * t401 * r->x;
  t912 = t402 * r->x;
  t913 = t458 * t912;
  t916 = t147 * r->x * t298;
  t917 = t462 * t916;
  t919 = t195 * t185;
  t922 = t461 * t130 * t307;
  t924 = t130 * t298;
  t925 = t468 * t924;
  t931 = -0.125e3 / 0.81e2 * t423 * t120 - 0.125000e6 / 0.19683e5 * t427 * r->x;
  t932 = t133 * t931;
  t934 = t474 * t912;
  t936 = t478 * t916;
  t938 = t200 * t185;
  t941 = t477 * t130 * t307;
  t943 = t484 * t924;
  t945 = t136 * t931;
  t947 = 0.25000e5 / 0.243e3 * t913 + 0.125e3 / 0.12e2 * t917 - 0.125e3 / 0.36e2 * t919 - 0.125e3 / 0.3e1 * t922 - 0.30e2 * t925 + 0.15e2 / 0.2e1 * t932 - 0.1250e4 / 0.243e3 * t934 - 0.25e2 / 0.36e2 * t936 + 0.25e2 / 0.36e2 * t938 + 0.25e2 / 0.9e1 * t941 + 0.3e1 * t943 - 0.3e1 / 0.2e1 * t945;
  t954 = t436 * t912;
  t956 = t440 * t916;
  t958 = t183 * t185;
  t961 = t439 * t130 * t307;
  t963 = t446 * t924;
  t965 = t118 * t931;
  t967 = -0.6250e4 / 0.243e3 * t954 - 0.25e2 / 0.9e1 * t956 + 0.25e2 / 0.18e2 * t958 + 0.100e3 / 0.9e1 * t961 + 0.9e1 * t963 - 0.3e1 * t965;
  t972 = t399 * t912;
  t974 = t407 * t916;
  t976 = t211 * t185;
  t979 = t406 * t130 * t307;
  t981 = t419 * t924;
  t983 = t141 * t931;
  t991 = -0.240625e6 / 0.729e3 * t972 - 0.875e3 / 0.27e2 * t974 + 0.875e3 / 0.108e3 * t976 + 0.3500e4 / 0.27e2 * t979 + 0.175e3 / 0.2e1 * t981 - 0.35e2 / 0.2e1 * t983 + 0.15625e5 / 0.243e3 * t954 + 0.125e3 / 0.18e2 * t956 - 0.125e3 / 0.36e2 * t958 - 0.250e3 / 0.9e1 * t961 - 0.45e2 / 0.2e1 * t963 + 0.15e2 / 0.2e1 * t965;
  t994 = -0.384749108e-1 * t91 * t876 - 0.1838634068e-5 * t94 * t876 - 0.907674492e-1 * t79 * t876 + 0.445300278e-1 * t84 * t876 - 0.667950417e-1 * t155 * t299 - 0.84043934444444444444e-1 * t847 * t888 + 0.41231507222222222222e-1 * t852 * t888 - 0.12369452166666666667e0 * t155 * t35 * t61 * t26 * t147 - 0.35624917407407407407e-1 * t855 * t888 - 0.17024389518518518519e-5 * t858 * t888 + 0.15563691563786008230e0 * t676 * t903 - 0.76354643004115226341e-1 * t684 * t903 + 0.65972069272976680384e-1 * t688 * t903 + 0.31526647256515775034e-5 * t692 * t903 - 0.608338264e-2 * t15 * t947 + 0.318024096e-1 * t154 * t336 - 0.608338264e-2 * t154 * t323 + 0.318024096e-1 * t15 * t967 - 0.521818079e-2 * t83 * t967 - 0.100478906e-6 * t15 * t991;
  t1003 = t48 * t931;
  t1034 = -0.100478906e-6 * t154 * t314 + 0.201895739e-6 * t83 * t991 - 0.657949254e-6 * t83 * t947 + 0.919317034e-6 * t177 * t362 + 0.919317034e-6 * t94 * t1003 + 0.453837246e-1 * t79 * t1003 - 0.222650139e-1 * t84 * t1003 + 0.453837246e-1 * t162 * t362 + 0.192374554e-1 * t170 * t362 + 0.192374554e-1 * t91 * t1003 + 0.605687217e-6 * t155 * t314 - 0.1565454237e-1 * t155 * t336 - 0.1973847762e-5 * t155 * t323 + 0.10559560650000000000e-1 * t932 - 0.69727705930000000000e-1 * t945 - 0.549909413e-7 * t90 * t947 - 0.549909413e-7 * t169 * t323 + 0.133707403e-6 * t90 * t967 + 0.397324768e-8 * t90 * t991 + 0.133707403e-6 * t169 * t336;
  t1063 = 0.397324768e-8 * t169 * t314 + 0.574317889e-7 * t24 * t947 + 0.574317889e-7 * t176 * t323 - 0.500749348e-6 * t24 * t967 - 0.500749348e-6 * t176 * t336 - 0.340722258e-8 * t24 * t991 - 0.340722258e-8 * t176 * t314 + 0.13060911586500000000e0 * t963 + 0.12383917525000000000e-1 * t981 - 0.42238242600000000000e-1 * t925 + 0.13945541186000000000e0 * t943 - 0.24767835050000000000e-2 * t983 - 0.43536371955000000000e-1 * t965 + 0.32281345337962962963e-1 * t938 - 0.48886854861111111111e-2 * t919 + 0.20155727756944444444e-1 * t958 + 0.11466590300925925926e-2 * t976 - 0.46715738263031550069e-1 * t972 + 0.16124582205555555556e0 * t961 - 0.58664225833333333333e-1 * t922;
  t1087 = t48 * t26;
  t1100 = 0.12912538135185185185e0 * t941 + 0.18346544481481481481e-1 * t979 + 0.84043934444444444444e-1 * t664 * t307 + 0.35624917407407407407e-1 * t668 * t307 + 0.17024389518518518519e-5 * t672 * t307 + 0.10307876805555555556e-1 * t282 * t185 - 0.89062293518518518519e-2 * t286 * t185 - 0.42560973796296296296e-6 * t290 * t185 - 0.21010983611111111111e-1 * t275 * t185 - 0.23912107657750342936e0 * t934 + 0.14484994032921810700e0 * t913 - 0.37325421772119341563e0 * t954 - 0.45866361203703703704e-2 * t974 - 0.40311455513888888888e-1 * t956 + 0.14666056458333333333e-1 * t917 - 0.32281345337962962963e-1 * t936 + 0.21010983611111111111e-1 * t274 * t1087 * t916 - 0.10307876805555555556e-1 * t281 * t1087 * t916 + 0.89062293518518518519e-2 * t285 * t1087 * t916 + 0.42560973796296296296e-6 * t289 * t1087 * t916;
  r->d2fdxt = t994 + t1034 + t1063 + t1100;
  r->d2fdxu = 0.0e0;
  r->d2fdtu = 0.0e0;

  if(r->order < 3) return;


}

#define maple2c_order 3
#define maple2c_func  xc_mgga_x_mbeefvdw_enhance
