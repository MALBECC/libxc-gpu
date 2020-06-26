/* 
  This file was generated automatically with /nfs/data-012/marques/software/source/libxc/svn/scripts/maple2c.pl.
  Do not edit this file directly as it can be overwritten!!

  This Source Code Form is subject to the terms of the Mozilla Public
  License, v. 2.0. If a copy of the MPL was not distributed with this
  file, You can obtain one at http://mozilla.org/MPL/2.0/.

  Maple version     : Maple 2016 (X86 64 LINUX)
  Maple source      : ../maple/lda_c_2d_prm.mpl
  Type of functional: work_lda
*/

static void
func0(const xc_func_type_cuda *p, xc_lda_work_t *r)
{
  double t1, t3, t5, t6, t7, t9, t10, t11;
  double t12, t13, t16, t17, t20, t21, t23, t26;
  double t27, t28, t31, t34, t35, t36, t39, t41;
  double t42, t43, t45, t46, t53, t60, t72, t75;
  double t80, t81, t82, t88, t89, t95, t96, t105;
  double t108, t114, t115, t127, t132, t140, t141, t149;
  double t150, t157, t161, t168, t171, t180, t183, t202;

  lda_c_2d_prm_params *params;

  assert(p->params != NULL);
  params = (lda_c_2d_prm_params * )(p->params);

  assert(params->N > 1);

  t1 = 0.1e1 / r->rs;
  t3 = sqrt(0.31415926535897932385e1);
  t5 = 0.22157981704254580414e1 * t1 + t3 / 0.2e1;
  t6 = 0.1e1 / t5;
  t7 = t1 * t6;
  t9 = 0.22157981704254580414e1 * t7 - 0.1e1;
  t10 = t1 * t9;
  t11 = 0.2e1 + params->c;
  t12 = sqrt(t11);
  t13 = 0.1e1 / t12;
  t16 = 0.1e1 / t11;
  t17 = t9 * t16;
  t20 = t5 * t5;
  t21 = 0.1e1 / t20;
  t23 = 0.1e1/POW_3_2(t11);
  t26 = 0.1e1 + params->c;
  t27 = sqrt(t26);
  t28 = 0.1e1 / t27;
  t31 = 0.1e1 / t26;
  r->f = 0.19997916265148655845e0 * t10 * t13 + 0.22565232098914243869e0 * t7 * t17 + 0.99989581325743279224e-1 * t1 * t21 * t23 + 0.39995832530297311691e0 * t10 * t28 + 0.22565232098914243869e0 * t7 * t31;

  if(r->order < 1) return;

  t34 = r->rs * r->rs;
  t35 = 0.1e1 / t34;
  t36 = t35 * t9;
  t39 = t35 * t6;
  t41 = t34 * r->rs;
  t42 = 0.1e1 / t41;
  t43 = t42 * t21;
  t45 = -0.22157981704254580414e1 * t39 + 0.49097615320608071993e1 * t43;
  t46 = t1 * t45;
  t53 = t45 * t16;
  t60 = 0.1e1 / t20 / t5;
  r->dfdrs = -0.19997916265148655845e0 * t36 * t13 + 0.19997916265148655845e0 * t46 * t13 - 0.22565232098914243869e0 * t39 * t17 + 0.50000000000000000004e0 * t43 * t17 + 0.22565232098914243869e0 * t7 * t53 - 0.99989581325743279224e-1 * t35 * t21 * t23 + 0.44311346272637900685e0 * t42 * t60 * t23 - 0.39995832530297311691e0 * t36 * t28 + 0.39995832530297311691e0 * t46 * t28 - 0.22565232098914243869e0 * t39 * t31 + 0.50000000000000000004e0 * t43 * t31;

  if(r->order < 2) return;

  t72 = t42 * t6;
  t75 = t42 * t9;
  t80 = t34 * t34;
  t81 = 0.1e1 / t80;
  t82 = t81 * t21;
  t88 = 0.1e1 / t80 / r->rs;
  t89 = t88 * t60;
  t95 = 0.44315963408509160828e1 * t72 - 0.19639046128243228797e2 * t82 + 0.21758081239931260892e2 * t89;
  t96 = t95 * t16;
  t105 = t35 * t45;
  t108 = t1 * t95;
  t114 = t20 * t20;
  t115 = 0.1e1 / t114;
  r->d2fdrs2 = 0.45130464197828487738e0 * t72 * t17 + 0.39995832530297311690e0 * t75 * t13 - 0.45130464197828487738e0 * t39 * t53 - 0.20000000000000000001e1 * t82 * t17 + 0.10000000000000000001e1 * t43 * t53 + 0.22157981704254580416e1 * t89 * t17 + 0.22565232098914243869e0 * t7 * t96 + 0.19997916265148655845e0 * t43 * t23 + 0.79991665060594623382e0 * t75 * t28 + 0.45130464197828487738e0 * t72 * t31 - 0.39995832530297311690e0 * t105 * t13 + 0.19997916265148655845e0 * t108 * t13 - 0.17724538509055160274e1 * t81 * t60 * t23 + 0.29455500000000000002e1 * t88 * t115 * t23 - 0.79991665060594623382e0 * t105 * t28 + 0.39995832530297311691e0 * t108 * t28 - 0.20000000000000000001e1 * t82 * t31 + 0.22157981704254580416e1 * t89 * t31;

  if(r->order < 3) return;

  t127 = t81 * t6;
  t132 = t88 * t21;
  t140 = 0.1e1 / t80 / t34;
  t141 = t140 * t60;
  t149 = 0.1e1 / t80 / t41;
  t150 = t149 * t115;
  t157 = -0.13294789022552748248e2 * t127 + 0.88375707577094529586e2 * t132 - 0.19582273115938134803e3 * t141 + 0.14463454981022450832e3 * t150;
  t161 = t81 * t9;
  t168 = -0.13539139259348546321e1 * t127 * t17 + 0.13539139259348546321e1 * t72 * t53 + 0.90000000000000000005e1 * t132 * t17 - 0.67695696296742731607e0 * t39 * t96 - 0.60000000000000000005e1 * t82 * t53 - 0.19942183533829122374e2 * t141 * t17 + 0.15000000000000000001e1 * t43 * t96 + 0.66473945112763741248e1 * t89 * t53 + 0.14729284596182421599e2 * t150 * t17 + 0.22565232098914243869e0 * t7 * t157 * t16 - 0.11998749759089193507e1 * t161 * t13 - 0.59993748795445967535e0 * t82 * t23 - 0.23997499518178387015e1 * t161 * t28;
  t171 = t42 * t45;
  t180 = t35 * t95;
  t183 = t1 * t157;
  t202 = -0.13539139259348546321e1 * t127 * t31 + 0.11998749759089193507e1 * t171 * t13 + 0.79760423290748221233e1 * t89 * t23 + 0.23997499518178387014e1 * t171 * t28 + 0.90000000000000000005e1 * t132 * t31 - 0.59993748795445967535e0 * t180 * t13 + 0.19997916265148655845e0 * t183 * t13 - 0.26509950000000000002e2 * t140 * t115 * t23 + 0.26106977203586831737e2 * t149 / t114 / t5 * t23 - 0.11998749759089193507e1 * t180 * t28 + 0.39995832530297311691e0 * t183 * t28 - 0.19942183533829122374e2 * t141 * t31 + 0.14729284596182421599e2 * t150 * t31;
  r->d3fdrs3 = t168 + t202;

  if(r->order < 4) return;


}

static void
func1(const xc_func_type_cuda *p, xc_lda_work_t *r)
{
  double t1, t3, t5, t6, t7, t9, t10, t11;
  double t12, t13, t16, t17, t20, t21, t23, t26;
  double t27, t28, t31, t34, t35, t36, t39, t41;
  double t42, t43, t45, t46, t53, t60, t72, t75;
  double t80, t81, t82, t88, t89, t95, t96, t105;
  double t108, t114, t115, t129, t132, t135, t142, t145;
  double t151, t152, t155, t156, t158, t159, t174, t202;

  lda_c_2d_prm_params *params;

  assert(p->params != NULL);
  params = (lda_c_2d_prm_params * )(p->params);

  assert(params->N > 1);

  t1 = 0.1e1 / r->rs;
  t3 = sqrt(0.31415926535897932385e1);
  t5 = 0.22157981704254580414e1 * t1 + t3 / 0.2e1;
  t6 = 0.1e1 / t5;
  t7 = t1 * t6;
  t9 = 0.22157981704254580414e1 * t7 - 0.1e1;
  t10 = t1 * t9;
  t11 = 0.2e1 + params->c;
  t12 = sqrt(t11);
  t13 = 0.1e1 / t12;
  t16 = 0.1e1 / t11;
  t17 = t9 * t16;
  t20 = t5 * t5;
  t21 = 0.1e1 / t20;
  t23 = 0.1e1/POW_3_2(t11);
  t26 = 0.1e1 + params->c;
  t27 = sqrt(t26);
  t28 = 0.1e1 / t27;
  t31 = 0.1e1 / t26;
  r->f = 0.19997916265148655845e0 * t10 * t13 + 0.22565232098914243869e0 * t7 * t17 + 0.99989581325743279224e-1 * t1 * t21 * t23 + 0.39995832530297311691e0 * t10 * t28 + 0.22565232098914243869e0 * t7 * t31;

  if(r->order < 1) return;

  t34 = r->rs * r->rs;
  t35 = 0.1e1 / t34;
  t36 = t35 * t9;
  t39 = t35 * t6;
  t41 = t34 * r->rs;
  t42 = 0.1e1 / t41;
  t43 = t42 * t21;
  t45 = -0.22157981704254580414e1 * t39 + 0.49097615320608071993e1 * t43;
  t46 = t1 * t45;
  t53 = t45 * t16;
  t60 = 0.1e1 / t20 / t5;
  r->dfdrs = -0.19997916265148655845e0 * t36 * t13 + 0.19997916265148655845e0 * t46 * t13 - 0.22565232098914243869e0 * t39 * t17 + 0.50000000000000000004e0 * t43 * t17 + 0.22565232098914243869e0 * t7 * t53 - 0.99989581325743279224e-1 * t35 * t21 * t23 + 0.44311346272637900685e0 * t42 * t60 * t23 - 0.39995832530297311691e0 * t36 * t28 + 0.39995832530297311691e0 * t46 * t28 - 0.22565232098914243869e0 * t39 * t31 + 0.50000000000000000004e0 * t43 * t31;
  r->dfdz = 0.0e0;

  if(r->order < 2) return;

  t72 = t42 * t6;
  t75 = t42 * t9;
  t80 = t34 * t34;
  t81 = 0.1e1 / t80;
  t82 = t81 * t21;
  t88 = 0.1e1 / t80 / r->rs;
  t89 = t88 * t60;
  t95 = 0.44315963408509160828e1 * t72 - 0.19639046128243228797e2 * t82 + 0.21758081239931260892e2 * t89;
  t96 = t95 * t16;
  t105 = t35 * t45;
  t108 = t1 * t95;
  t114 = t20 * t20;
  t115 = 0.1e1 / t114;
  r->d2fdrs2 = 0.45130464197828487738e0 * t72 * t17 + 0.39995832530297311690e0 * t75 * t13 - 0.45130464197828487738e0 * t39 * t53 - 0.20000000000000000001e1 * t82 * t17 + 0.10000000000000000001e1 * t43 * t53 + 0.22157981704254580416e1 * t89 * t17 + 0.22565232098914243869e0 * t7 * t96 + 0.19997916265148655845e0 * t43 * t23 + 0.79991665060594623382e0 * t75 * t28 + 0.45130464197828487738e0 * t72 * t31 - 0.39995832530297311690e0 * t105 * t13 + 0.19997916265148655845e0 * t108 * t13 - 0.17724538509055160274e1 * t81 * t60 * t23 + 0.29455500000000000002e1 * t88 * t115 * t23 - 0.79991665060594623382e0 * t105 * t28 + 0.39995832530297311691e0 * t108 * t28 - 0.20000000000000000001e1 * t82 * t31 + 0.22157981704254580416e1 * t89 * t31;
  r->d2fdrsz = 0.0e0;
  r->d2fdz2 = 0.0e0;

  if(r->order < 3) return;

  t129 = t81 * t9;
  t132 = t81 * t6;
  t135 = t42 * t45;
  t142 = t88 * t21;
  t145 = t35 * t95;
  t151 = 0.1e1 / t80 / t34;
  t152 = t151 * t60;
  t155 = 0.1e1 / t80 / t41;
  t156 = t155 * t115;
  t158 = -0.13294789022552748248e2 * t132 + 0.88375707577094529586e2 * t142 - 0.19582273115938134803e3 * t152 + 0.14463454981022450832e3 * t156;
  t159 = t1 * t158;
  t174 = -0.59993748795445967535e0 * t82 * t23 - 0.23997499518178387015e1 * t129 * t28 - 0.13539139259348546321e1 * t132 * t31 + 0.11998749759089193507e1 * t135 * t13 + 0.79760423290748221233e1 * t89 * t23 + 0.23997499518178387014e1 * t135 * t28 + 0.90000000000000000005e1 * t142 * t31 - 0.59993748795445967535e0 * t145 * t13 + 0.19997916265148655845e0 * t159 * t13 - 0.26509950000000000002e2 * t151 * t115 * t23 + 0.26106977203586831737e2 * t155 / t114 / t5 * t23 - 0.11998749759089193507e1 * t145 * t28 + 0.39995832530297311691e0 * t159 * t28;
  t202 = -0.19942183533829122374e2 * t152 * t31 + 0.14729284596182421599e2 * t156 * t31 - 0.11998749759089193507e1 * t129 * t13 - 0.13539139259348546321e1 * t132 * t17 + 0.13539139259348546321e1 * t72 * t53 + 0.90000000000000000005e1 * t142 * t17 - 0.67695696296742731607e0 * t39 * t96 - 0.60000000000000000005e1 * t82 * t53 - 0.19942183533829122374e2 * t152 * t17 + 0.15000000000000000001e1 * t43 * t96 + 0.66473945112763741248e1 * t89 * t53 + 0.14729284596182421599e2 * t156 * t17 + 0.22565232098914243869e0 * t7 * t158 * t16;
  r->d3fdrs3 = t174 + t202;
  r->d3fdrs2z = 0.0e0;
  r->d3fdrsz2 = 0.0e0;
  r->d3fdz3 = 0.0e0;

  if(r->order < 4) return;


}

void 
xc_lda_c_2d_prm_func(const xc_func_type_cuda *p, xc_lda_work_t *r)
{
  if(p->nspin == XC_UNPOLARIZED)
    func0(p, r);
  else
    func1(p, r);
}

#define maple2c_order 3
#define maple2c_func  xc_lda_c_2d_prm_func
