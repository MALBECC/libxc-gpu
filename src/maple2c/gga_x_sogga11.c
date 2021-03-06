/* 
  This file was generated automatically with /nfs/data-012/marques/software/source/libxc/svn/scripts/maple2c.pl.
  Do not edit this file directly as it can be overwritten!!

  This Source Code Form is subject to the terms of the Mozilla Public
  License, v. 2.0. If a copy of the MPL was not distributed with this
  file, You can obtain one at http://mozilla.org/MPL/2.0/.

  Maple version     : Maple 2016 (X86 64 LINUX)
  Maple source      : ../maple/gga_x_sogga11.mpl
  Type of functional: work_gga_x
*/

void xc_gga_x_sogga11_enhance
  (const xc_func_type_cuda *p,  xc_gga_work_x_t *r)
{
  double t2, t3, t4, t5, t6, t7, t8, t9;
  double t10, t11, t14, t15, t17, t19, t20, t22;
  double t23, t25, t26, t28, t32, t33, t34, t36;
  double t37, t39, t40, t42, t43, t45, t48, t49;
  double t51, t52, t54, t57, t58, t62, t66, t70;
  double t75, t76, t77, t80, t84, t88, t92, t103;
  double t105, t106, t107, t109, t110, t111, t112, t114;
  double t117, t118, t120, t124, t125, t127, t131, t132;
  double t134, t138, t152, t164, t165, t166, t169, t170;
  double t171, t174, t177, t178, t181, t184, t185, t188;
  double t191, t192, t195, t196, t199, t202, t203, t206;
  double t209, t210, t213, t216, t218, t220, t221, t223;
  double t224, t225, t226, t230, t237, t247, t254, t259;
  double t263, t268, t274, t305, t310, t313, t321, t369;

  gga_x_sogga11_params *params;
 
  assert(p->params != NULL);
  params = (gga_x_sogga11_params * )(p->params);

  t2 = params->a[1];
  t3 = M_CBRT6;
  t4 = params->mu * t3;
  t5 = 0.31415926535897932385e1 * 0.31415926535897932385e1;
  t6 = cbrt(t5);
  t7 = t6 * t6;
  t8 = 0.1e1 / t7;
  t9 = 0.1e1 / params->kappa;
  t10 = t8 * t9;
  t11 = r->x * r->x;
  t14 = t4 * t10 * t11 / 0.24e2;
  t15 = 0.1e1 + t14;
  t17 = 0.1e1 - 0.1e1 / t15;
  t19 = params->a[2];
  t20 = t17 * t17;
  t22 = params->a[3];
  t23 = t20 * t17;
  t25 = params->a[4];
  t26 = t20 * t20;
  t28 = params->a[5];
  t32 = params->b[1];
  t33 = exp(-t14);
  t34 = 0.1e1 - t33;
  t36 = params->b[2];
  t37 = t34 * t34;
  t39 = params->b[3];
  t40 = t37 * t34;
  t42 = params->b[4];
  t43 = t37 * t37;
  t45 = params->b[5];
  r->f = t28 * t26 * t17 + t45 * t43 * t34 + t2 * t17 + t19 * t20 + t22 * t23 + t25 * t26 + t32 * t34 + t36 * t37 + t39 * t40 + t42 * t43 + params->a[0] + params->b[0];

  if(r->order < 1) return;

  t48 = t15 * t15;
  t49 = 0.1e1 / t48;
  t51 = t2 * t49 * params->mu;
  t52 = t3 * t8;
  t54 = t52 * t9 * r->x;
  t57 = t19 * t17;
  t58 = t49 * params->mu;
  t62 = t22 * t20;
  t66 = t25 * t23;
  t70 = t28 * t26;
  t75 = t32 * params->mu * t3;
  t76 = r->x * t33;
  t77 = t10 * t76;
  t80 = t36 * t34;
  t84 = t39 * t37;
  t88 = t42 * t40;
  t92 = t45 * t43;
  r->dfdx = t51 * t54 / 0.12e2 + t57 * t58 * t54 / 0.6e1 + t62 * t58 * t54 / 0.4e1 + t66 * t58 * t54 / 0.3e1 + 0.5e1 / 0.12e2 * t70 * t58 * t54 + t75 * t77 / 0.12e2 + t80 * t4 * t77 / 0.6e1 + t84 * t4 * t77 / 0.4e1 + t88 * t4 * t77 / 0.3e1 + 0.5e1 / 0.12e2 * t92 * t4 * t77;

  if(r->order < 2) return;

  t103 = 0.1e1 / t48 / t15;
  t105 = params->mu * params->mu;
  t106 = t2 * t103 * t105;
  t107 = t3 * t3;
  t109 = 0.1e1 / t6 / t5;
  t110 = t107 * t109;
  t111 = params->kappa * params->kappa;
  t112 = 0.1e1 / t111;
  t114 = t110 * t112 * t11;
  t117 = t48 * t48;
  t118 = 0.1e1 / t117;
  t120 = t19 * t118 * t105;
  t124 = t32 * t105 * t107;
  t125 = t109 * t112;
  t127 = t125 * t11 * t33;
  t131 = t36 * t105 * t107;
  t132 = t33 * t33;
  t134 = t125 * t11 * t132;
  t138 = t4 * t10;
  t152 = t52 * t9 * t33;
  t164 = t51 * t52 * t9 / 0.12e2 + t75 * t10 * t33 / 0.12e2 - t106 * t114 / 0.72e2 + t120 * t114 / 0.72e2 - t124 * t127 / 0.144e3 + t131 * t134 / 0.72e2 + t57 * t49 * t138 / 0.6e1 + t62 * t49 * t138 / 0.4e1 + t66 * t49 * t138 / 0.3e1 + 0.5e1 / 0.12e2 * t70 * t49 * t138 + t80 * params->mu * t152 / 0.6e1 + t84 * params->mu * t152 / 0.4e1 + t88 * params->mu * t152 / 0.3e1 + 0.5e1 / 0.12e2 * t92 * params->mu * t152;
  t165 = t103 * t105;
  t166 = t57 * t165;
  t169 = t22 * t17;
  t170 = t118 * t105;
  t171 = t169 * t170;
  t174 = t62 * t165;
  t177 = t25 * t20;
  t178 = t177 * t170;
  t181 = t66 * t165;
  t184 = t28 * t23;
  t185 = t184 * t170;
  t188 = t70 * t165;
  t191 = t105 * t107;
  t192 = t80 * t191;
  t195 = t39 * t34;
  t196 = t195 * t191;
  t199 = t84 * t191;
  t202 = t42 * t37;
  t203 = t202 * t191;
  t206 = t88 * t191;
  t209 = t45 * t40;
  t210 = t209 * t191;
  t213 = t92 * t191;
  t216 = -t166 * t114 / 0.36e2 + t171 * t114 / 0.24e2 - t174 * t114 / 0.24e2 + t178 * t114 / 0.12e2 - t181 * t114 / 0.18e2 + 0.5e1 / 0.36e2 * t185 * t114 - 0.5e1 / 0.72e2 * t188 * t114 - t192 * t127 / 0.72e2 + t196 * t134 / 0.24e2 - t199 * t127 / 0.48e2 + t203 * t134 / 0.12e2 - t206 * t127 / 0.36e2 + 0.5e1 / 0.36e2 * t210 * t134 - 0.5e1 / 0.144e3 * t213 * t127;
  r->d2fdx2 = t164 + t216;

  if(r->order < 3) return;

  t218 = t105 * params->mu;
  t220 = t5 * t5;
  t221 = 0.1e1 / t220;
  t223 = 0.1e1 / t111 / params->kappa;
  t224 = t221 * t223;
  t225 = t11 * r->x;
  t226 = t224 * t225;
  t230 = 0.1e1 / t117 / t15;
  t237 = t223 * t225;
  t247 = 0.1e1 / t117 / t48;
  t254 = t132 * t33;
  t259 = t110 * t112 * r->x;
  t263 = t125 * r->x * t132;
  t268 = t125 * t76;
  t274 = t218 * t221 * t237;
  t305 = t224 * t225 * t33;
  t310 = t224 * t225 * t132;
  t313 = t57 * t118 * t274 / 0.24e2 - t169 * t230 * t274 / 0.8e1 + t62 * t118 * t274 / 0.16e2 + t25 * t17 * t247 * t274 / 0.12e2 - t177 * t230 * t274 / 0.4e1 + t66 * t118 * t274 / 0.12e2 + 0.5e1 / 0.24e2 * t28 * t20 * t247 * t274 - 0.5e1 / 0.12e2 * t184 * t230 * t274 + 0.5e1 / 0.48e2 * t70 * t118 * t274 + t80 * t218 * t305 / 0.144e3 - t195 * t218 * t310 / 0.16e2;
  t321 = t224 * t225 * t254;
  t369 = t178 * t259 / 0.4e1 - t181 * t259 / 0.6e1 + 0.5e1 / 0.12e2 * t185 * t259 - 0.5e1 / 0.24e2 * t188 * t259 - t192 * t268 / 0.24e2 + t196 * t263 / 0.8e1 - t199 * t268 / 0.16e2 + t203 * t263 / 0.4e1 - t206 * t268 / 0.12e2 + 0.5e1 / 0.12e2 * t210 * t263 - 0.5e1 / 0.48e2 * t213 * t268;
  r->d3fdx3 = t2 * t118 * t218 * t226 / 0.48e2 - t19 * t230 * t218 * t226 / 0.24e2 + t32 * t218 * t221 * t237 * t33 / 0.288e3 - t36 * t218 * t221 * t237 * t132 / 0.48e2 + t22 * t247 * t218 * t226 / 0.48e2 + t39 * t218 * t221 * t237 * t254 / 0.48e2 + t120 * t259 / 0.24e2 + t131 * t263 / 0.24e2 - t106 * t259 / 0.24e2 - t124 * t268 / 0.48e2 + t313 + t84 * t218 * t305 / 0.96e2 + t42 * t34 * t218 * t321 / 0.12e2 - t202 * t218 * t310 / 0.8e1 + t88 * t218 * t305 / 0.72e2 + 0.5e1 / 0.24e2 * t45 * t37 * t218 * t321 - 0.5e1 / 0.24e2 * t209 * t218 * t310 + 0.5e1 / 0.288e3 * t92 * t218 * t305 - t166 * t259 / 0.12e2 + t171 * t259 / 0.8e1 - t174 * t259 / 0.8e1 + t369;

  if(r->order < 4) return;


}

#define maple2c_order 3
#define maple2c_func  xc_gga_x_sogga11_enhance
