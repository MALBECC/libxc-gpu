#include "util.h"

extern xc_func_info_type_cuda xc_func_info_hyb_gga_x_n12_sx;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_b97_1p;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_pbe_mol0;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_pbe_sol0;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_pbeb0;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_pbe_molb0;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_pbe50;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_b3pw91;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_b3lyp;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_b3p86;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_o3lyp;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_mpw1k;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_pbeh;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_b97;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_b97_1;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_b97_2;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_x3lyp;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_b1wc;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_b97_k;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_b97_3;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_mpw3pw;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_b1lyp;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_b1pw91;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_mpw1pw;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_mpw3lyp;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_sb98_1a;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_sb98_1b;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_sb98_1c;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_sb98_2a;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_sb98_2b;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_sb98_2c;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_x_sogga11_x;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_hse03;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_hse06;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_hjs_pbe;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_hjs_pbe_sol;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_hjs_b88;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_hjs_b97x;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_cam_b3lyp;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_tuned_cam_b3lyp;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_bhandh;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_bhandhlyp;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_mb3lyp_rc04;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_mpwlyp1m;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_revb3lyp;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_camy_blyp;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_pbe0_13;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_b3lyps;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_wb97;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_wb97x;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_lrc_wpbeh;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_wb97x_v;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_lcy_pbe;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_lcy_blyp;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_lc_vv10;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_camy_b3lyp;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_wb97x_d;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_hpbeint;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_lrc_wpbe;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_b3lyp5;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_edf2;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_cap0;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_lc_wpbe;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_hse12;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_hse12s;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_hse_sol;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_cam_qtp_01;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_mpw1lyp;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_mpw1pbe;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_kmlyp;
extern xc_func_info_type_cuda xc_func_info_hyb_gga_xc_b5050lyp;


const xc_func_info_type_cuda *xc_hyb_gga_known_funct[] = {
  &xc_func_info_hyb_gga_x_n12_sx,
  &xc_func_info_hyb_gga_xc_b97_1p,
  &xc_func_info_hyb_gga_xc_pbe_mol0,
  &xc_func_info_hyb_gga_xc_pbe_sol0,
  &xc_func_info_hyb_gga_xc_pbeb0,
  &xc_func_info_hyb_gga_xc_pbe_molb0,
  &xc_func_info_hyb_gga_xc_pbe50,
  &xc_func_info_hyb_gga_xc_b3pw91,
  &xc_func_info_hyb_gga_xc_b3lyp,
  &xc_func_info_hyb_gga_xc_b3p86,
  &xc_func_info_hyb_gga_xc_o3lyp,
  &xc_func_info_hyb_gga_xc_mpw1k,
  &xc_func_info_hyb_gga_xc_pbeh,
  &xc_func_info_hyb_gga_xc_b97,
  &xc_func_info_hyb_gga_xc_b97_1,
  &xc_func_info_hyb_gga_xc_b97_2,
  &xc_func_info_hyb_gga_xc_x3lyp,
  &xc_func_info_hyb_gga_xc_b1wc,
  &xc_func_info_hyb_gga_xc_b97_k,
  &xc_func_info_hyb_gga_xc_b97_3,
  &xc_func_info_hyb_gga_xc_mpw3pw,
  &xc_func_info_hyb_gga_xc_b1lyp,
  &xc_func_info_hyb_gga_xc_b1pw91,
  &xc_func_info_hyb_gga_xc_mpw1pw,
  &xc_func_info_hyb_gga_xc_mpw3lyp,
  &xc_func_info_hyb_gga_xc_sb98_1a,
  &xc_func_info_hyb_gga_xc_sb98_1b,
  &xc_func_info_hyb_gga_xc_sb98_1c,
  &xc_func_info_hyb_gga_xc_sb98_2a,
  &xc_func_info_hyb_gga_xc_sb98_2b,
  &xc_func_info_hyb_gga_xc_sb98_2c,
  &xc_func_info_hyb_gga_x_sogga11_x,
  &xc_func_info_hyb_gga_xc_hse03,
  &xc_func_info_hyb_gga_xc_hse06,
  &xc_func_info_hyb_gga_xc_hjs_pbe,
  &xc_func_info_hyb_gga_xc_hjs_pbe_sol,
  &xc_func_info_hyb_gga_xc_hjs_b88,
  &xc_func_info_hyb_gga_xc_hjs_b97x,
  &xc_func_info_hyb_gga_xc_cam_b3lyp,
  &xc_func_info_hyb_gga_xc_tuned_cam_b3lyp,
  &xc_func_info_hyb_gga_xc_bhandh,
  &xc_func_info_hyb_gga_xc_bhandhlyp,
  &xc_func_info_hyb_gga_xc_mb3lyp_rc04,
  &xc_func_info_hyb_gga_xc_mpwlyp1m,
  &xc_func_info_hyb_gga_xc_revb3lyp,
  &xc_func_info_hyb_gga_xc_camy_blyp,
  &xc_func_info_hyb_gga_xc_pbe0_13,
  &xc_func_info_hyb_gga_xc_b3lyps,
  &xc_func_info_hyb_gga_xc_wb97,
  &xc_func_info_hyb_gga_xc_wb97x,
  &xc_func_info_hyb_gga_xc_lrc_wpbeh,
  &xc_func_info_hyb_gga_xc_wb97x_v,
  &xc_func_info_hyb_gga_xc_lcy_pbe,
  &xc_func_info_hyb_gga_xc_lcy_blyp,
  &xc_func_info_hyb_gga_xc_lc_vv10,
  &xc_func_info_hyb_gga_xc_camy_b3lyp,
  &xc_func_info_hyb_gga_xc_wb97x_d,
  &xc_func_info_hyb_gga_xc_hpbeint,
  &xc_func_info_hyb_gga_xc_lrc_wpbe,
  &xc_func_info_hyb_gga_xc_b3lyp5,
  &xc_func_info_hyb_gga_xc_edf2,
  &xc_func_info_hyb_gga_xc_cap0,
  &xc_func_info_hyb_gga_xc_lc_wpbe,
  &xc_func_info_hyb_gga_xc_hse12,
  &xc_func_info_hyb_gga_xc_hse12s,
  &xc_func_info_hyb_gga_xc_hse_sol,
  &xc_func_info_hyb_gga_xc_cam_qtp_01,
  &xc_func_info_hyb_gga_xc_mpw1lyp,
  &xc_func_info_hyb_gga_xc_mpw1pbe,
  &xc_func_info_hyb_gga_xc_kmlyp,
  &xc_func_info_hyb_gga_xc_b5050lyp,
  NULL
};
