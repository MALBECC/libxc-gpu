## Process this file with automake to produce Makefile.in

## Copyright (C) 2002 M. Marques, A. Castro, A. Rubio, G. Bertsch
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.
##

SUBDIRS = maple2c helper

bin_PROGRAMS = xc-info xc-threshold
noinst_PROGRAMS = genwiki xc-sanity

AM_CFLAGS = $(NVCCFLAGS)
AM_LDFLAGS =
COMPILE = $(NVCC) -DHAVE_CONFIG_H -I. -I.. -I$(srcdir)/../src/ -I$(top_builddir)/src -g -O2  $(NVCCFLAGS) $(AM_LDFLAGS)
LTCOMPILE = $(NVCC) -DHAVE_CONFIG_H -I. -I.. $(NVCCFLAGS) $(AM_LDFLAGS)

xc_info_SOURCES = xc-info.c
xc_info_LDADD = libxc.a -lm
xc_info_LDFLAGS = $(NVCCFLAGS)

xc_threshold_SOURCES = xc-threshold.c
xc_threshold_LDADD = libxc.a -lm
xc_threshold_LDFLAGS = $(NVCCFLAGS)

genwiki_SOURCES = genwiki.c
genwiki_LDADD = libxc.a -lm
genwiki_LDFLAGS = $(NVCCFLAGS)

xc_sanity_SOURCES = xc-sanity.c
xc_sanity_LDADD = libxc.a -lm
xc_sanity_LDFLAGS =

lib_LTLIBRARIES = libxc.la

libxc_la_FUNC_SOURCES = \
	special_functions.c bessel.c expint_e1.c integrate.c util.c	\
	mix_func.c func_info.c functionals.c version.c func_reference.c \
        lda.c \
        lda_x.c lda_x_1d.c lda_x_2d.c lda_c_wigner.c lda_c_gombas.c	\
        lda_c_rpa.c lda_c_hl.c lda_c_vwn.c lda_c_vwn_1.c lda_c_vwn_2.c  \
        lda_c_vwn_3.c lda_c_vwn_4.c lda_c_vwn_rpa.c lda_c_pz.c          \
        lda_c_pw.c lda_c_pk09.c lda_c_lp96.c \
        lda_c_ml1.c lda_c_rc04.c lda_xc_teter93.c lda_c_1d_csc.c	\
        lda_c_1d_loos.c lda_c_2d_amgb.c lda_c_2d_prm.c		\
        lda_k_tf.c lda_xc_zlp.c lda_xc_ksdt.c lda_c_chachiyo.c          \
        lda_xc_1d_ehwlrg.c lda_x_rel.c lda_x_erf.c lda_k_zlp.c          \
        lda_c_gk72.c \
	gga.c \
	gga_x_lg93.c gga_x_pbe.cu gga_x_rpbe.c gga_x_pbea.c		\
	gga_x_mpbe.c gga_x_herman.c gga_x_b86.c gga_x_b88.c		\
	gga_x_g96.c gga_x_pw86.c gga_x_pw91.c gga_x_optx.c		\
	gga_x_airy.c gga_x_c09x.c gga_x_dk87.c gga_x_ft97.c		\
	gga_x_wc.c gga_x_am05.c gga_x_bayesian.c gga_x_kt.c		\
	gga_x_htbs.c gga_c_pbe.cu gga_c_lyp.c gga_c_p86.c gga_c_pw91.c	\
	gga_c_am05.c gga_c_lm.c gga_c_wl.c gga_c_wi.c gga_x_lb.c	\
	gga_xc_b97.c gga_xc_edf1.c gga_xc_1w.c gga_c_optc.c		\
	gga_c_tca.c gga_x_bpccac.c gga_x_sogga11.c gga_c_sogga11.c	\
	gga_xc_th1.c gga_x_wpbeh.c gga_x_hjs.c gga_x_ityh.c		\
	gga_x_sfat.c gga_x_ev93.c gga_x_ak13.c gga_x_q2d.c gga_c_q2d.c	\
	gga_x_ssb_sw.c gga_c_ft97.c gga_x_n12.c	gga_x_lv_rpw86.c        \
	gga_x_2d_b86.c gga_x_2d_b86_mgc.c		\
	gga_x_2d_b88.c gga_x_2d_pbe.c gga_k_tflw.c gga_k_pearson.c	\
	gga_k_ol1.c gga_k_ol2.c gga_k_dk.c gga_k_meyer.c gga_x_vmt.c	\
	gga_x_hcth_a.c gga_x_cap.c gga_x_beefvdw.c gga_x_sg4.c          \
	gga_x_gg99.c gga_x_pbepow.c gga_x_lag.c gga_k_thakkar.c         \
	gga_xc_oblyp_d.c hyb_gga_xc_b3lyp.c hyb_gga_xc_o3lyp.c		\
	gga_x_pbeint.c gga_x_rge2.c gga_x_vmt84.c gga_x_hjs_b88_v2.c    \
	gga_c_revtca.c gga_c_bcgp.c gga_c_zpbeint.c gga_c_pbeloc.c      \
	gga_c_sg4.c gga_c_regtpss.c gga_c_scan_e0.c gga_xc_th2.c        \
	gga_xc_th3.c gga_c_hcth_a.c hyb_gga_xc_wb97.c gga_c_bmk.c       \
	gga_c_gapc.c gga_c_gaploc.c gga_c_zvpbeint.c gga_c_w94.c        \
	gga_c_cs1.c gga_c_op_xalpha.c gga_c_op_b88.c gga_c_op_pbe.c     \
	gga_c_op_g96.c gga_c_op_pw91.c gga_x_pbetrans.c \
	hyb_gga_xc_pbeh.c hyb_gga_xc_b1wc.c 				\
	hyb_gga_xc_hse.c hyb_gga_xc_cam_b3lyp.c				\
	hyb_gga_xc_camy_blyp.c hyb_gga_xc_camy_b3lyp.c			\
	hyb_gga_xc_lcy_pbe.c hyb_gga_xc_lcy_blyp.c gga_xc_vv10.c        \
	hyb_gga_xc_edf2.c                                               \
	mgga.c                                                          \
	mgga_x_lta.c mgga_x_tpss.c mgga_x_br89.c mgga_x_gvt4.c	        \
	mgga_x_m06l.c mgga_x_tau_hcth.c mgga_x_2d_prhg07.c	        \
	mgga_x_pkzb.c hyb_mgga_x_m05.c mgga_x_m08.c mgga_x_ms.c	        \
	mgga_x_mn12.c mgga_x_mk00.c mgga_c_vsxc.c mgga_c_pkzb.c	        \
	mgga_c_bc95.c mgga_c_m08.c mgga_xc_cc06.c mgga_c_cs.c	        \
	mgga_xc_otpss_d.c mgga_xc_zlp.c hyb_mgga_xc_b88b95.c	        \
	mgga_x_mbeef.c mgga_xc_b97mv.c mgga_x_scan.c mgga_c_scan.c      \
	hyb_mgga_xc_tpssh.c mgga_x_mvs.c hyb_mgga_x_mvsh.c              \
	mgga_x_tm.c mgga_x_vt84.c mgga_x_sa_tpss.c mgga_k_pc07.c        \
	mgga_x_mbeefvdw.c mgga_x_m11.c mgga_x_m11_l.c                   \
        hyb_mgga_xc_wb97mv.c mgga_c_m05.c mgga_c_m06l.c                 \
        hyb_mgga_x_dldf.c mgga_c_tpss.c mgga_c_revtpss.c                \
        mgga_c_tpssloc.c mgga_xc_hle17.c mgga_c_kcis.c mgga_xc_lp90.c   \
	hyb_mgga_xc_kcis.c mgga_c_b88.c mgga_x_gx.c mgga_x_pbe_gx.c

#Truco para engañar al automake
am__objects_1 = \
	special_functions.o bessel.o expint_e1.o integrate.o util.o	\
	mix_func.o func_info.o functionals.o version.o func_reference.o \
        lda.o \
        lda_x.o lda_x_1d.o lda_x_2d.o lda_c_wigner.o lda_c_gombas.o	\
        lda_c_rpa.o lda_c_hl.o lda_c_vwn.o lda_c_vwn_1.o lda_c_vwn_2.o  \
        lda_c_vwn_3.o lda_c_vwn_4.o lda_c_vwn_rpa.o lda_c_pz.o          \
        lda_c_pw.o lda_c_pk09.o lda_c_lp96.o \
        lda_c_ml1.o lda_c_rc04.o lda_xc_teter93.o lda_c_1d_csc.o	\
        lda_c_1d_loos.o lda_c_2d_amgb.o lda_c_2d_prm.o		\
        lda_k_tf.o lda_xc_zlp.o lda_xc_ksdt.o lda_c_chachiyo.o          \
        lda_xc_1d_ehwlrg.o lda_x_rel.o lda_x_erf.o lda_k_zlp.o          \
        lda_c_gk72.o \
	gga.o \
	gga_x_lg93.o gga_x_pbe.o gga_x_rpbe.o gga_x_pbea.o		\
	gga_x_mpbe.o gga_x_herman.o gga_x_b86.o gga_x_b88.o		\
	gga_x_g96.o gga_x_pw86.o gga_x_pw91.o gga_x_optx.o		\
	gga_x_airy.o gga_x_c09x.o gga_x_dk87.o gga_x_ft97.o		\
	gga_x_wc.o gga_x_am05.o gga_x_bayesian.o gga_x_kt.o		\
	gga_x_htbs.o gga_c_pbe.o gga_c_lyp.o gga_c_p86.o gga_c_pw91.o	\
	gga_c_am05.o gga_c_lm.o gga_c_wl.o gga_c_wi.o gga_x_lb.o	\
	gga_xc_b97.o gga_xc_edf1.o gga_xc_1w.o gga_c_optc.o		\
	gga_c_tca.o gga_x_bpccac.o gga_x_sogga11.o gga_c_sogga11.o	\
	gga_xc_th1.o gga_x_wpbeh.o gga_x_hjs.o gga_x_ityh.o		\
	gga_x_sfat.o gga_x_ev93.o gga_x_ak13.o gga_x_q2d.o gga_c_q2d.o	\
	gga_x_ssb_sw.o gga_c_ft97.o gga_x_n12.o	gga_x_lv_rpw86.o        \
	gga_x_2d_b86.o gga_x_2d_b86_mgc.o		\
	gga_x_2d_b88.o gga_x_2d_pbe.o gga_k_tflw.o gga_k_pearson.o	\
	gga_k_ol1.o gga_k_ol2.o gga_k_dk.o gga_k_meyer.o gga_x_vmt.o	\
	gga_x_hcth_a.o gga_x_cap.o gga_x_beefvdw.o gga_x_sg4.o          \
	gga_x_gg99.o gga_x_pbepow.o gga_x_lag.o gga_k_thakkar.o         \
	gga_xc_oblyp_d.o hyb_gga_xc_b3lyp.o hyb_gga_xc_o3lyp.o		\
	gga_x_pbeint.o gga_x_rge2.o gga_x_vmt84.o gga_x_hjs_b88_v2.o    \
	gga_c_revtca.o gga_c_bcgp.o gga_c_zpbeint.o gga_c_pbeloc.o      \
	gga_c_sg4.o gga_c_regtpss.o gga_c_scan_e0.o gga_xc_th2.o        \
	gga_xc_th3.o gga_c_hcth_a.o hyb_gga_xc_wb97.o gga_c_bmk.o       \
	gga_c_gapc.o gga_c_gaploc.o gga_c_zvpbeint.o gga_c_w94.o        \
	gga_c_cs1.o gga_c_op_xalpha.o gga_c_op_b88.o gga_c_op_pbe.o     \
	gga_c_op_g96.o gga_c_op_pw91.o gga_x_pbetrans.o \
	hyb_gga_xc_pbeh.o hyb_gga_xc_b1wc.o 				\
	hyb_gga_xc_hse.o hyb_gga_xc_cam_b3lyp.o				\
	hyb_gga_xc_camy_blyp.o hyb_gga_xc_camy_b3lyp.o			\
	hyb_gga_xc_lcy_pbe.o hyb_gga_xc_lcy_blyp.o gga_xc_vv10.o        \
	hyb_gga_xc_edf2.o                                               \
	mgga.o                                                          \
	mgga_x_lta.o mgga_x_tpss.o mgga_x_br89.o mgga_x_gvt4.o	        \
	mgga_x_m06l.o mgga_x_tau_hcth.o mgga_x_2d_prhg07.o	        \
	mgga_x_pkzb.o hyb_mgga_x_m05.o mgga_x_m08.o mgga_x_ms.o	        \
	mgga_x_mn12.o mgga_x_mk00.o mgga_c_vsxc.o mgga_c_pkzb.o	        \
	mgga_c_bc95.o mgga_c_m08.o mgga_xc_cc06.o mgga_c_cs.o	        \
	mgga_xc_otpss_d.o mgga_xc_zlp.o hyb_mgga_xc_b88b95.o	        \
	mgga_x_mbeef.o mgga_xc_b97mv.o mgga_x_scan.o mgga_c_scan.o      \
	hyb_mgga_xc_tpssh.o mgga_x_mvs.o hyb_mgga_x_mvsh.o              \
	mgga_x_tm.o mgga_x_vt84.o mgga_x_sa_tpss.o mgga_k_pc07.o        \
	mgga_x_mbeefvdw.o mgga_x_m11.o mgga_x_m11_l.o                   \
        hyb_mgga_xc_wb97mv.o mgga_c_m05.o mgga_c_m06l.o                 \
        hyb_mgga_x_dldf.o mgga_c_tpss.o mgga_c_revtpss.o                \
        mgga_c_tpssloc.o mgga_xc_hle17.o mgga_c_kcis.o mgga_xc_lp90.o   \
	hyb_mgga_xc_kcis.o mgga_c_b88.o mgga_x_gx.o mgga_x_pbe_gx.o
am_libxc_la_OBJECTS = references.o $(am__objects_1)

libxc_la_SOURCES = references.c $(libxc_la_FUNC_SOURCES)
#libxc_la_LDFLAGS = -version-info $(XC_LT_VERSION)
#libxc_la_LDFLAGS = -lstdc++
#libxc_la_LINK = $(NVCC) $(NVCCFLAGS) $(AM_CFLAGS) $(CFLAGS) $(libxc_la_LDFLAGS) $(LDFLAGS) -o $@
libxc_la_LDFLAGS = 
libxc_la_LINK = $(NVCC) -lib -shared -o libxc.a *.o *.dlink.o
#libxc_la_LINK = gcc -shared -o libxc.so *.o *.dlink.o


noinst_HEADERS = \
	string_f.h references.h util.h work_lda.c \
	work_gga_x.cuh work_gga_c.cuh \
	work_mgga_x.c work_mgga_c.c \
	libxc_master.F90 libxc_master.F03
include_HEADERS = xc.h xc_funcs_removed.h
nodist_include_HEADERS = xc_funcs.h

BUILT_SOURCES = xc_funcs.h

CLEANFILES = *~ *.bak *.mod *.il *.d *.pc* ifc* funcs_*.c funcs.h \
	libxc.f90 libxc_funcs.f90 libxcf03.f90 libxc_inc.f03 xc_funcs.h xc_f_inc.c libxc_docs.txt \
	*.dlink.o *.a *.la

.c.o:
	$(AM_V_CC)$(NVCC) --compiler-options '-fPIC' -dc -g -O2 -DHAVE_CONFIG_H -I.. -I$(srcdir)/../src -I$(top_builddir)/src  -o $@ -c $<
	$(AM_V_CC)$(NVCC) --compiler-options '-fPIC' -dlink -o $@.dlink.o $@

.c.lo:
	$(AM_V_CC)$(NVCC) --compiler-options '-fPIC' -dc -g -O2 -DHAVE_CONFIG_H -I.. -I$(srcdir)/../src -I$(top_builddir)/src  -o $@ -c $<
	$(AM_V_CC)$(NVCC) --compiler-options '-fPIC' -dlink -o $@.dlink.o $@

.cuh.o:
	$(AM_V_CC)$(NVCC) --compiler-options '-fPIC' -dc $(NVCCFLAGS) -I$(srcdir)/../src -I$(top_builddir)/src  -o $@ -c $<
	$(AM_V_CC)$(NVCC) --compiler-options '-fPIC' $(NVCCFLAGS) -dlink -o $@.dlink.o $@

.cu.o:
	$(AM_V_CC)$(NVCC) --compiler-options '-fPIC' -dc $(NVCCFLAGS) -I.. -I$(srcdir)/../src -I$(top_builddir)/src  -o $@ -c $<
	$(AM_V_CC)$(NVCC) --compiler-options '-fPIC' -dlink $(NVCCFLAGS) -o $@.dlink.o $@

#Otro truco
libxc.la: $(libxc_la_OBJECTS) $(libxc_la_DEPENDENCIES) $(EXTRA_libxc_la_DEPENDENCIES)
	$(AM_V_GEN)$(libxc_la_LINK) $(libxc_la_OBJECTS) $(libxc_la_LIBADD) $(LIBS)

xc_f.o : xc_f_inc.c

xc_funcs.h libxc_inc.f03 libxc_funcs.f90 \
funcs_lda.c funcs_gga.c funcs_hyb_gga.c \
funcs_mgga.c funcs_hyb_mgga.c funcs_key.c: $(srcdir)/../scripts/get_funcs.pl
	$(srcdir)/../scripts/get_funcs.pl $(srcdir) $(top_builddir)/src

$(LIBFUNCMOD) $(LIBSFUNCMOD): $(LTPREF)libxc_funcs.lo

$(LTPREF)libxc.lo   $(LTPREF)libxc.o   : $(LIBFUNCMOD)

$(XCLIBMODS)   : $(LTPREF)libxc.lo
