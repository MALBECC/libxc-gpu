## Process this file with automake to produce Makefile.in

## Copyright (C) 2002 M. Marques, A. Castro, A. Rubio, G. Bertsch
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at http://mozilla.org/MPL/2.0/.
##

noinst_PROGRAMS = xc-get_data xc-consistency xc-regression xc-error template test001
#noinst_PROGRAMS =  template
dist_noinst_SCRIPTS = xc-run_testsuite xc-reference.pl
TESTS = xc-run_testsuite
TESTS_ENVIRONMENT = buildir=$(top_builddir)/testsuite

# Extra flags for cuda libraries and link options
AM_CFLAGS = $(NVCCFLAGS)
AM_LDFLAGS = -lstdc++ -L$(HOME)/workspace/common/libs/ -lxc -lm
AM_CPPFLAGS = -D_HAVE_CONFIG_H -I. -I.. -I$(srcdir)/../src/ -I$(top_buildir)/src -g -O2

#xc_get_data_SOURCES = xc-get_data.c xc-consistency.c xc_regression.c xc-error.c
#xc_get_data_LDADD = -L../src/ -lxc -lm
#xc_get_data_CPPFLAGS = -I$(srcdir)/../src/ -I$(top_builddir)/src

#xc_consistency_SOURCES = xc-consistency.c
#xc_consistency_LDADD = -L../src/ -lxc -lm
#xc_consistency_CPPFLAGS = -I$(srcdir)/../src/ -I$(top_builddir)/src

#xc_regression_SOURCES = xc-regression.c
#xc_regression_LDADD = -L../src/ -lxc -lm
#xc_regression_CPPFLAGS = -I$(srcdir)/../src/ -I$(top_builddir)/src

#xc_error_SOURCES = xc-error.c
#xc_error_LDADD = -lm
#xc_error_CPPFLAGS = -I$(srcdir)/../src/ -I$(top_builddir)/src

# CUDA files
template_SOURCES = template.cu
template_LDADD = $(CUDA_LIBS)
template_NVCCFLAGS = $(NVCCFLAGS) -I$(srcdir)/../src/ -I$(top_builddir)/src

dist_noinst_DATA = $(srcdir)/df_repo/*.data $(srcdir)/regression/*/*.bz2 $(srcdir)/input/*

# Delete temporary working directories from testsuite runs, if they are older than MINTIME (in minutes)
MINTIME=600
# note: beginning line with '-' means ignore the exit status of the command

.c.o:
	$(NVCC) -D_HAVE_CONFIG_H -I. -I.. -I$(srcdir)/../src/ -I$(top_buildir)/src -g -O2  -o $@ -c $<

.cu.o:
	$(NVCC) $(NVCCFLAGS) -o $@ -c $<

.cu.lo:
	$(NVCC) $(NVCCFLAGS) -o $@ -c $<

clean-local:
	-if test "x$$tmpdir" == "x"; then tmpdir=/tmp; fi; \
	find $$tmpdir -name 'libxc*' -type d -mmin +$(MINTIME) -exec rm -rf {} \; 2> /dev/null

CLEANFILES = *~ *.bak

.PHONY: clean clean-local
