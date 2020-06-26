################################################################################
# UNIT TEST MAKEFILE
################################################################################

# Test result
result:=1

all: work_gga_c work_gga_x work_lda work_mgga_c work_mgga_x test_gpu

.PHONY: work_gga_c
work_gga_c:
	$(MAKE) -C work_gga_c

.PHONY: work_gga_x
work_gga_x:
	$(MAKE) -C work_gga_x

.PHONY: work_lda
work_lda:
	$(MAKE) -C work_lda

.PHONY: work_mgga_c
work_mgga_c:
	$(MAKE) -C work_mgga_c

.PHONY: work_mgga_x
work_mgga_x:
	$(MAKE) -C work_mgga_x


.PHONY: test_gpu
test_gpu:
	@echo "Running gpu test..."
	$(eval result:=$(shell work_gga_c/test_work_gga_c > /dev/null 2>&1 ; echo $$?))
	@echo "work_gga_c test: " $(result)
	$(eval result:=$(shell work_gga_x/test_work_gga_x > /dev/null 2>&1 ; echo $$?))
	@echo "work_gga_x test: " $(result)
	$(eval result:=$(shell work_lda/test_functionals > /dev/null 2>&1 ; echo $$?))
	@echo "work_lda test: " $(result)
	$(eval result:=$(shell work_mgga_c/test_functionals > /dev/null 2>&1 ; echo $$?))
	@echo "work_mgga_c test: " $(result)
	$(eval result:=$(shell work_mgga_x/test_functionals > /dev/null 2>&1 ; echo $$?))
	@echo "work_mgga_x test: " $(result)

.PHONY: check_test
check_test:
# Check test results
ifeq ($(a),0)
	@echo "Test ok"
else
	@echo "Test fail"
endif


.PHONY: clean
clean:
	$(MAKE) clean -C work_gga_c
	$(MAKE) clean -C work_gga_x
	$(MAKE) clean -C work_lda
	$(MAKE) clean -C work_mgga_c
	$(MAKE) clean -C work_mgga_x


################################################################################
