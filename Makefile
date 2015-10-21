CC = gcc
AOC = aoc
NUM_TRAIN_USED = 30
NUM_PIXEL = 9
BOARD = p385_hpc_d5
HOST_TARGET = lcr
CLTARGET = src_GS/lcr_GS.cl
EXE_FILE = lcr_iter_30_9.aocx
EXE_FOLDER = lcr_iter_30_9

HOST_SOURCES = src_GS/main_iterative.c src_GS/load_pgm.c src_GS/util.c
CL_SOURCES = src_GS/cl_setup.c

HR_TRAIN = bin/trainHR/FaceHR_
LR_TRAIN = bin/trainLR/Face_
LR_INPUT = bin/testLR/FaceTest_
HR_OUTPUT = bin/Result/FaceOutput_
EXE_FILE_NAME = /home/wm813/FaceSuperResolution_c_opencl/lcr_iter_30_9.aocx

GSL_HEADERS = -I/mnt/applications/gsl/1.16/include
GSL_LDFLAGS = -L/mnt/applications/gsl/1.16/lib
GSL_LDLIBS = -lgsl -lgslcblas -lm

AOCL_COMPILE_CONFIG = -I/mnt/applications/altera/aocl-sdk/host/include -I/mnt/applications/altera/aocl-sdk/board/nalla_pcie/include #$(aocl compile-config)                        
AOCL_LDFLAGS = -L/mnt/applications/altera/aocl-sdk/board/nalla_pcie/linux64/lib -L/mnt/applications/altera/aocl-sdk/host/linux64/lib #$(aocl ldflags)                              
AOCL_LDLIBS = -lalteracl -ldl -lacl_emulator_kernel_rt  -lalterahalmmd -lnalla_pcie_mmd -lelf -lrt -lstdc++ #$({aocl ldlibs)                                                       


bin/$(HOST_TARGET) :
	mkdir -p bin
	$(CC) $(HOST_SOURCES) $(CL_SOURCES) -o $@ -DUSE_OPENCL $(GSL_HEADERS) $(GSL_LDFLAGS) $(GSL_LDLIBS) $(AOCL_COMPILE_CONFIG) $(AOCL_LDFLAGS) $(AOCL_LDLIBS)

run:
	bin/lcr $(HR_TRAIN) $(LR_TRAIN) $(LR_INPUT) $(HR_OUTPUT) $(EXE_FILE_NAME) 

kernel:
	$(AOC) -o $(EXE_FILE) -v $(CLTARGET) -D M=$(NUM_TRAIN_USED) -D N=$(NUM_PIXEL) --board $(BOARD)

simu_iter:
	$(AOC) -o $(EXE_FILE) -v -march=emulator $(CLTARGET) --board $(BOARD)

without_hw:
	$(AOC) -c -v -g $(CLTARGET) --board $(BOARD)

c_only:
	mkdir -p bin
	$(CC) $(HOST_SOURCES) -o bin/lcr_c $(GSL_HEADERS) $(GSL_LDFLAGS) $(GSL_LDLIBS)

cleanall:
	rm -rf bin/$(HOST_TARGET)
	rm -rf $(EXE_FOLDER)*

clean:
	rm -rf bin/$(HOST_TARGET)