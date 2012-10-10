MPICXX = mpicc 
CFLAGS := $(CFLAGS) -g -Wall 
LIBS := $(LIBS) -lpthread -lstdc++
LIBS_DIR := $(LIBS_DIR) -L/usr/lib64

EQUAL_ENERGY_HOME = /home/f1hxw01/equal_energy_hw
INCLUDE_DIR := $(INCLUDE_DIR) -I$(EQUAL_ENERGY_HOME)/include
LIBS := $(LIBS) -lgsl -lgslcblas -lm

DISTR_MODEL_DIR = $(EQUAL_ENERGY_HOME)/equi_energy_generic
DISTR_MODEL_OBJS = $(DISTR_MODEL_DIR)/CMixtureModel.o $(DISTR_MODEL_DIR)/CModel.o $(DISTR_MODEL_DIR)/CSimpleGaussianModel.o $(DISTR_MODEL_DIR)/CTransitionModel_SimpleGaussian.o $(DISTR_MODEL_DIR)/CUniformModel.o $(DISTR_MODEL_DIR)/CBoundedModel.o $(DISTR_MODEL_DIR)/AddScaledLogs.o 

SINGLE_CORE_VERSION_DIR = $(EQUAL_ENERGY_HOME)/equi_energy_storage
SINGLE_CORE_VERSION_OBJS = $(SINGLE_CORE_VERSION_DIR)/CEES_Node.o $(SINGLE_CORE_VERSION_DIR)/CPutGetBin.o $(SINGLE_CORE_VERSION_DIR)/CSampleIDWeight.o $(SINGLE_CORE_VERSION_DIR)/CStorageHead.o $(SINGLE_CORE_VERSION_DIR)/MHAdaptive.o $(SINGLE_CORE_VERSION_DIR)/CParameterPackage.o

all:  test_GMM_mpi

test_GMM_mpi_obj = test_GMM_mpi.o mpi_master_single_thread.o  mpi_slave_single_thread.o DispatchBurnInTask.o DispatchTuningTask.o DispatchTrackingTask.o DispatchSimulationTask_LevelByLevel.o DispatchTrackingTask_LevelByLevel.o $(SINGLE_CORE_VERSION_OBJS) $(DISTR_MODEL_OBJS)

test_GMM_mpi: $(test_GMM_mpi_obj)
	$(MPICXX) $(CFLAGS) $(LIBS_DIR) $(LIBS) $(test_GMM_mpi_obj) -o test_GMM_mpi 

test_GMM_mpi.o: test_GMM_mpi.cpp
	$(MPICXX) $(CFLAGS) $(INCLUDE_DIR) -c test_GMM_mpi.cpp 

mpi_master_single_thread.o: mpi_master_single_thread.cpp
	$(MPICXX) $(CFLAGS) $(INCLUDE_DIR) -c mpi_master_single_thread.cpp

mpi_slave_single_thread.o: mpi_slave_single_thread.cpp
	$(MPICXX) $(CFLAGS) $(INCLUDE_DIR) -c mpi_slave_single_thread.cpp

DispatchBurnInTask.o: DispatchBurnInTask.cpp
	$(MPICXX) $(CFLAGS) $(INCLUDE_DIR) -c DispatchBurnInTask.cpp

DispatchTuningTask.o: DispatchTuningTask.cpp
	$(MPICXX) $(CFLAGS) $(INCLUDE_DIR) -c DispatchTuningTask.cpp

DispatchSimulationTask_LevelByLevel.o: DispatchSimulationTask_LevelByLevel.cpp
	$(MPICXX) $(CFLAGS) $(INCLUDE_DIR) -c DispatchSimulationTask_LevelByLevel.cpp

DispatchTrackingTask.o: DispatchTrackingTask.cpp
	$(MPICXX) $(CFLAGS) $(INCLUDE_DIR) -c DispatchTrackingTask.cpp

DispatchTrackingTask_LevelByLevel.o: DispatchTrackingTask_LevelByLevel.cpp
	$(MPICXX) $(CFLAGS) $(INCLUDE_DIR) -c DispatchTrackingTask_LevelByLevel.cpp


clean: 
	rm -f *.o  test_GMM_mpi 
