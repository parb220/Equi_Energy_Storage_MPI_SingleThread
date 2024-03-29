MPICXX = mpicc 
CFLAGS := $(CFLAGS) -g -Wall 

EQUAL_ENERGY_HOME = /home/f1hxw01/equal_energy_hw
INCLUDE_DIR := $(INCLUDE_DIR) -I$(EQUAL_ENERGY_HOME)/include
#LIBS := $(LIBS) -static /usr/lib/gcc/x86_64-redhat-linux/4.6.3/libstdc++.a -lgsl -lgslcblas -lm
LIBS := $(LIBS) -lpthread -lstdc++ -lgsl -lgslcblas -lm 
#LIBS_DIR := $(LIBS_DIR) -L/usr/lib64

DISTR_MODEL_DIR = $(EQUAL_ENERGY_HOME)/equi_energy_generic
DISTR_MODEL_OBJS = $(DISTR_MODEL_DIR)/CMixtureModel.o $(DISTR_MODEL_DIR)/CModel.o $(DISTR_MODEL_DIR)/CSimpleGaussianModel.o $(DISTR_MODEL_DIR)/CTransitionModel_SimpleGaussian.o $(DISTR_MODEL_DIR)/CUniformModel.o $(DISTR_MODEL_DIR)/CBoundedModel.o $(DISTR_MODEL_DIR)/AddScaledLogs.o 

SINGLE_CORE_VERSION_DIR = $(EQUAL_ENERGY_HOME)/equi_energy_storage
SINGLE_CORE_VERSION_OBJS = $(SINGLE_CORE_VERSION_DIR)/CEES_Node.o $(SINGLE_CORE_VERSION_DIR)/CPutGetBin.o $(SINGLE_CORE_VERSION_DIR)/CSampleIDWeight.o $(SINGLE_CORE_VERSION_DIR)/CStorageHead.o $(SINGLE_CORE_VERSION_DIR)/MHAdaptive.o $(SINGLE_CORE_VERSION_DIR)/CParameterPackage.o

all:  test_GMM_mpi

test_GMM_mpi_obj = test_GMM_mpi.o mpi_master_single_thread.o  mpi_slave_single_thread.o DispatchBurnInTask.o DispatchTuningTask.o DispatchSimulationTask_LevelByLevel.o DispatchTrackingTask_LevelByLevel.o

test_GMM_mpi: $(test_GMM_mpi_obj) $(SINGLE_CORE_VERSION_OBJS) $(DISTR_MODEL_OBJS)
	$(MPICXX) -o $@ $^ $(CFLAGS) $(LIBS) 

%.o : %.cpp 
	$(MPICXX) -o $@ -c $< $(CFLAGS) $(INCLUDE_DIR)

#test_GMM_mpi.o: test_GMM_mpi.cpp 
#	$(MPICXX) $(CFLAGS) $(INCLUDE_DIR) -c test_GMM_mpi.cpp 

#mpi_master_single_thread.o: mpi_master_single_thread.cpp mpi_parameter.h
#	$(MPICXX) $(CFLAGS) $(INCLUDE_DIR) -c mpi_master_single_thread.cpp

#mpi_slave_single_thread.o: mpi_slave_single_thread.cpp mpi_parameter.h
#	$(MPICXX) $(CFLAGS) $(INCLUDE_DIR) -c mpi_slave_single_thread.cpp

#DispatchBurnInTask.o: DispatchBurnInTask.cpp mpi_parameter.h
#	$(MPICXX) $(CFLAGS) $(INCLUDE_DIR) -c DispatchBurnInTask.cpp

#DispatchTuningTask.o: DispatchTuningTask.cpp mpi_parameter.h
#	$(MPICXX) $(CFLAGS) $(INCLUDE_DIR) -c DispatchTuningTask.cpp

#DispatchSimulationTask_LevelByLevel.o: DispatchSimulationTask_LevelByLevel.cpp mpi_parameter.h
#	$(MPICXX) $(CFLAGS) $(INCLUDE_DIR) -c DispatchSimulationTask_LevelByLevel.cpp

#DispatchTrackingTask_LevelByLevel.o: DispatchTrackingTask_LevelByLevel.cpp mpi_parameter.h
#	$(MPICXX) $(CFLAGS) $(INCLUDE_DIR) -c DispatchTrackingTask_LevelByLevel.cpp

clean: 
	rm -f *.o  test_GMM_mpi 
