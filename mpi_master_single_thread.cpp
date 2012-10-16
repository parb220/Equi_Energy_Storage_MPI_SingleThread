#include <fstream>
#include <iostream>
#include <cstring>
#include <sstream>
#include <gsl/gsl_rng.h>
#include <pthread.h>
#include <mpi.h>
#include "CModel.h"
#include "CEES_Node.h"
#include "CStorageHead.h"
#include "CParameterPackage.h"
#include "mpi_parameter.h"
#include "CSampleIDWeight.h"

using namespace std;


void DispatchBurnInTask(int nClusterNode, const CParameterPackage &parameter); 
void DispatchTuningTask(int nClusterNode, CParameterPackage &parameter);
void DispatchTrackingTask_LevelByLevel(int nClusterNode, CParameterPackage &parameter, CStorageHead &storage); 
void DispatchSimulation_LevelByLevel(int nClusterNode, const CParameterPackage &parameter, int highest_level, CStorageHead &storage);

void master_single_thread(string storage_filename_base, CStorageHead &storage, CParameterPackage &parameter, int highest_level, bool if_resume, CModel *target, const gsl_rng *r) 
{	
	/* Find out how many processes there are in the default communicator */
	int nTasks, my_rank;   
	MPI_Comm_size(MPI_COMM_WORLD, &nTasks); 
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); 

	if(!if_resume)
	{
		storage.makedir(); 	
		
		cout << "Initialize, burn in, tune/estimate MH stepsize" << endl; 
		DispatchBurnInTask(nTasks, parameter); 
		DispatchTuningTask(nTasks, parameter); 
		
		// scale for proposl distribution should be updated
		for (int nEnergyLevelTuning=0; nEnergyLevelTuning < parameter.energy_level_tuning_max_time; nEnergyLevelTuning ++)
		{
			cout << "Energy level tuning: " << nEnergyLevelTuning << endl; ;
			DispatchTrackingTask_LevelByLevel(nTasks, parameter, storage); 
			// h0 and hk_1 should be updated
			DispatchTuningTask(nTasks, parameter);
			// scale for proposal distribution should be updated 
		}
		// Save parameter
		parameter.number_cluster_node = nTasks; 
		parameter.TraceStorageHead(storage);
		stringstream convert; 
		convert.str(string());
		convert << parameter.run_id << "/" << parameter.run_id << ".parameter"; 
		string file_name = storage_filename_base + convert.str(); 
		parameter.SaveParameterToFile(file_name); 
	}
	else 
		storage.consolidate(); 

	cout << "Simulation (level by level) for " << parameter.simulation_length << " steps.\n"; 
	
	/* run simulation*/
	if (parameter.simulation_length > 0)
		DispatchSimulation_LevelByLevel(nTasks, parameter, highest_level, storage); 	
		 
	cout << "Done simulation" << endl; 

	// Save summary file 
	parameter.number_cluster_node = nTasks;
	parameter.TraceStorageHead(storage);
	stringstream convert; 
	convert.str(string()); 
	convert << parameter.run_id << "/" << parameter.run_id << ".summary";
 	string file_name = storage_filename_base + convert.str(); 
        parameter.WriteSummaryFile(file_name);
	
	// tell all the slaves to exit by sending an empty messag with 0 simulation length 
	double *sMessage= new double [parameter.GetMHProposalScaleSize()+parameter.data_dimension+N_MESSAGE];  
	for (int rank=1; rank<nTasks; rank++)
		MPI_Send(sMessage, parameter.GetMHProposalScaleSize()+parameter.data_dimension+N_MESSAGE, MPI_DOUBLE, rank, END_TAG, MPI_COMM_WORLD);
	delete [] sMessage; 
}

