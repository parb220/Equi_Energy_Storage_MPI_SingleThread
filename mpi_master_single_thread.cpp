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
#include "CSampleIDWeight.h"

using namespace std;

void RunSimulation(CParameterPackage &, int, int, CModel*, CStorageHead &, const gsl_rng *); 

void DispatchBurnInTask(int nClusterNode, const CParameterPackage &parameter); 
void DispatchTuningTask(int nClusterNode, CParameterPackage &parameter);
void DispatchTrackingTask(int nClusterNode, CParameterPackage &parameter); 
void DispatchSimulation(int nClusterNode, const CParameterPackage &parameter, int highest_level); 
 
void master_single_thread(string storage_filename_base, CStorageHead &storage, CParameterPackage &parameter, int highest_level, bool if_resume, CModel *target, const gsl_rng *r) 
{	
	/* Find out how many processes there are in the default communicator */
	int nTasks, my_rank;   
	MPI_Comm_size(MPI_COMM_WORLD, &nTasks); 
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); 

	/* WORKTAG: -1: exit; 0: tuning; 1: tracking; 2: simulation */
	if(!if_resume)
	{
		storage.makedir(); 	
		/*Tuning; 
		Loop for some times
		{
			TracingMaxMinEnergyLevel; 
			Tuning; 
		}*/
		cout << "Initialize, burn in, tune/estimate MH stepsize" << endl; 
		DispatchBurnInTask(nTasks, parameter); 
		DispatchTuningTask(nTasks, parameter); 
		// scale for proposl distribution should be updated
		int nEnergyLevelTuning = 0;
		while (nEnergyLevelTuning < parameter.energy_level_tuning_max_time)
		{
			cout << "Energy level tuning: " << nEnergyLevelTuning << " for " << parameter.energy_level_tracking_window_length << " steps.\n";
			DispatchTrackingTask(nTasks, parameter); 
			// h0 and hk_1 should be updated
			DispatchTuningTask(nTasks, parameter);
			// scale for proposal distribution should be updated 
			nEnergyLevelTuning ++;
		}
		// Disregard samples generated during tuning and tracking
		storage.DisregardHistorySamples(); 
		// Save parameter
		parameter.number_cluster_node = nTasks; 
		parameter.TraceStorageHead(storage);
		stringstream convert; 
		convert.str(string());
		convert << parameter.run_id << "/" << parameter.run_id << ".parameter"; 
		string file_name = storage_filename_base + convert.str(); 
		parameter.SaveParameterToFile(file_name); 
	}

/*	int nX=0; 
	while (nX == 0)
		nX = 0; */

	storage.consolidate(); 
	
	// run simulation
	cout << "Simulation for " << parameter.simulation_length << " steps.\n"; 
	int total_length = parameter.simulation_length; 
	int segment_length = 10000; 
	int nSegment = total_length/segment_length; 
	for (int i=0; i<nSegment; i++)
	{
		parameter.simulation_length =segment_length; 
		DispatchSimulation(nTasks, parameter, highest_level); 
		storage.consolidate(); 
	}
	int rSegment = total_length%segment_length; 
	if (rSegment)
	{
		parameter.simulation_length = rSegment; 
		DispatchSimulation(nTasks, parameter, highest_level);
		storage.consolidate(); 
	}
		 
	cout << "Done simulation" << endl; 

	/*int xloop = 0; 
	while (xloop == 0)
		xloop = 0; */

	// Save summary file 
	parameter.number_cluster_node = nTasks;
	parameter.TraceStorageHead(storage);
	stringstream convert; 
	convert.str(string()); 
	convert << parameter.run_id << "/" << parameter.run_id << ".summary";
 	string file_name = storage_filename_base + convert.str(); 
        parameter.WriteSummaryFile(file_name);
	
	// tell all the slaves to exit by sending an empty messag with 0 simulation length 
	double *sMessage= new double [parameter.GetMHProposalScaleSize()+4];  
	for (int rank=1; rank<nTasks; rank++)
		MPI_Send(sMessage, parameter.GetMHProposalScaleSize()+4, MPI_DOUBLE, rank, 0, MPI_COMM_WORLD);
	delete [] sMessage; 
}

