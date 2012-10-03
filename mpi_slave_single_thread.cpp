#include <fstream>
#include <iostream>
#include <cstring>
#include <sstream>
#include <cmath>
#include <gsl/gsl_rng.h>
#include <pthread.h>
#include <mpi.h>
#include "CTransitionModel_SimpleGaussian.h"
#include "CMixtureModel.h"
#include "CEES_Node.h"
#include "CStorageHead.h"
#include "CParameterPackage.h"

using namespace std;

void WrapUpSimulation(CParameterPackage &parameter, const CEES_Node &simulator, string filename_base, int energy_level); 

void InitializeSimulator(CEES_Node &, int energy_level, CParameterPackage &parameter, string filename_base, CStorageHead &storage, const gsl_rng *r, CModel *target);

void ParseReceivedMessage(double *, int, CParameterPackage&, int&, int&); 

CEES_Node *GenerateSimulator(int energy_level, string filename_base, CStorageHead &storage, const CParameterPackage &parameter, CModel *target, const gsl_rng *r);

void slave_single_thread(string filename_base, CStorageHead &storage, CParameterPackage &parameter, int highest_level, bool if_resume, CModel *target, const gsl_rng* r)
{
	int my_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);  
	MPI_Status status; 

	double *rPackage = new double [parameter.GetMHProposalScaleSize()+4]; 
	double *sPackage = new double [parameter.GetMHProposalScaleSize()+1]; 
	
	/*int x_break=0, counter=0; 
	while (x_break == 0)
		counter++;*/

	int energy_level, simulation_length; 
	while (1)
	{
		MPI_Recv(rPackage, parameter.GetMHProposalScaleSize()+4, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status); 
		if (status.MPI_TAG == 1)
		{ // tuning scale	
			ParseReceivedMessage(rPackage, parameter.GetMHProposalScaleSize()+4, parameter, energy_level, simulation_length); 

			CEES_Node *simulator = GenerateSimulator(energy_level, filename_base, storage, parameter, target, r); 
			
			// Always start with mode for tuning scales
			CSampleIDWeight mode; 
			target->GetMode(mode, 0); 
			simulator[energy_level].Initialize(mode); 	
		
			// Tuning
			cout << energy_level << " ... Burn In" << endl; 
			simulator[energy_level].BurnIn(r, storage, parameter.burn_in_period, parameter.multiple_try_mh); 
			cout << energy_level << " ... Tune/Estimate MH Proposal Scales" << endl; 
			simulator[energy_level].MH_StepSize_Tune(parameter.mh_tracking_length, parameter.mh_stepsize_tuning_max_time, r, parameter.multiple_try_mh);  
			
			WrapUpSimulation(parameter, simulator[energy_level], filename_base, energy_level);
			// Send back messages
			sPackage[0] = (double)(energy_level);
			parameter.GetMHProposalScale(energy_level, sPackage+1, parameter.GetMHProposalScaleSize());
			delete [] simulator; 
			MPI_Send(sPackage, parameter.GetMHProposalScaleSize()+1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD); 
		}
		else if (status.MPI_TAG == 2)
		{ // tracking
			storage.restore(); 
			ParseReceivedMessage(rPackage, parameter.GetMHProposalScaleSize()+4, parameter, energy_level, simulation_length);
                        
			CEES_Node *simulator = GenerateSimulator(energy_level, filename_base, storage, parameter, target, r);
		
			InitializeSimulator(simulator[energy_level], energy_level, parameter, filename_base, storage, r, target); 

			cout << energy_level << " ... Simulating for ... " << simulation_length << " steps.\n"; 	
			// Simulation
			simulator[energy_level].Simulate(r, storage, simulation_length, parameter.deposit_frequency, parameter.multiple_try_mh);

			storage.finalize(); 
			WrapUpSimulation(parameter, simulator[energy_level], filename_base, energy_level); 
			
			sPackage[0] = simulator[energy_level].GetMinEnergy(0) < parameter.h0 ? simulator[energy_level].GetMinEnergy(0) : parameter.h0; 
			sPackage[1] = simulator[energy_level].GetMaxEnergy(0)< parameter.hk_1 ? simulator[energy_level].GetMaxEnergy(0) : parameter.hk_1; 
			delete [] simulator; 
			MPI_Send(sPackage, 2, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD); 
		}
		else if (status.MPI_TAG == 3)
		{ // run simulation
			storage.restore(); 
			ParseReceivedMessage(rPackage, parameter.GetMHProposalScaleSize()+4, parameter, energy_level, simulation_length); 

                        CEES_Node *simulator = GenerateSimulator(energy_level, filename_base, storage, parameter, target, r);

			InitializeSimulator(simulator[energy_level], energy_level, parameter, filename_base, storage, r, target);
			cout << energy_level << " ... Simulating for ... " << simulation_length << " steps.\n"; 
			// Simulation
                        simulator[energy_level].Simulate(r, storage, simulation_length, parameter.deposit_frequency, parameter.multiple_try_mh);

			storage.finalize(); 
			WrapUpSimulation(parameter, simulator[energy_level], filename_base, energy_level); 			

			// Send back message
			int done = 1; 
			MPI_Send(&done, 1, MPI_INT, 0, 3, MPI_COMM_WORLD); 
		}
		else 
		{
			delete [] rPackage; 
			delete [] sPackage; 
			exit(0); 
		}
	}
}

CEES_Node *GenerateSimulator(int energy_level, string filename_base, CStorageHead &storage, const CParameterPackage &parameter, CModel *target, const gsl_rng *r)
{
	// Dimensions
	CEES_Node::SetEnergyLevelNumber(parameter.number_energy_level); 
	CEES_Node::SetEquiEnergyJumpProb(parameter.pee); 
	CEES_Node::SetDataDimension(parameter.data_dimension); 
	
	// Energy bounds and temperatures
	double *temp_buffer_float = new double[parameter.number_energy_level]; 
	parameter.GetEnergyBound(temp_buffer_float, parameter.number_energy_level); 
	CEES_Node::SetEnergyLevels(temp_buffer_float, parameter.number_energy_level); 
	parameter.GetTemperature(temp_buffer_float, parameter.number_energy_level); 
	CEES_Node::SetTemperatures(temp_buffer_float, parameter.number_energy_level); 
	delete [] temp_buffer_float; 

	// Min-Max Energy
	CEES_Node::InitializeMinMaxEnergy(parameter.energy_tracking_number); 
	// Target acceptance rate for tuning 
	CEES_Node::SetTargetAcceptanceRate(parameter.mh_target_acc); 

	// Block_size
	int *temp_buffer_int = new int[parameter.number_block]; 
	parameter.GetBlockSize(temp_buffer_int, parameter.number_block); 
	CEES_Node::SetBlockSize(temp_buffer_int, parameter.number_block); 
	delete [] temp_buffer_int; 		

	// Generating simulators
	CEES_Node *simulator = new CEES_Node[parameter.number_energy_level]; 
	for (int i=0; i<parameter.number_energy_level; i++)
	{
		simulator[i].ultimate_target = target; 
		simulator[i].SetID_LocalTarget(i); 
		if (i < parameter.number_energy_level-1)
			simulator[i].SetHigherNodePointer(simulator+i+1); 
		else 
			simulator[i].SetHigherNodePointer(NULL); 
	}

	// Set proposal scales for simulator[energy_level];
	temp_buffer_float = new double [parameter.data_dimension]; 
	parameter.GetMHProposalScale(energy_level, temp_buffer_float, parameter.data_dimension); 
	int dim_cum_sum = 0; 
	for (int iBlock=0; iBlock<parameter.number_block; iBlock++)
	{
		simulator[energy_level].SetProposal(new CTransitionModel_SimpleGaussian(parameter.GetBlockSize(iBlock), temp_buffer_float+dim_cum_sum), iBlock); 
		dim_cum_sum += parameter.GetBlockSize(iBlock); 
	}
	delete [] temp_buffer_float;

	return simulator; 
}

void ParseReceivedMessage(double *rPackage, int package_size, CParameterPackage &parameter, int& energy_level, int &simulation_length)
{
	simulation_length = (int)(rPackage[0]); 
	energy_level = (int)(rPackage[1]);
        parameter.h0 = rPackage[2];
        parameter.hk_1 = rPackage[3];
        parameter.SetEnergyBound();
        parameter.SetTemperature();
        parameter.SetMHProposalScale(energy_level, rPackage+4, parameter.GetMHProposalScaleSize());
}

void InitializeSimulator(CEES_Node &simulator, int energy_level, CParameterPackage &parameter, string filename_base,  CStorageHead &storage, const gsl_rng *r, CModel *target)
{
	stringstream convert;
        convert.str(string());
        convert << parameter.run_id << "/" << parameter.run_id << ".current_state."  << energy_level;
        string filename = filename_base + convert.str();
        
	if (parameter.LoadCurrentStateFromFile(filename, energy_level) || parameter.LoadCurrentStateFromStorage(storage, r, energy_level))
		simulator.Initialize(parameter.GetCurrentState(energy_level));
	else
        {
        	CSampleIDWeight mode;
                target->GetMode(mode, 0);
                simulator.Initialize(mode);           
        }
	return; 
}

void WrapUpSimulation(CParameterPackage &parameter, const CEES_Node &simulator, string filename_base, int energy_level)
{
	/* Save current state */
	parameter.TraceSimulator(simulator);
	
	stringstream convert; 
	convert.str(string()); 
	convert << parameter.run_id << "/" << parameter.run_id << ".current_state."  << energy_level; 
	string filename = filename_base + convert.str(); 
	parameter.SaveCurrentStateToFile(filename, energy_level);  
	return; 
}
