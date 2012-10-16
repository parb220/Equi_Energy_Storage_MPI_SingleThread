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
#include "mpi_parameter.h"

using namespace std;

/* MPI_TAG: 
 0: exit
 1: burn-in 
 2: tuning
 3: tracking
 4: simulation
*/

void InitializeSimulator(CEES_Node &, int energy_level, CParameterPackage &parameter, string filename_base, CStorageHead &storage, const gsl_rng *r, CModel *target);

int GetLengthLevelH0(const double *, int, CParameterPackage &) ; 
void GetMHScale(const double *, int, CParameterPackage &, int); 
void GetCurrentState(const double *, int, CParameterPackage &, int); 

CEES_Node *GenerateSimulator(int energy_level, string filename_base, CStorageHead &storage, const CParameterPackage &parameter, CModel *target, const gsl_rng *r);

void slave_single_thread(string filename_base, CStorageHead &storage, CParameterPackage &parameter, int highest_level, bool if_resume, CModel *target, const gsl_rng* r)
{
	int my_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);  
	
	int nTasks; 
	MPI_Comm_size(MPI_COMM_WORLD, &nTasks); 

	MPI_Status status; 
	
	const int scale_size = parameter.GetMHProposalScaleSize();
	const int state_size = parameter.data_dimension; 

	double *rPackage = new double [scale_size+state_size+N_MESSAGE]; 
	double *sPackage = NULL;   
	CEES_Node *simulator; 

	int energy_level;
	while (1)
	{
		MPI_Recv(rPackage, scale_size+state_size+N_MESSAGE, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status); 
		if (status.MPI_TAG == BURN_TAG)
		{ 
			energy_level = GetLengthLevelH0(rPackage, scale_size+state_size+N_MESSAGE, parameter); 
			simulator = GenerateSimulator(energy_level, filename_base, storage, parameter, target, r); 
			// randomly set current state 
			parameter.SetCurrentState(r, energy_level); 
                       	simulator[energy_level].Initialize(parameter.GetCurrentState(energy_level));
			
			// cout << energy_level << ": " << my_rank << " ... Burn In ... " << parameter.simulation_length << endl;
			simulator[energy_level].BurnIn(r, storage, parameter.simulation_length, parameter.multiple_try_mh);
			int done=1; 
			MPI_Send(&done, 1, MPI_INT, 0, BURN_TAG, MPI_COMM_WORLD); 
			delete [] simulator; 
		}
		else if (status.MPI_TAG == TUNE_TAG)
		{ 
			energy_level = GetLengthLevelH0(rPackage, scale_size+state_size+N_MESSAGE, parameter); 
			GetMHScale(rPackage, scale_size+state_size+N_MESSAGE, parameter, energy_level);
			simulator = GenerateSimulator(energy_level, filename_base, storage, parameter, target, r); 
				
			// always use mode as the starting state for tuning
			CSampleIDWeight mode;
                       	target->GetMode(mode, 0);
                       	simulator[energy_level].Initialize(mode);
				
			// cout << energy_level << ": " << my_rank << " ... Tune MH Proposal Scales" << endl; 
			simulator[energy_level].MH_StepSize_Tune(parameter.mh_tracking_length, parameter.mh_stepsize_tuning_max_time, r, parameter.multiple_try_mh);  
			
			// to get the scale of the proposal distributions
			parameter.TraceSimulator(simulator[energy_level]);

			sPackage = new double [scale_size+1]; 	
			sPackage[0] = (double)(energy_level);
			parameter.GetMHProposalScale(energy_level, sPackage+1, scale_size);
			MPI_Send(sPackage, scale_size+1, MPI_DOUBLE, 0, TUNE_TAG, MPI_COMM_WORLD); 
			delete [] simulator; 
		}
		else if (status.MPI_TAG == TRACKING_TAG || status.MPI_TAG == SIMULATION_TAG)
		{ 
			energy_level = GetLengthLevelH0(rPackage, scale_size+state_size+N_MESSAGE, parameter);
			GetMHScale(rPackage, scale_size+state_size+N_MESSAGE, parameter, energy_level);
			simulator = GenerateSimulator(energy_level, filename_base, storage, parameter, target, r); 
			// restore partial storage of higher level for fetching but no updating
			if (energy_level < parameter.number_energy_level-1)
				storage.RestoreForFetch(simulator[energy_level+1].BinID(0), simulator[energy_level+1].BinID(parameter.number_energy_level-1)); 
			// restore partial storage (previously obtained at this node) for updating
			storage.restore(simulator[energy_level].BinID(0), simulator[energy_level].BinID(parameter.number_energy_level-1)); 
			if (my_rank == 1)
			{
				if (energy_level == parameter.number_energy_level-1 && !parameter.LoadCurrentStateFromStorage(storage, r, energy_level))
				{
                                	CSampleIDWeight mode;
					target->GetMode(mode, 0);
                                       	simulator[energy_level].Initialize(mode);
				}
				else 
				{
					if (parameter.LoadCurrentStateFromStorage(storage, r, energy_level))
                               			simulator[energy_level].Initialize(parameter.GetCurrentState(energy_level));
					else 
						cerr << "Error in initializing " << energy_level << " " << my_rank << endl; 
				}
			}
			else 
			{
				if (!parameter.LoadCurrentStateFromStorage(storage, r, energy_level))
					// Get current state from rPackage
					GetCurrentState(rPackage, scale_size+state_size+N_MESSAGE, parameter, energy_level); 
				simulator[energy_level].Initialize(parameter.GetCurrentState(energy_level));
			} 

			// cout << energy_level << ": " << my_rank << " tracking for simulating " << parameter.simulation_length << endl; 
			if (my_rank == nTasks-1)
				simulator[energy_level].Simulate(r, storage, parameter.simulation_length, parameter.deposit_frequency, parameter.multiple_try_mh);
			else 
			{
				int initial_length = 1000 < parameter.simulation_length ? 1000 : parameter.simulation_length; 
				// initiate next node by sending over the current state after 1000 steps of simulation
				simulator[energy_level].Simulate(r, storage, initial_length, parameter.deposit_frequency, parameter.multiple_try_mh); 
				CSampleIDWeight x = simulator[energy_level].GetCurrentState(); 
				double *chainSPackage = new double[scale_size+state_size+N_MESSAGE]; 
				memcpy(chainSPackage, rPackage, sizeof(double)*(scale_size+state_size+N_MESSAGE)); 
				x.CopyData(chainSPackage+SCALE_INDEX+scale_size, state_size); 
				// only responsible for sending, but not for collecting
				MPI_Send(chainSPackage, scale_size+state_size+N_MESSAGE, MPI_DOUBLE, my_rank+1, status.MPI_TAG, MPI_COMM_WORLD); 
				delete [] chainSPackage; 

				int continue_length = 0 > parameter.simulation_length - initial_length ? 0 : parameter.simulation_length - initial_length; 
				simulator[energy_level].Simulate(r, storage, continue_length, parameter.deposit_frequency, parameter.multiple_try_mh);
			}
				
			// finalize storage
			storage.finalize(simulator[energy_level].BinID(0), simulator[energy_level].BinID(parameter.number_energy_level-1)); 
			// clear memory of storage
			storage.ClearDepositDrawHistory(simulator[energy_level].BinID(0), simulator[energy_level].BinID(parameter.number_energy_level-1)); 
			storage.ClearDepositDrawHistory(simulator[energy_level+1].BinID(0), simulator[energy_level+1].BinID(parameter.number_energy_level-1)); 


			if (status.MPI_TAG == TRACKING_TAG)
			{
				sPackage = new double [4]; 
				sPackage[0] = energy_level; 
				sPackage[1] = simulator[energy_level].GetMinEnergy(0); 
				sPackage[2] = simulator[energy_level].BinID(0);
				sPackage[3] = simulator[energy_level].BinID(parameter.number_energy_level-1);
				MPI_Send(sPackage, 4, MPI_DOUBLE, 0, TRACKING_TAG, MPI_COMM_WORLD); 
				delete [] sPackage;
			}
			else 
			{
				sPackage = new double[3]; 
				sPackage[0] = energy_level; 
				sPackage[1] = simulator[energy_level].BinID(0);
				sPackage[2] = simulator[energy_level].BinID(parameter.number_energy_level-1);
				MPI_Send(sPackage, 3, MPI_DOUBLE, 0, SIMULATION_TAG, MPI_COMM_WORLD); 
				delete [] sPackage; 
			}
			delete [] simulator; 
		}
		else 
		{
			delete [] rPackage; 
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

	// Min-Max Energy Size
	CEES_Node::InitializeMinMaxEnergy(parameter.energy_tracking_number); 
	// Target acceptance rate for tuning 
	CEES_Node::SetTargetAcceptanceRate(parameter.mh_target_acc); 

	// Block_size
	int *temp_buffer_int = new int[parameter.number_block]; 
	parameter.GetBlockSize(temp_buffer_int, parameter.number_block); 
	CEES_Node::SetBlockSize(temp_buffer_int, parameter.number_block); 
	delete [] temp_buffer_int; 		

	// Energy bounds and temperatures
	double *temp_buffer_float = new double[parameter.number_energy_level]; 
	parameter.GetEnergyBound(temp_buffer_float, parameter.number_energy_level); 
	CEES_Node::SetEnergyLevels(temp_buffer_float, parameter.number_energy_level); 
	parameter.GetTemperature(temp_buffer_float, parameter.number_energy_level); 
	CEES_Node::SetTemperatures(temp_buffer_float, parameter.number_energy_level); 
	delete [] temp_buffer_float; 
	
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

int GetLengthLevelH0(const double *rPackage, int package_size, CParameterPackage&parameter) 
{
	parameter.simulation_length = (int)(rPackage[LENGTH_INDEX]); 
	int energy_level = (int)(rPackage[LEVEL_INDEX]);
        parameter.h0 = rPackage[H0_INDEX];
        parameter.SetEnergyBound();
        parameter.SetTemperature();
	return energy_level; 
}

void GetMHScale(const double *rPackage, int package_size, CParameterPackage& parameter, int energy_level)
{
	const int scale_size = parameter.GetMHProposalScaleSize(); 
        parameter.SetMHProposalScale(energy_level, rPackage+SCALE_INDEX, scale_size); 
}

void GetCurrentState(const double *rPackage, int package_size, CParameterPackage& parameter, int energy_level)
{
	const int scale_size = parameter.GetMHProposalScaleSize(); 
	const int state_size = parameter.data_dimension; 
	parameter.SetCurrentState(energy_level, rPackage+SCALE_INDEX+scale_size, state_size); 
}
