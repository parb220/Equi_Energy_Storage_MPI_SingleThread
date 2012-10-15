#include <mpi.h>
#include <cmath>
#include "mpi_parameter.h"
#include "CParameterPackage.h"
#include "CStorageHead.h"

using namespace std; 

/* WORKTAG: 3 run simulation for tracking*/

void DispatchTrackingTask_LevelByLevel(int nTasks, CParameterPackage &parameter, CStorageHead &storage)
{
	// Information sent into slave nodes
	// sPackage[0] simulation length
	// sPackage[1] energy level
	// sPackage[2] min energy
	// sPackage[3] max energy
	// sPackage[4..end] scale 
	const int scale_size = parameter.GetMHProposalScaleSize(); 
	const int state_size = parameter.data_dimension; 
	double *sPackage = new double [scale_size+state_size+N_MESSAGE]; 
	// Information received from slave nodes
	// rPackage[0]: min energy
	// rPackage[1]: max energy
	double *rPackage = new double[4]; 
	
	MPI_Status status; 

	sPackage[LENGTH_INDEX] = ceil(parameter.energy_level_tracking_window_length/(nTasks-1.0));
	sPackage[H0_INDEX] = parameter.h0;
	double h0 = parameter.h0; 

	for (int level=parameter.number_energy_level-1; level>=0; level--)
	{
		sPackage[LEVEL_INDEX] = (double)(level); 
                parameter.GetMHProposalScale(level, sPackage+SCALE_INDEX, scale_size);
                // Tasks are sent out in chain
                // master send to 1st node, 1st node sent to 2nd node ...
		MPI_Send(sPackage, scale_size+state_size+N_MESSAGE, MPI_DOUBLE, 1, TRACKING_TAG, MPI_COMM_WORLD);
                
		// Taskes are collected in centralized way, all by master
		for (int rank=1; rank<nTasks; rank++)
		{
	        	MPI_Recv(rPackage, 4, MPI_DOUBLE, MPI_ANY_SOURCE, TRACKING_TAG, MPI_COMM_WORLD, &status);
                        h0 = h0 < rPackage[1] ? h0 : rPackage[1];
                }

		// Consolidate partial record files
		int binStart = (int)(rPackage[2]); 
		int binEnd = (int)(rPackage[3]); 
		storage.consolidate(binStart, binEnd); 
	}
	if (h0 < parameter.h0 )
	{
		parameter.h0 = h0; 
		parameter.SetEnergyBound(); 
		parameter.SetTemperature(); 
		storage.DisregardHistorySamples(); 		
	}

	delete [] rPackage; 
	delete [] sPackage; 
}
