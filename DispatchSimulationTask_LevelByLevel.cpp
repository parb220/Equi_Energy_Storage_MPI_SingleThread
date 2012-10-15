#include <mpi.h>
#include <cmath>
#include "mpi_parameter.h"
#include "CParameterPackage.h"
#include "CStorageHead.h"

using namespace std; 


void DispatchSimulation_LevelByLevel(int nTasks, const CParameterPackage &parameter, int highest_level, CStorageHead &storage)
{
	const int scale_size = parameter.GetMHProposalScaleSize(); 
	const int state_size = parameter.data_dimension; 

	double *sPackage = new double [scale_size+state_size+N_MESSAGE]; 
	double *rMessage = new double[3];  	
	MPI_Status status; 

	sPackage[H0_INDEX] = parameter.h0; 
	sPackage[LENGTH_INDEX] = ceil(parameter.simulation_length/(nTasks-1.0)); 
	for (int level =highest_level; level>=0; level--)
	{
		cout << "level: " << level << " simulation for " << (int)(sPackage[0])*(nTasks-1) << endl; 
		sPackage[LEVEL_INDEX] = (double)(level);
               	parameter.GetMHProposalScale(level, sPackage+SCALE_INDEX, scale_size);

		// Tasks are sent out in chain	
		MPI_Send(sPackage, scale_size+state_size+N_MESSAGE, MPI_DOUBLE, 1, SIMULATION_TAG, MPI_COMM_WORLD);
		// Tasks are collected in a centralized way
		for (int rank=1; rank<nTasks; rank++)
			MPI_Recv(rMessage, 3, MPI_DOUBLE, MPI_ANY_SOURCE, SIMULATION_TAG, MPI_COMM_WORLD, &status);
		
		int binStart = (int)(rMessage[1]); 
		int binEnd = (int)(rMessage[2]); 
		storage.consolidate(binStart, binEnd); 
	}

	delete [] sPackage; 
	delete [] rMessage;
} 
