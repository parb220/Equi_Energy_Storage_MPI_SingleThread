#include <mpi.h>
#include <cmath>
#include "CParameterPackage.h"
#include "CStorageHead.h"

using namespace std; 

/* WORKTAG: 3 run simulation */

void DispatchSimulation_LevelByLevel(int nTasks, const CParameterPackage &parameter, int highest_level, CStorageHead &storage)
{
	double *sPackage = new double [parameter.GetMHProposalScaleSize()+4]; 
	// sPackage[0]: simulation length
	// sPackage[1]: energy level
	// sPackage[2]: min energy
	// sPackage[3]: max energy
	// spackage[4..end]: scale of proposal distribution
	
	sPackage[0] = ceil(parameter.simulation_length/(nTasks-1.0)); 
	sPackage[2] = parameter.h0; 
	sPackage[3] = parameter.hk_1; 

	double *rMessage = new double[3];  	
	MPI_Status status; 

	for (int level =highest_level; level>=0; level--)
	{
		cout << "level: " << level << " simulation for " << (int)(sPackage[0])*(nTasks-1) << endl; 
		sPackage[1] = (double)(level);
               	parameter.GetMHProposalScale(level, sPackage+4, parameter.GetMHProposalScaleSize());
		for (int rank=1; rank<nTasks; rank++)
			MPI_Send(sPackage, parameter.GetMHProposalScaleSize()+4, MPI_DOUBLE, rank, 4, MPI_COMM_WORLD);
		for (int rank=1; rank<nTasks; rank++)
			MPI_Recv(rMessage, 3, MPI_DOUBLE, MPI_ANY_SOURCE, 4, MPI_COMM_WORLD, &status);
		
		int binStart = (int)(rMessage[1]); 
		int binEnd = (int)(rMessage[2]); 
		storage.consolidate(binStart, binEnd); 
	}

	delete [] sPackage; 
} 
