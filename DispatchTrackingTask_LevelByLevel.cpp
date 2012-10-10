#include <mpi.h>
#include <cmath>
#include "CParameterPackage.h"

using namespace std; 

/* WORKTAG: 2 run simulation for tracking*/

void DispatchTrackingTask_LevelByLevel(int nTasks, CParameterPackage &parameter)
{
	double h0 = parameter.h0; 
	double hk_1 = parameter.hk_1; 
	// Information sent into slave nodes
	// sPackage[0] simulation length
	// sPackage[1] energy level
	// sPackage[2] min energy
	// sPackage[3] max energy
	// sPackage[4..end] scale 
	double *sPackage = new double [parameter.GetMHProposalScaleSize()+4]; 
	sPackage[0] = ceil(parameter.energy_level_tracking_window_length/(nTasks-1.0));
	// Information received from slave nodes
	// rPackage[0]: min energy
	// rPackage[1]: max energy
	double *rPackage = new double[2]; 
	
	MPI_Status status; 

	for (int level=parameter.number_energy_level-1; level>=0; level--)
	{
		sPackage[1] = (double)(level); 
		sPackage[2] = h0;
                sPackage[3] = hk_1;
                parameter.GetMHProposalScale(level, sPackage+4, parameter.GetMHProposalScaleSize());
		for (int rank=1; rank<nTasks; rank++)
                        MPI_Send(sPackage, parameter.GetMHProposalScaleSize()+4, MPI_DOUBLE, rank, 3, MPI_COMM_WORLD);
		for (int rank=1; rank<nTasks; rank++)
		{
                        MPI_Recv(rPackage, 3, MPI_DOUBLE, MPI_ANY_SOURCE, 3, MPI_COMM_WORLD, &status);
			int rLevel = (int)(rPackage[0]); 
                        h0 = h0 < rPackage[1] ? h0 : rPackage[1];
			if (rLevel == parameter.number_energy_level -1)
                        	hk_1 = hk_1 < rPackage[2] ? hk_1 : rPackage[2];
                }
	}
	parameter.h0 = h0 < parameter.h0 ? h0 : parameter.h0;
        parameter.hk_1 = hk_1 < parameter.hk_1 ? hk_1 : parameter.hk_1;

	delete [] rPackage; 
	delete [] sPackage; 
}
