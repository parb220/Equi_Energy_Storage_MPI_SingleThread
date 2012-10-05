#include <mpi.h>
#include "CParameterPackage.h"

using namespace std; 

/* WORKTAG: 3 run simulation */

void DispatchSimulation(int nTasks, const CParameterPackage &parameter, int highest_level)
{
	double *sPackage = new double [parameter.GetMHProposalScaleSize()+4]; 
	// sPackage[0]: simulation length
	// sPackage[1]: energy level
	// sPackage[2]: min energy
	// sPackage[3]: max energy
	// spackage[4..end]: scale of proposal distribution
	
	sPackage[0] = parameter.simulation_length; 
	sPackage[2] = parameter.h0; 
	sPackage[3] = parameter.hk_1; 

	int rMessage; 
	MPI_Status status; 
	int rank; 

	if (nTasks-1 >= (highest_level+1))
	{
		for (int rank=1; rank<nTasks; rank++)
		{
			sPackage[1] = (double)((rank-1)%(highest_level+1)); 
			parameter.GetMHProposalScale((int)(sPackage[1]), sPackage+4, parameter.GetMHProposalScaleSize());
			MPI_Send(sPackage, parameter.GetMHProposalScaleSize()+4, MPI_DOUBLE, rank, 4, MPI_COMM_WORLD);
		}
		for (int rank=1; rank<nTasks; rank++)
			MPI_Recv(&rMessage, 1, MPI_INT, MPI_ANY_SOURCE, 4, MPI_COMM_WORLD, &status);
	}
	else 
	{	/* need to recycle using slave_nodes */
		int level = highest_level;  
		vector<bool> send_receive(nTasks); 
		for (rank=1; rank<nTasks; rank++)
		{
			sPackage[1] = (double)(level); 
			parameter.GetMHProposalScale(level, sPackage+4, parameter.GetMHProposalScaleSize());
			MPI_Send(sPackage, parameter.GetMHProposalScaleSize()+4, MPI_DOUBLE, rank, 4, MPI_COMM_WORLD); 
			send_receive[rank] = true; 
			level --; 
		} 
		while (level >= 0)
		{
			MPI_Recv(&rMessage, 1, MPI_INT, MPI_ANY_SOURCE, 4, MPI_COMM_WORLD, &status); 
			rank = status.MPI_SOURCE; 
			send_receive[rank] = false; 

			sPackage[1] = (double)(level); 
			parameter.GetMHProposalScale(level, sPackage+4, parameter.GetMHProposalScaleSize());
			MPI_Send(sPackage, parameter.GetMHProposalScaleSize()+4, MPI_DOUBLE, rank, 4, MPI_COMM_WORLD);
			send_receive[rank] = true; 

			level --; 
		}
		for (int rank=1; rank<nTasks; rank++)
		{
			if (send_receive[rank])
				MPI_Recv(&rMessage, 1, MPI_INT, MPI_ANY_SOURCE, 4, MPI_COMM_WORLD, &status);
		}
	}

	delete [] sPackage; 
} 
