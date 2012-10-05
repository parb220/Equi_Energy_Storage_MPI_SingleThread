#include <mpi.h>
#include "CParameterPackage.h"

using namespace std; 

/* WORKTAG: 2 run simulation for tracking*/

void DispatchTrackingTask(int nTasks, CParameterPackage &parameter)
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
	sPackage[0] = (double)(parameter.energy_level_tracking_window_length);
	// Information received from slave nodes
	// rPackage[0]: min energy
	// rPackage[1]: max energy
	double *rPackage = new double[2]; 
	
	MPI_Status status; 
	int rank; 

	if (nTasks-1 >= parameter.number_energy_level)
	{
		for (int rank=1; rank<nTasks; rank++)
		{
			sPackage[1] = (double)((rank-1)%parameter.number_energy_level); // energy level; 
			sPackage[2] = h0; 
			sPackage[3] = hk_1; 
			parameter.GetMHProposalScale((int)(sPackage[1]), sPackage+4, parameter.GetMHProposalScaleSize()); 
			MPI_Send(sPackage, parameter.GetMHProposalScaleSize()+4, MPI_DOUBLE, rank, 2, MPI_COMM_WORLD);  
		}
		for (int rank=1; rank<nTasks; rank++)
		{
			MPI_Recv(rPackage, 2, MPI_DOUBLE, MPI_ANY_SOURCE, 2, MPI_COMM_WORLD, &status); 
			h0 = h0 < rPackage[0] ? h0 : rPackage[0]; 
			hk_1 = hk_1 > rPackage[1] ? hk_1 : rPackage[1]; 
		}
	} else 
	{	/* need to recycle using slave_nodes to run through all energy levels */ 
		int level = parameter.number_energy_level-1; 
		vector <bool> send_receive(nTasks); 
		for (rank=1; rank<nTasks; rank++)
		{
			sPackage[1] = (double)(level); 	// energy level
			sPackage[2] = h0; 
			sPackage[3] = hk_1; 
			parameter.GetMHProposalScale(level, sPackage+4, parameter.GetMHProposalScaleSize());
			MPI_Send(sPackage, parameter.GetMHProposalScaleSize()+4, MPI_DOUBLE, rank, 2, MPI_COMM_WORLD); 
			send_receive[rank] = true; 
			level --; 
		}	
		while (level >= 0)
		{
			MPI_Recv(rPackage, 2, MPI_DOUBLE, MPI_ANY_SOURCE, 2, MPI_COMM_WORLD, &status); 
			rank = status.MPI_SOURCE; 
			send_receive[rank] = false; 

			h0 = h0 < rPackage[0] ? h0 : rPackage[0]; 
                        hk_1 = hk_1 > rPackage[1] ? hk_1 : rPackage[1];
\			
			sPackage[1] = (double)(level);    // energy level
			sPackage[2] = h0; 
			sPackage[3] = hk_1; 
			parameter.GetMHProposalScale(level, sPackage+4, parameter.GetMHProposalScaleSize());
                        MPI_Send(sPackage, parameter.GetMHProposalScaleSize()+4, MPI_DOUBLE, rank, 2, MPI_COMM_WORLD);
			send_receive[rank] = true; 
                        level --;
		}
		for (int rank=1; rank<nTasks; rank++)
		{
			if (send_receive[rank])
			{
				MPI_Recv(rPackage, 2, MPI_DOUBLE, MPI_ANY_SOURCE, 2, MPI_COMM_WORLD, &status); 
				h0 = h0 < rPackage[0] ? h0 : rPackage[0]; 
				hk_1 = hk_1 > rPackage[1] ? hk_1 : rPackage[1]; 
			} 	
		} 
	}
	double new_h0 = h0 < parameter.h0 ? h0 : parameter.h0;
	double new_hk_1 = hk_1 < parameter.hk_1 ? hk_1 : parameter.hk_1; 
	if (new_h0 < parameter.h0 || new_hk_1 > parameter.hk_1)
	{
		parameter.h0 = new_h0; 
		parameter.hk_1 = new_hk_1; 
		parameter.SetEnergyBound(); 
		parameter.SetTemperature(); 
	}
	
	delete [] rPackage; 
	delete [] sPackage; 
}
