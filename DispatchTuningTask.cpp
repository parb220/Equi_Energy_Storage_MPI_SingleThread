#include <mpi.h>
#include "CParameterPackage.h"

using namespace std;  

/* WORKTAG: 1 : tuning */
void DispatchTuningTask(int nTasks, CParameterPackage &parameter)
{
	/* Let a number (number_energy_level) of nodes run tuning (scale of proposal 
 	* distribution), and send the scales back */
	MPI_Status status;
	int rank;
	/* scale [0]: level id
 	   scale [1..end]: scale */
	double *sPackage = new double [parameter.GetMHProposalScaleSize()+4]; 
	sPackage[0] = 0; 
	sPackage[2] = parameter.h0; 
	sPackage[3] = parameter.hk_1; 
	// sPackage[0] unused
	// sPackage[1] energy level
	// sPackage[2] min energy
	// sPackage[3] max energy
	// sPackage[4..end] old proposal scales

        double *rPackage = new double [parameter.GetMHProposalScaleSize()+1];
	// rPackage[0] energy_level
	// rPackage[1..end] new proposal scales

	if (nTasks-1 >= parameter.number_energy_level)
        {
        	/* only need a number (number_energy_level) for tuning */
        	for (int i=0; i<parameter.number_energy_level; i++)
        	{
        		sPackage[1] = (double)(i);
			parameter.GetMHProposalScale((int)(sPackage[1]), sPackage+4, parameter.GetMHProposalScaleSize()); 
        		rank = i+1;
        		MPI_Send(sPackage, parameter.GetMHProposalScaleSize()+4, MPI_DOUBLE, rank, 1, MPI_COMM_WORLD);
        	}
       		for (int i=0; i<parameter.number_energy_level; i++)
       		{
       			MPI_Recv(rPackage, parameter.GetMHProposalScaleSize()+1, MPI_DOUBLE, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);  
       			parameter.SetMHProposalScale((int)(rPackage[0]), rPackage+1, parameter.GetMHProposalScaleSize());
       		}
	}
	else
        {
        	/* because nTasks-1 (=number_slave_nodes) < number_energy_level
 		 * need to recycle using slave_nodes */
        	int level = parameter.number_energy_level-1;
		// maintaining a list flagging send/receive status (t/f)
		vector <bool> send_receive(nTasks); 
        	for (rank=1; rank<nTasks; rank++)
        	{
			sPackage[1] = (double)(level); 
			parameter.GetMHProposalScale((int)(sPackage[1]), sPackage+4, parameter.GetMHProposalScaleSize()); 
        		MPI_Send(sPackage, parameter.GetMHProposalScaleSize()+4, MPI_DOUBLE, rank, 1, MPI_COMM_WORLD);
			send_receive[rank] = true; 
        		level --;
        	}
        	while (level >= 0)
        	{
        		MPI_Recv(rPackage, parameter.GetMHProposalScaleSize()+1, MPI_DOUBLE, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);
        		rank = status.MPI_SOURCE;
			send_receive[rank] = false; 
        		parameter.SetMHProposalScale((int)(rPackage[0]), rPackage+1, parameter.GetMHProposalScaleSize());
			sPackage[1] = (double)(level); 
			parameter.GetMHProposalScale((int)(sPackage[1]), sPackage+4, parameter.GetMHProposalScaleSize());
        		MPI_Send(sPackage, parameter.GetMHProposalScaleSize()+4, MPI_DOUBLE, rank, 1, MPI_COMM_WORLD);
			send_receive[rank] = true; 
        		level --;
        	}
		for (rank=1; rank<nTasks; rank++)
		{
			if (send_receive[rank]) 
			{
				MPI_Recv(rPackage, parameter.GetMHProposalScaleSize()+1, MPI_DOUBLE, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);
                        	parameter.SetMHProposalScale((int)(rPackage[0]), rPackage+1, parameter.GetMHProposalScaleSize());
			}
		}
        }
	delete [] sPackage; 
	delete [] rPackage; 
}
