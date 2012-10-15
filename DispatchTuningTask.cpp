#include <mpi.h>
#include "mpi_parameter.h"
#include "CParameterPackage.h"

using namespace std;  

void DispatchTuningTask(int nTasks, CParameterPackage &parameter)
{
	/* Let a number (number_energy_level) of nodes run tuning (scale of proposal 
 	* distribution), and send the scales back */
	MPI_Status status;
	int rank;
	const int scale_size = parameter.GetMHProposalScaleSize(); 
	const int state_size = parameter.data_dimension; 

	double *sPackage = new double [scale_size+state_size+N_MESSAGE]; 
	sPackage[LENGTH_INDEX] = 0;  
	sPackage[H0_INDEX] = parameter.h0; 

        double *rPackage = new double [scale_size+1];
	// rPackage[0] energy_level
	// rPackage[1..end] new proposal scales

	if (nTasks-1 >= parameter.number_energy_level)
        {
        	/* only need a number (number_energy_level) for burn-in */
        	for (int level=0; level<parameter.number_energy_level; level++)
        	{
        		sPackage[LEVEL_INDEX] = (double)(level);
			parameter.GetMHProposalScale(level, sPackage+SCALE_INDEX, scale_size); 
        		rank = level+1;
        		MPI_Send(sPackage, scale_size+state_size+N_MESSAGE, MPI_DOUBLE, rank, TUNE_TAG, MPI_COMM_WORLD);
        	}
       		for (int level=0; level<parameter.number_energy_level; level++)
       		{
       			MPI_Recv(rPackage, scale_size+1, MPI_DOUBLE, MPI_ANY_SOURCE, TUNE_TAG, MPI_COMM_WORLD, &status);  
       			parameter.SetMHProposalScale((int)(rPackage[0]), rPackage+1, scale_size);
       		}
	}
	else
        {
        	/* because nTasks-1 (=number_slave_nodes) < number_energy_level
 		 * need to recycle using slave_nodes */
        	int level = parameter.number_energy_level-1;
		// maintaining a list flagging send/receive status (t/f)
        	for (rank=1; rank<nTasks; rank++)
        	{
			sPackage[LEVEL_INDEX] = (double)(level); 
			parameter.GetMHProposalScale(level, sPackage+SCALE_INDEX, scale_size); 
        		MPI_Send(sPackage, scale_size+state_size+N_MESSAGE, MPI_DOUBLE, rank, TUNE_TAG, MPI_COMM_WORLD);
        		level --;
        	}
        	while (level >= 0)
        	{
        		MPI_Recv(rPackage, scale_size+1, MPI_DOUBLE, MPI_ANY_SOURCE, TUNE_TAG, MPI_COMM_WORLD, &status);
        		rank = status.MPI_SOURCE;
        		parameter.SetMHProposalScale((int)(rPackage[0]), rPackage+1, scale_size);
			
			sPackage[LEVEL_INDEX] = (double)(level); 
			parameter.GetMHProposalScale(level, sPackage+SCALE_INDEX, scale_size);
        		MPI_Send(sPackage, scale_size+state_size+N_MESSAGE, MPI_DOUBLE, rank, TUNE_TAG, MPI_COMM_WORLD);
        		level --;
        	}
		for (rank=1; rank<nTasks; rank++)
		{
			MPI_Recv(rPackage, scale_size+1, MPI_DOUBLE, MPI_ANY_SOURCE, TUNE_TAG, MPI_COMM_WORLD, &status);
                       	parameter.SetMHProposalScale((int)(rPackage[0]), rPackage+1, scale_size);
		}
        }
	delete [] sPackage; 
	delete [] rPackage; 
}
