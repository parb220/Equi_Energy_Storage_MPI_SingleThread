#include <mpi.h>
#include "mpi_parameter.h"
#include "CParameterPackage.h"

using namespace std;  

/* WORKTAG: 
 1: burn-in
 2: tuning */
void DispatchBurnInTask(int nTasks, const CParameterPackage &parameter)
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
	
	int rMessage;

	if (nTasks-1 >= parameter.number_energy_level)
        {
        	/* only need a number (number_energy_level) for burn-in */
        	for (int level=0; level<parameter.number_energy_level; level++)
        	{
        		sPackage[LEVEL_INDEX] = (double)(level);
        		rank = level+1;
        		MPI_Send(sPackage, scale_size+state_size+N_MESSAGE, MPI_DOUBLE, rank, BURN_TAG, MPI_COMM_WORLD);
        	}
       		for (int level=0; level<parameter.number_energy_level; level++)
       			MPI_Recv(&rMessage, 1, MPI_INT, MPI_ANY_SOURCE, BURN_TAG, MPI_COMM_WORLD, &status);  
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
        		MPI_Send(sPackage, scale_size+state_size+N_MESSAGE, MPI_DOUBLE, rank, BURN_TAG, MPI_COMM_WORLD);
        		level --;
        	}
        	while (level >= 0)
        	{
        		MPI_Recv(&rMessage, 1, MPI_INT, MPI_ANY_SOURCE, BURN_TAG, MPI_COMM_WORLD, &status);

        		rank = status.MPI_SOURCE;
			sPackage[LEVEL_INDEX] = (double)(level); 
        		MPI_Send(sPackage, scale_size+state_size+N_MESSAGE, MPI_DOUBLE, rank, BURN_TAG, MPI_COMM_WORLD);
        		level --;
        	}
		for (rank=1; rank<nTasks; rank++)
			MPI_Recv(&rMessage, 1, MPI_INT, MPI_ANY_SOURCE, BURN_TAG, MPI_COMM_WORLD, &status);
        }
	delete [] sPackage;
}
