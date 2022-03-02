#include <mpi.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <string.h>
#include <map>

using namespace std;

int main(int argc, char **argv)
{

	MPI_Init(&argc, &argv);
	map<int, vector<int>> depth_map;
	int myrank, n_proc, max_level = 1;
	int scatter_count, recvcount;
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &n_proc);

	if (n_proc == 1)
	{
		cout << "Invalid np -1 . Program requires atleast two processes two run." << endl;
		MPI_Finalize();
		return 0;
	}

	if (myrank == 0)
	{

		/* INPUT GRAPH */
		int no_of_vertices, no_of_edges, no_of_visited = 0, source, level = 1;
		cout << "Enter no_of_vertices, no_of_edges: ";
		cin >> no_of_vertices >> no_of_edges;
		int graph[no_of_vertices][no_of_vertices];
		cout << "Enter Edges:"<<endl;
		int visited[no_of_vertices], frontier[no_of_vertices], updated[no_of_vertices], output[no_of_vertices];
		int backup_n_proc = n_proc;
		if ((n_proc - 1) > no_of_vertices)
		{
			n_proc = no_of_vertices + 1;
		}

		/* INITIALIZATION */
		for (int i = 0; i < no_of_vertices; i++)
		{
			visited[i] = 0;
			frontier[i] = 0;
			updated[i] = 0;
			output[i] = 1;
			for (int j = 0; j < no_of_vertices; j++)
			{
				graph[i][j] = 0;
			}
		}
		for (int i = 0; i < no_of_edges; ++i)
		{
			int start, end;
			cin >> start >> end;
			start--;
			end--;
			graph[start][end] = 1;
			graph[end][start] = 1;
		}
		cout << "Enter Source: " ;
		cin>>source;
		// by default source strts from node 1
		// source = 0;
		frontier[source] = 1;
		visited[source] = 1;
		no_of_visited++;
		output[source] = level;
		depth_map[level].push_back(source + 1);
		level++;

		int partition_sz = ceil(((float)(no_of_vertices) / (n_proc - 1)));
		int iter = 1, remaining = no_of_vertices;
		int splice_sz = partition_sz;

		/* SCATTERING GRAPH CHUNKS TO ALL WORKERS*/
		// need to send partition_sz, no_of_vertices before sending chunk
		// need to send 1-1 since the all data partitions might
		// not have the same size ==> substitute for scatter
		int spawn = 1;
		while (remaining > 0)
		{

			if (remaining < partition_sz)
				splice_sz = remaining;

			int graph_chunk[splice_sz][no_of_vertices];

			for (int i = 0; i < splice_sz; i++)
			{
				memcpy(graph_chunk[i], graph[i + partition_sz * (iter - 1)], sizeof(graph[i]));
			}
			MPI_Send(&spawn, 1, MPI_INT, iter, 1, MPI_COMM_WORLD);
			MPI_Send(&splice_sz, 1, MPI_INT, iter, 2, MPI_COMM_WORLD);
			MPI_Send(&no_of_vertices, 1, MPI_INT, iter, 3, MPI_COMM_WORLD);
			MPI_Send(&graph_chunk, splice_sz * no_of_vertices, MPI_INT, iter, 4, MPI_COMM_WORLD);

			iter++;
			remaining -= splice_sz;
		}

		int n_proc_sent = (--iter);
		spawn = 0;
		while (iter < (backup_n_proc - 1))
		{
			iter++;
			MPI_Send(&spawn, 1, MPI_INT, iter, 1, MPI_COMM_WORLD);
		}

		while (1)
		{

			/* BROADCAST FRONTIER TO WORKERS */
			MPI_Bcast(&frontier, no_of_vertices, MPI_INT, 0, MPI_COMM_WORLD);

			remaining = no_of_vertices;
			splice_sz = partition_sz;

			/* MERGING MINI UPDATES TO UPDATE VECTOR */
			MPI_Status status;
			for (int i = 1; i <= n_proc_sent; i++)
			{

				if (remaining < partition_sz && remaining != 0)
					splice_sz = remaining;

				int mini_update[splice_sz];
				MPI_Recv(&mini_update, splice_sz, MPI_INT, i, 1, MPI_COMM_WORLD, &status);

				memcpy(&updated[splice_sz * (i - 1)], mini_update, sizeof(mini_update));
			}

			// -------- AFTER GATHERING ALL THE UPDATES

			/* SET NEW FRONTIER */
			memcpy(frontier, updated, sizeof(frontier));

			/* UPDATE LEVELS OF NEW FRONTIER */
			for (int i = 0; i < no_of_vertices; i++)
			{
				if (frontier[i] == 1)
				{
					if (visited[i] == 1)
					{
						frontier[i] = 0;
					}
					else
					{
						visited[i] = 1;
						output[i] = level;
						depth_map[level].push_back(i + 1);
						no_of_visited++;
					}
				}
			}

			level++;

			/* MASTER EXIT CONDITION */
			int master_exit = 0, done = 1;

			// /* CHECKS WETHER THE CURRENT CONNECTED GRAPH BFS IS COMPLETE */
			for (int i = 0; i < no_of_vertices; i++)
			{
				if (frontier[i] == 1)
				{
					done = 0;
					break;
				}
			}

			//  CHECKS IF ANY UNCONNECTED GRAPH IS REMAINING TO BFS
			if (done == 1)
			{
				memset(frontier, 0, sizeof(frontier));
				for (int j = 0; j < no_of_vertices; j++)
				{
					if (visited[j] == 0)
					{
						frontier[j] = 1;
						output[j] = 1;
						visited[j] = 1;
						level = 1;
						depth_map[1].push_back(j + 1);
						no_of_visited++;
					}
				}
			}

			if (no_of_vertices == no_of_visited)
				break;
		}

		/* WORKER TERMINATE CONDITION */
		frontier[0] = -1;
		MPI_Bcast(&frontier, no_of_vertices, MPI_INT, 0, MPI_COMM_WORLD);

		/* DISPLAY OUTPUT */
		map<int, vector<int>>::reverse_iterator rit;
		rit = depth_map.rbegin();

		cout << rit->first << endl;

		map<int, vector<int>>::iterator it;
		for (it = depth_map.begin(); it != depth_map.end(); ++it)
		{

			int sz = it->second.size(), cnt = 1;
			for (auto i : it->second)
			{
				cout << i;
				if (cnt < sz)
				{
					cout << ", ";
				}
				cnt++;
			}
			cout << endl;
		}
	}

	/* WORKER PROCESS */

	else
	{

		/* RECIEVE GRAPH CHUNK AT THE BEGINNING */
		MPI_Status status;
		int partition_sz, no_of_vertices, spawn;

		// indicates wether the process needs to be spawned or not
		MPI_Recv(&spawn, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);

		if (spawn == 1)
		{

			MPI_Recv(&partition_sz, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, &status);
			MPI_Recv(&no_of_vertices, 1, MPI_INT, 0, 3, MPI_COMM_WORLD, &status);

			int graph_chunk[partition_sz][no_of_vertices];
			MPI_Recv(&graph_chunk, partition_sz * no_of_vertices, MPI_INT, 0, 4, MPI_COMM_WORLD, &status);

			int updated[partition_sz], frontier[no_of_vertices];
			// cout << "chunks recieved at process "<< myrank <<endl;
			while (1)
			{

				MPI_Bcast(&frontier, no_of_vertices, MPI_INT, 0, MPI_COMM_WORLD);

				/* WORKER EXIT CONDITION */
				if (frontier[0] == -1)
					break;

				/* COMPUTE UPDATE VECTOR */
				for (int i = 0; i < sizeof(graph_chunk) / sizeof(graph_chunk[0]); i++)
				{

					int temp[no_of_vertices];
					transform(frontier, frontier + no_of_vertices, graph_chunk[i], temp, multiplies<int>());

					for (int j = 0; j < sizeof(temp) / sizeof(int); j++)
					{

						if (temp[j] == 1)
						{
							updated[i] = 1;
							break;
						}
						updated[i] = 0;
					}
				}
				/* SEND UPDATE VECTOR*/
				// gather does not work since all update vectors won't be of same size, due to uneven partition
				MPI_Send(&updated, partition_sz, MPI_INT, 0, 1, MPI_COMM_WORLD);
			}
		}
	}

	MPI_Finalize();
}