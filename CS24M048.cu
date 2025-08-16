
#include <chrono>
#include <cuda.h>
#include <cooperative_groups.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <vector>
#include <string>

#define MOD 1000000007
#define INF 0xFFFFFFFFFFFFFFFFULL
#define ll long long int
#define ui unsigned int
#define ull unsigned long long int
#define MAX_VERTICES 1000000 // Up to 10^6 vertices
#define MAX_EDGES 10000000   // Up to 10^7 edges
#define BLOCKSIZE 1024
#define MAX_BLOCKS 40 // Limited by grid.sync() capability on T4

using namespace std;
using namespace cooperative_groups;

__device__ int find_repres(volatile int *comp, int v)
{
    int p = comp[v];
    if (v == p)
        return v;

    // Two-jump path compression that avoids excessive atomic operations
    int gp = comp[p];
    while (p != gp)
    {
        // Only attempt to compress when needed
        atomicCAS((int *)&comp[v], p, gp);
        v = gp;
        p = comp[v];
        gp = comp[p];
    }
    return p;
}

__device__ int get_weight(int w, char c)
{
    if (c == 'g')
    {
        return w * 2;
    }
    else if (c == 't')
    {
        return w * 5;
    }
    else if (c == 'd')
    {
        return w * 3;
    }
    else
    {
        return w;
    }
}

__global__ void initializing(ull *safe, volatile int *comp, int V)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = bid * blockDim.x + tid;
    int total_threads = blockDim.x * gridDim.x;

    for (int i = gid; i < V; i += total_threads)
    {
        safe[i] = INF;
        comp[i] = i;
    }
}

__global__ void boruvkaMST(
    ull *safe, int *isactive, volatile int *comp, int *source, int *w, int *desti,
    ull *result, ui *cnt, int V, int E, int *iterations, char *type)
{
    grid_group grid = this_grid();

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = bid * blockDim.x + tid;
    int total_threads = blockDim.x * gridDim.x;
    ull local_sum = 0;

    while (*cnt > 1 && *iterations < 100)
    {

        // Phase 2: Find minimum weight edges between components
        for (int i = gid; i < E; i += total_threads)
        {
            if (isactive[i])
            {
                continue;
            }
            int u = source[i];
            int v = desti[i];

            int compU = comp[u];
            int compV = comp[v];

            if (compU != compV)
            {
                int weight = get_weight(w[i], type[i]);
                unsigned long long value = ((unsigned long long)weight << 32) | (unsigned int)i;
                atomicMin(safe + compU, value);
                atomicMin(safe + compV, value);
            }
        }

        grid.sync();

        // Phase 3: Add edge weights to MST total
        for (int i = gid; i < V; i += total_threads)
        {
            ull value = safe[i];
            if (value == INF)
                continue;

            int EID = (int)(value & 0xFFFFFFFF);

            if (EID != -1 && atomicCAS(isactive + EID, 0, 1) == 0)
            {
                local_sum += get_weight(w[EID], type[EID]);
            }
        }

        // Phase 4: Merge components based on safe edges

        for (int i = gid; i < V; i += total_threads)
        {
            ull value = safe[i];
            if (value == INF)
                continue;

            int EID = (int)(value & 0xFFFFFFFF);

            int u_root = find_repres(comp, source[EID]);
            int v_root = find_repres(comp, desti[EID]);
            while (u_root != v_root)
            {
                if (u_root > v_root)
                {
                    int temp = u_root;
                    u_root = v_root;
                    v_root = temp;
                }
                if (atomicCAS((int *)comp + u_root, u_root, v_root) == u_root)
                {
                    break;
                }
                u_root = find_repres(comp, source[EID]);
                v_root = find_repres(comp, desti[EID]);
            }
        }

        if (gid == 0)
            *cnt = 0;
        grid.sync();

        // Phase 5: Path compression to flatten component trees
        // Combine Phases 5 and 6
        int local_count = 0;
        for (int i = gid; i < V; i += total_threads)
        {
            // Path compression
            int par = find_repres(comp, i);
            atomicExch((int *)(comp + i), par);
            if (comp[i] == i)
            {
                local_count++;
            }
            safe[i] = INF;
        }

        atomicAdd(cnt, local_count);
        // Increment iteration counter
        if (gid == 0)
        {
            (*iterations)++;
        }

        grid.sync();
    }
    if (local_sum)
        atomicAdd(result, local_sum);
}

int main()
{

    int V, E;
    cin >> V >> E;

    vector<int> h_w(E), h_source(E), h_desti(E);

    char h_type[E];

    // Read edges with weights and edge types
    for (int i = 0; i < E; i++)
    {
        int u, v, w;
        string s;
        cin >> u >> v >> w;
        cin >> s;
        h_type[i] = s[0];
        h_w[i] = w;
        h_source[i] = u;
        h_desti[i] = v;
    }

    // Initialize  variables
    ui h_cnt = V;
    ull h_result = 0;
    int h_iterations = 0;

    ull *d_safe;
    int *isactive;
    char *type;
    ull *d_result;
    int *d_source, *d_w, *d_desti, *d_iterations;
    volatile int *d_comp;
    ui *d_cnt;

    // cudaMallocs
    cudaMalloc(&d_safe, V * sizeof(long long));
    cudaMalloc(&type, E * sizeof(char));
    cudaMalloc(&isactive, E * sizeof(int));
    cudaMalloc(&d_comp, V * sizeof(int));
    cudaMalloc(&d_source, E * sizeof(int));
    cudaMalloc(&d_w, E * sizeof(int));
    cudaMalloc(&d_desti, E * sizeof(int));
    cudaMalloc(&d_result, sizeof(long long));
    cudaMalloc(&d_cnt, sizeof(ui));
    cudaMalloc(&d_iterations, sizeof(int));

    // Initialize device memory

    cudaMemset(isactive, 0, E * sizeof(int));
    cudaMemcpy(type, h_type, E * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_source, h_source.data(), E * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w.data(), E * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_desti, h_desti.data(), E * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cnt, &h_cnt, sizeof(ui), cudaMemcpyHostToDevice);
    cudaMemcpy(d_iterations, &h_iterations, sizeof(int), cudaMemcpyHostToDevice);

    // Configure kernel launch parameters
    int threadsPerBlock = min(BLOCKSIZE, V);
    int numBlocks = min(MAX_BLOCKS, (V + threadsPerBlock - 1) / threadsPerBlock);
    int numBlocks2 = min(MAX_BLOCKS, (E + threadsPerBlock - 1) / threadsPerBlock);

    // Answer should be calculated in Kernel. No operations should be performed here.
    // Only copy data to device, kernel call, copy data back to host, and print the answer.
    auto start = std::chrono::high_resolution_clock::now();
    void *kernelArgs[] = {
        (void *)&d_safe, (void *)&isactive, (void *)&d_comp, (void *)&d_source, (void *)&d_w,
        (void *)&d_desti, (void *)&d_result, (void *)&d_cnt,
        (void *)&V, (void *)&E, (void *)&d_iterations, (void *)&type};
    initializing<<<numBlocks, threadsPerBlock>>>(d_safe, d_comp, V);
    cudaError_t launchStatus = cudaLaunchCooperativeKernel(
        (void *)boruvkaMST,
        dim3(numBlocks),
        dim3(threadsPerBlock),
        kernelArgs);
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed1 = end - start;
    cudaError_t errorCode = cudaGetLastError();
    if (errorCode != cudaSuccess)
    {
        cerr << "CUDA error: " << cudaGetErrorString(errorCode) << endl;
        return 1;
    }
    cudaMemcpy(&h_result, d_result, sizeof(long long), cudaMemcpyDeviceToHost);

    std::ofstream file("cuda.out");
    if (file.is_open())
    {
        cout << h_result % MOD << endl;
        file.close();
    }
    else
    {
        std::cout << "Unable to open file";
    }

    std::ofstream file2("cuda_timing.out");
    if (file2.is_open())
    {
        file2 << elapsed1.count() << "\n";
        file2.close();
    }
    else
    {
        std::cout << "Unable to open file";
    }

    // Free device memory
    cudaFree(d_safe);
    cudaFree((int *)d_comp);
    cudaFree(d_source);
    cudaFree(d_w);
    cudaFree(d_desti);
    cudaFree(d_result);
    cudaFree(d_cnt);
    cudaFree(d_iterations);

    return 0;
}
