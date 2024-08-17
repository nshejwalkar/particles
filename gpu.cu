#include "common.h"
#include <cuda.h>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <chrono>

using namespace std;

#define NUM_THREADS 256
#define NUM_THREADS_FORCE 16

// Put any static global variables here that you will use throughout the simulation.
#define bin_size (5*cutoff)
int blks;
int blks_bins;
// Number of bins in a row of the grid
int row_num_bins;

// Total number of bins in the grid
int total_num_bins;

// Array of particle_id's representing the grid
int* bins;

// GPU copy of bins
int* bins_gpu;

// counts[i] says how many particles are in bin i (bin count array)


// Prefix sum of counts (starting position in the bins array)
int* bin_id;
int* copy_bin_idd;
// GPU copy of bin_id
int* bin_id_gpu;

// thrust::device_vector<int> counts_gpu;
int* d_output;
int* counts_gpu;
int* counts;

double exclusive_scan_time;
double sort_time;
double copy_bin_id_time;
double compute_forces_gpu_time;
double move_gpu_time;
double zero_count_time;

__device__ void apply_force_gpu(particle_t& particle, particle_t& neighbor) {
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    if (r2 > cutoff * cutoff)
        return;
    // r2 = fmax( r2, min_r*min_r );
    r2 = (r2 > min_r * min_r) ? r2 : min_r * min_r;
    double r = sqrt(r2);

    //  very simple short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

__device__ void apply_force_gpu_condensed(particle_t& particle, double dx, double dy) {
    // double dx = neighbor_x - particle.x;
    // double dy = neighbor_y - particle.y;
    // if (dx == dy && dx == 0)
    //     return;
    double r2 = dx * dx + dy * dy;

    if (r2 > cutoff * cutoff)
        return;
    // r2 = fmax( r2, min_r*min_r );
    r2 = (r2 > min_r * min_r) ? r2 : min_r * min_r;
    double r = sqrt(r2);

    //  very simple short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

__global__ void compute_forces_condensed_gpu(particle_t* particles,int* bin_id, int* bins, int num_parts, int row_num_bins, int total_num_bins, int* counts) {
    // Get thread (particle) ID
    // {{-1,-1},{-1,0},{-1,1},{0,-1},{0,0}, {0,1}, {1,-1}, {1,0}, {1,1}}
    const int NUM_NEIGH = 9;
    __shared__ double neigh_parts_x[NUM_THREADS_FORCE*NUM_NEIGH];
    __shared__ double neigh_parts_y[NUM_THREADS_FORCE*NUM_NEIGH];

    int num_neigh_part = 0;

    // Where in bins_gpu to index - starting point of ith neighbor bin
    int starting_id[NUM_NEIGH] = {-1};

    // bin_sizes[i] is how big ith neighbor bin is
    int bin_sizes[NUM_NEIGH] = {0};
    const int bin_num = blockIdx.x;
    const int tid = threadIdx.x;
    int bin_sizes_prefix[NUM_NEIGH] = {0};

    // Array of bin numbers corresponding to neighbors
    const int bin_nums[NUM_NEIGH] = {bin_num - row_num_bins - 1, bin_num - row_num_bins, bin_num - row_num_bins + 1, bin_num - 1, bin_num, bin_num + 1,
    bin_num + row_num_bins- 1 , bin_num + row_num_bins, bin_num + row_num_bins + 1};

    for (int i = 0; i < NUM_NEIGH; i++) {
        // Make sure bin is in bounds
        if (!(bin_nums[i] < 0 || bin_nums[i] >= total_num_bins)) {
            // skips bins that actually don't exist but map to something because of the contiguous layout
            if (!(((bin_num % row_num_bins == 0) && (i % 3 == 0)) || ((bin_num % row_num_bins == -1) && (i % 3 == -1)))) {
            starting_id[i] = bin_id[bin_nums[i]];
            bin_sizes[i] = counts[bin_nums[i]];
            num_neigh_part += bin_sizes[i];
            }
        }

        if (i != 0)
            bin_sizes_prefix[i] = bin_sizes_prefix[i-1] + bin_sizes[i-1];
    }

    // copy particles to neigh_parts
    for(int i = 0; i < NUM_NEIGH; i++) {
        if (starting_id[i] == -1 || tid >= bin_sizes[i]) continue;
        int particles_src_id = bins[starting_id[i]+tid];
        int dest_id = bin_sizes_prefix[i] + tid;
        neigh_parts_x[dest_id] = particles[particles_src_id].x;
        neigh_parts_y[dest_id] = particles[particles_src_id].y;
    }
    __syncthreads();

    if (tid >= bin_sizes[4])    //exceeds own bin
        return;

    const int part_id = bins[starting_id[4] + tid];
    particle_t my_part = particles[part_id];
    my_part.ax = my_part.ay = 0;

    // get part_id from starting index of that bin + tid
    // const int part_id = bins[starting_id[4] + tid];
    // particle_t* my_part = &particles[part_id];
    // my_part->ax = my_part->ay = 0;

    for (int j = 0; j < num_neigh_part; j++) {
        double dx = neigh_parts_x[j] - my_part.x;
        double dy = neigh_parts_y[j] - my_part.y;
        if (!(dx == 0 && dx == dy))
            apply_force_gpu_condensed(my_part, dx, dy);
    }
    particles[part_id].ax = my_part.ax;
    particles[part_id].ay = my_part.ay;
}

__global__ void move_gpu(particle_t* particles, int num_parts, double size, double bs, int row_num_bins, int* counts) {

    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    particle_t* p = &particles[tid];

    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x += p->vx * dt;
    p->y += p->vy * dt;
    // p->ax = p->ay = 0;
    //  bounce from walls
    while (p->x < 0 || p->x > size) {
        p->x = p->x < 0 ? -(p->x) : 2 * size - p->x;
        p->vx = -(p->vx);
    }
    while (p->y < 0 || p->y > size) {
        p->y = p->y < 0 ? -(p->y) : 2 * size - p->y;
        p->vy = -(p->vy);
    }
    int idx = floor(p->x / bs);
    int idy = floor(p->y / bs);
    atomicAdd(&counts[idy*row_num_bins + idx], 1);
}
__global__ void zero_count(int* counts, int total_num_bins) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < total_num_bins)
        counts[tid] = 0;
}

__global__ void count(particle_t* parts, int num_parts, double bs, int* counts, int row_num_bins) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;
    int idx = floor(parts[tid].x / bs);
    int idy = floor(parts[tid].y / bs);
    atomicAdd(&counts[idy*row_num_bins + idx], 1);
}

__global__ void dec_bin_id(int* bin_id, int* counts, int total_num_bins) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < total_num_bins)
        bin_id[tid] -= counts[tid];
}

__global__ void copy_bin_id(int* bin_id, int* bin_id_copy, int total_num_bins) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < total_num_bins)
        bin_id_copy[tid] = bin_id[tid];
}

__global__ void sort(particle_t* parts, int* bins, int num_parts, double bs, int* copy_bin_id, int row_num_bins) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    int idx = floor(parts[tid].x / bs);
    int idy = floor(parts[tid].y / bs);
    int bid = idy*row_num_bins + idx;


    int old_value = copy_bin_id[bid];
    int new_value = old_value + 1;
    while (atomicCAS(&copy_bin_id[bid], old_value, new_value) != old_value) {
        old_value = copy_bin_id[bid];
        new_value = old_value + 1;
    }

    bins[old_value] = tid;
}

void init_simulation(particle_t* parts, int num_parts, double size) {
    // You can use this space to initialize data objects that you may need
    // This function will be called once before the algorithm begins
    // parts live in GPU memory
    // Do not do any particle simulation here

    row_num_bins = ceil(size / bin_size);
    total_num_bins = row_num_bins*row_num_bins;

    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;
    blks_bins = (total_num_bins + NUM_THREADS - 1)/NUM_THREADS;
    cudaMalloc((void**)&counts_gpu, total_num_bins * sizeof(int));
    
    cudaMalloc((void **)&counts_gpu, total_num_bins * sizeof(int));
    cudaMalloc((void **)&d_output, total_num_bins * sizeof(int));

    cudaMalloc((void**)&bins_gpu, num_parts * sizeof(int));
    cudaMalloc((void**)&bin_id_gpu, total_num_bins * sizeof(int));
    zero_count<<<blks_bins, NUM_THREADS>>>(counts_gpu, total_num_bins);
    count<<<blks, NUM_THREADS>>>(parts, num_parts, bin_size, counts_gpu, row_num_bins);
    counts = (int*) malloc(total_num_bins * sizeof(int));

    exclusive_scan_time = 0;
    sort_time = 0;
    copy_bin_id_time = 0;
    compute_forces_gpu_time = 0;
    move_gpu_time = 0;
    zero_count_time = 0;
}
int time_step = 0;
void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // parts live in GPU memory
    // Rewrite this function
    thrust::device_ptr<int> dev_ptr(counts_gpu);
    thrust::device_ptr<int> dev_ptr_out(d_output);

    thrust::exclusive_scan(dev_ptr, dev_ptr+total_num_bins, dev_ptr_out);
    sort<<<blks, NUM_THREADS>>>(parts, bins_gpu, num_parts, bin_size, d_output, row_num_bins);
    dec_bin_id<<<blks_bins, NUM_THREADS>>>(d_output, counts_gpu, total_num_bins);
    compute_forces_condensed_gpu<<<total_num_bins, NUM_THREADS_FORCE>>>(parts, d_output, bins_gpu, num_parts, row_num_bins, total_num_bins, counts_gpu);
    zero_count<<<blks_bins, NUM_THREADS>>>(counts_gpu, total_num_bins);
    move_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, size, bin_size, row_num_bins, counts_gpu);
}