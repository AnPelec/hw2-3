#include "common.h"
#include <cuda.h>

#include <cuda_runtime.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/scan.h>
#include <thrust/copy.h>
#include <iostream>

#include <algorithm>

#define NUM_THREADS 256

// Put any static global variables here that you will use throughout the simulation.
int blks;

int grid_side_length;       // number of buckets on each side
int num_buckets;            // total number of buckets

int* bucket_sizes;          // show where each bucket starts and ends
int* bucket_index;          // used to place each particle at a unique index in each bucket

int* particles_in_buckets;  // stores the particle indices according to their buckets
                            // has size 2 * num_parts since we need extra space for the rebucketing
int cnt;                    // whether we are using the first or the second half of particles_in_buckets

// This function computes the bucket of a particle
__device__ void particle_to_bucket(particle_t particle, double size, int grid_side_length, int &bx, int &by) {
    bx = (particle.x * grid_side_length)/size;
    by = (particle.y * grid_side_length)/size;

    // Sanity checks
    assert(0 <= bx && bx < grid_side_length);
    assert(0 <= by && by < grid_side_length);
}

__device__ void apply_force_gpu(particle_t& particle, particle_t& neighbor) {
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    if (r2 > cutoff * cutoff)
        return;
    // r2 = fmax( r2, min_r*min_r );
    r2 = (r2 > min_r * min_r) ? r2 : min_r * min_r;
    double r = sqrt(r2);

    //
    //  very simple short-range repulsive force
    //
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

__global__ void compute_forces_gpu(particle_t* parts, int num_parts, int* particles_in_buckets, double size, int grid_side_length, int cnt, int* bucket_sizes) {
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < num_parts; i += stride) {
        int particle_index = particles_in_buckets[cnt * num_parts + i];

        parts[particle_index].ax = parts[particle_index].ay = 0;

        // find nearby buckets
        int bucket_row, bucket_col;
        particle_to_bucket(parts[particle_index], size, grid_side_length, bucket_row, bucket_col);
        
        for (int bx = max(bucket_row-1, 0); bx <= min(bucket_row+1, grid_side_length-1); bx ++) {
            for (int by = max(bucket_col-1, 0); by <= min(bucket_col+1, grid_side_length-1); by ++) {
                
                int neighbor_bucket = bx * grid_side_length + by;
                
                int start_index, end_index;
                if (neighbor_bucket == 0) {
                    start_index = 0;
                } else {
                    start_index = bucket_sizes[neighbor_bucket-1];
                }
                end_index = bucket_sizes[neighbor_bucket];

                for (int j = start_index; j < end_index; ++ j) {
                    int neighbor_index = particles_in_buckets[cnt*num_parts + j];

                    apply_force_gpu(parts[particle_index], parts[neighbor_index]);
                }
            }
        }
    }
}

__global__ void move_gpu(particle_t* parts, int num_parts, int* particles_in_buckets, double size, int cnt) {
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < num_parts; i += stride) {
        int particle_index = particles_in_buckets[cnt*num_parts + i];

        particle_t* p = &parts[particle_index];
        //
        //  slightly simplified Velocity Verlet integration
        //  conserves energy better than explicit Euler method
        //
        p->vx += p->ax * dt;
        p->vy += p->ay * dt;
        p->x += p->vx * dt;
        p->y += p->vy * dt;

        //
        //  bounce from walls
        //
        while (p->x < 0 || p->x > size) {
            p->x = p->x < 0 ? -(p->x) : 2 * size - p->x;
            p->vx = -(p->vx);
        }
        while (p->y < 0 || p->y > size) {
            p->y = p->y < 0 ? -(p->y) : 2 * size - p->y;
            p->vy = -(p->vy);
        }
    }
}

/*
    HELPER FUNCTIONS
*/

__global__ void compute_bucket_sizes(particle_t* parts, int num_parts, int* particles_in_buckets, int cnt, int* bucket_sizes, double size, int grid_side_length) { 
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < num_parts; i += stride) {
        int particle_index = particles_in_buckets[cnt*num_parts + i];
        if (particle_index == -1) { // for the first
            particle_index = i;
        }

        // compute which bucket you are in
        int bucket_row, bucket_col;
        particle_to_bucket(parts[particle_index], size, grid_side_length, bucket_row, bucket_col);
        int current_bucket = bucket_row * grid_side_length + bucket_col;

        // increase the size of this bucket (atomically)
        // the second parameter is used if we want to take modulo something
        auto old_value = atomicAdd(bucket_sizes + current_bucket, 1);
    }
}

__global__ void rebucket_particles(particle_t* parts, int num_parts, int* particles_in_buckets, int cnt, int* bucket_sizes, double size, int grid_side_length, int* bucket_index) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < num_parts; i += stride) {
        int particle_index = particles_in_buckets[cnt*num_parts + i];
        if (particle_index == -1) { // for the first time
            particle_index = i;
        }

        // compute which bucket you are in
        int bucket_row, bucket_col;
        particle_to_bucket(parts[particle_index], size, grid_side_length, bucket_row, bucket_col);
        int current_bucket = bucket_row * grid_side_length + bucket_col;

        // obtain your index in the bucket
        auto particle_index_in_bucket = atomicAdd(bucket_index + current_bucket, 1);

        int bucket_offset = 0;
        if (current_bucket > 0) {
            bucket_offset = bucket_sizes[current_bucket-1];
        }

        // move to the new bucket
        particles_in_buckets[(1-cnt)*num_parts + bucket_offset + particle_index_in_bucket] = particle_index;
    }
} 


/*
    MAIN SIMULATION
*/

void init_simulation(particle_t* parts, int num_parts, double size) {
    // You can use this space to initialize data objects that you may need
    // This function will be called once before the algorithm begins
    // parts live in GPU memory
    // Do not do any particle simulation here

    // I think we can adjust this parameter
    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;

    grid_side_length = std::min(int(size/(2*cutoff)), int(sqrt(num_parts))); // Number of rows/columns in our grid of buckets
    num_buckets = grid_side_length*grid_side_length;

    // 1. Allocate space on the GPU
    cnt = 0;
    cudaMalloc((void **)&particles_in_buckets, 2 * num_parts * sizeof(int));
    cudaMalloc((void **)&bucket_sizes, num_buckets * sizeof(int));
    cudaMalloc((void **)&bucket_index, num_buckets * sizeof(int));

    // 2. Move particles to the GPU (the first time they point to something empty)
    cudaMemset(particles_in_buckets, -1, num_parts * sizeof(int));

    // 3. Initialize the buckets
    cudaMemset(bucket_sizes, 0, num_buckets * sizeof(int)); // zero out bucket sizes
    // 3a. Compute bucket sizes
    compute_bucket_sizes<<<blks, NUM_THREADS>>>(parts, num_parts, particles_in_buckets, cnt, bucket_sizes, size, grid_side_length);
    // 3b. Inclusive scan for indices
    thrust::device_ptr<int> bucket_sizes_ptr = thrust::device_pointer_cast(bucket_sizes);
    thrust::inclusive_scan(bucket_sizes_ptr,
                           bucket_sizes_ptr + num_buckets, 
                           bucket_sizes_ptr);
    // 3c. Zero out bucket index
    cudaMemset(bucket_index, 0, num_buckets * sizeof(int));
    // 3d. Move particles to the correct bucket
    rebucket_particles<<<blks, NUM_THREADS>>>(parts, num_parts, particles_in_buckets, cnt, bucket_sizes, size, grid_side_length, bucket_index);
    // set cnt to 1-cnt
    cnt = 1 - cnt;
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // parts live in GPU memory
    // Rewrite this function

    // Step 1. Compute forces
    compute_forces_gpu<<<blks, NUM_THREADS>>>(
        parts, num_parts, particles_in_buckets, size, grid_side_length, cnt, bucket_sizes
    );

    // Step 2. Move particles
    move_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, particles_in_buckets, size, cnt);

    // Step 3. Compute new bucket sizes
    cudaMemset(bucket_sizes, 0, num_buckets * sizeof(int)); // zero out current sizes
    compute_bucket_sizes<<<blks, NUM_THREADS>>>(parts, num_parts, particles_in_buckets, cnt, bucket_sizes, size, grid_side_length); // actual bucket size computation
    // inclusive scan
    thrust::device_ptr<int> bucket_sizes_ptr = thrust::device_pointer_cast(bucket_sizes);
    thrust::inclusive_scan(bucket_sizes_ptr, 
                           bucket_sizes_ptr + num_buckets, 
                           bucket_sizes_ptr);
    // 3c. Zero out bucket index
    cudaMemset(bucket_index, 0, num_buckets * sizeof(int));

    // Step 4. Rebucket particles
    rebucket_particles<<<blks, NUM_THREADS>>>(parts, num_parts, particles_in_buckets, cnt, bucket_sizes, size, grid_side_length, bucket_index);

    // set cnt to 1-cnt
    cnt = 1 - cnt;
}
