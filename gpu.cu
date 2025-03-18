#include "common.h"
#include <cuda.h>

#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/scan.h>
#include <thrust/copy.h>
#include <iostream>

#include <algorithm>

#define NUM_THREADS 256

// TODO: fix corner cases
// TODO: bucket sizes must start with 0

// Put any static global variables here that you will use throughout the simulation.
int blks;

static int grid_side_length;
int num_buckets;
int* bucket_sizes;             // show where each bucket starts and ends
int* bucket_index;             // used to place each particle at a unique index in each bucket

int cnt;                       // which bucket we are populating
particle_t* particles_in_buckets;       // need two buckets to move from one to the other
                                        // effectively a 2d array, we will use the first num_parts for one copy,
                                        // and the second num_parts for a second copy

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

__global__ void compute_forces_gpu(particle_t* particles_in_buckets, int num_parts, double size, int grid_side_length, int cnt, int* bucket_sizes) {
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    int offset = cnt * num_parts;

    for (int i = tid; i < num_parts; i += stride) {
        particles_in_buckets[offset + i].ax = particles_in_buckets[offset + i].ay = 0;

        // find nearby buckets
        int bucket_row, bucket_col;
        particle_to_bucket(particles_in_buckets[offset + i], size, bucket_row, bucket_col, grid_side_length);
        
        for (int bx = std::max(bucket_row-1, 0); bx <= std::min(bucket_row+1, grid_side_length-1); bx ++) {
            for (int by = std::max(bucket_col-1, 0); by <= std::min(bucket_col+1, grid_side_length-1); by ++) {
                
                int neighbor_bucket = bx * grid_side_length + by;
                
                int start_index, end_index;
                if (neighbor_bucket == 0) {
                    start_index = 0;
                } else {
                    start_index = bucket_sizes[neighbor_bucket-1];
                }
                end_index = bucket_sizes[neighbor_bucket];

                for (int j = start_index; j < end_index; ++ j) {
                    apply_force_gpu(particles_in_buckets[offset + i], particles_in_buckets[offset + j]);
                }
            }
        }
    }
}

__global__ void move_gpu(particle_t* particles_in_buckets, int num_parts, double size) {
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < num_parts; i += stride) {

        particle_t* p = &particles_in_buckets[i];
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

__global__ void compute_bucket_sizes(int num_parts, particle_t* particles_in_buckets, int cnt, int* bucket_sizes, double size, int grid_side_length) { 
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < num_parts; i += stride) {
        // compute which bucket you are in
        int bucket_row, bucket_col;
        particle_to_bucket(particles_in_buckets[cnt*num_parts + i], size, bucket_row, bucket_col, grid_side_length);
        int current_bucket = bucket_row * grid_side_length + bucket_col;

        // increase the size of this bucket (atomically)
        // the second parameter is used if we want to take modulo something
        auto old_value = atomicAdd(bucket_sizes + current_bucket, 1);
    }
}

__global__ void rebucket_particles(int num_parts, particle_t* particles_in_buckets, int cnt, int* bucket_sizes, double size, int grid_side_length, int* bucket_index) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < num_parts; i += stride) {
        // compute which bucket you are in
        int bucket_row, bucket_col;
        particle_to_bucket(particles_in_buckets[cnt*num_parts + i], size, bucket_row, bucket_col, grid_side_length);
        int current_bucket = bucket_row * grid_side_length + bucket_col;

        // obtain your index in the bucket
        auto particle_index_in_bucket = atomicAdd(bucket_index + current_bucket, 1);

        // move to the new bucket
        particles_in_buckets[(1-cnt)*num_parts + bucket_sizes[current_bucket] + particle_index_in_bucket] = particles_in_buckets[cnt*num_parts + i];
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

    grid_side_length = std::min(int(size/(2*cutoff)), int(4 * sqrt(num_parts))); // Number of rows/columns in our grid of buckets
    num_buckets = grid_side_length*grid_side_length;

    // 1. Allocate space on the GPU
    cnt = 0;
    cudaMalloc((void **)&particles_in_buckets, 2*num_parts*sizeof(particle_t));

    cudaMalloc((void **)&bucket_sizes, num_buckets*sizeof(int));
    cudaMalloc((void **)&bucket_index, num_buckets*sizeof(int));
    // 2. Move particles to the GPU
    cudaMemcpy(particles_in_buckets, parts, num_parts*sizeof(particle_t), cudaMemcpyHostToDevice);
    // 3. Initialize the buckets
    // zero out bucket sizes
    cudaMemset(bucket_sizes, 0, num_buckets * sizeof(int));
    // 3a. Compute bucket sizes
    compute_bucket_sizes<<<blks, NUM_THREADS>>>(num_parts, particles_in_buckets, cnt, bucket_sizes, size, grid_side_length);
    // 3b. Inclusive scan for indices
    thrust::inclusive_scan(thrust::device_pointer_cast(bucket_sizes), 
                            thrust::device_pointer_cast(bucket_sizes + num_buckets), 
                            thrust::device_pointer_cast(bucket_sizes));
    // 3c. Zero out bucket index
    cudaMemset(bucket_index, 0, num_buckets * sizeof(int));
    // 3d. Move particles to the correct bucket
    rebucket_particles<<<blks, NUM_THREADS>>>(num_parts, particles_in_buckets, cnt, bucket_sizes, size, grid_side_length, bucket_index);
    // set cnt to 1-cnt
    cnt = 1 - cnt;
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // parts live in GPU memory
    // Rewrite this function

    // Step 1. Compute forces
    compute_forces_gpu<<<blks, NUM_THREADS>>>(particles_in_buckets, num_parts, size, grid_side_length, cnt, bucket_sizes);

    // Step 2. Move particles
    move_gpu<<<blks, NUM_THREADS>>>(particles_in_buckets, num_parts, size);

    // Step 3. Compute new bucket sizes
    // zero out current sizes
    cudaMemset(bucket_sizes, 0, num_buckets * sizeof(int));
    // compute bucket sizes
    compute_bucket_sizes<<<blks, NUM_THREADS>>>(num_parts, particles_in_buckets, cnt, bucket_sizes, grid_side_length);
    // inclusive scan
    thrust::inclusive_scan(thrust::device_pointer_cast(bucket_sizes), 
                            thrust::device_pointer_cast(bucket_sizes + num_buckets), 
                            thrust::device_pointer_cast(bucket_sizes));

    // 3c. Zero out bucket index
    cudaMemset(bucket_index, 0, num_buckets * sizeof(int));
    // Step 4. Rebucket particles
    rebucket_particles<<<blks, NUM_THREADS>>>(num_parts, particles_in_buckets, cnt, bucket_sizes, size, grid_side_length, bucket_index);
    // set cnt to 1-cnt
    cnt = 1 - cnt;
}
