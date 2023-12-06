#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <string.h> 
#include "pgmio.h"

#define M 256
#define N 256
#define THRESH 100

void sobel_edge_detection_parallel(float input[M][N], float output[M][N]) {
    #pragma omp parallel
    {
        #pragma omp for collapse(2)
        for (int i = 1; i < M - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                float gradient_h = (-1.0 * input[i - 1][j - 1]) + (1.0 * input[i + 1][j - 1]) +
                                   (-2.0 * input[i - 1][j]) + (2.0 * input[i + 1][j]) +
                                   (-1.0 * input[i - 1][j + 1]) + (1.0 * input[i + 1][j + 1]);

                float gradient_v = (-1.0 * input[i - 1][j - 1]) + (-2.0 * input[i][j - 1]) +
                                   (-1.0 * input[i + 1][j - 1]) + (1.0 * input[i - 1][j + 1]) +
                                   (2.0 * input[i][j + 1]) + (1.0 * input[i + 1][j + 1]);

                float gradient = sqrt((gradient_h * gradient_h) + (gradient_v * gradient_v));

                if (gradient < THRESH) {
                    output[i][j] = 0;
                } else {
                    output[i][j] = 255;
                }
            }
        }
    }
}

int main() {
    float input[M][N];
    float output[M][N];

    char *input_filename = "image256x256.pgm";
    char *output_filename = "image-output256x25679.pgm";

    // Read input image
    pgmread(input_filename, input, M, N);

    printf("Width: %d\nHeight: %d\n", M, N);

    double start_time_total = omp_get_wtime();

    // Clear output array for parallel execution
    memset(output, 0, sizeof(output));

    double start_time_parallel = omp_get_wtime();

    // Sobel edge detection (Parallel)
    sobel_edge_detection_parallel(input, output);

    double end_time_parallel = omp_get_wtime();

    printf("Parallel Execution Time: %f seconds\n", end_time_parallel - start_time_parallel);

    // Write output image
    pgmwrite(output_filename, output, M, N);

    double end_time_total = omp_get_wtime();

    printf("Total Execution Time: %f seconds\n", end_time_total - start_time_total);

    // Get the number of threads
    int num_threads;
    #pragma omp parallel
    {
        #pragma omp master
        num_threads = omp_get_num_threads();
    }

    printf("Number of threads: %d\n", num_threads);

    return 0;
}
