#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <string.h> 
#include "pgmio.h"

#define M 256
#define N 256
#define THRESH 100

void sobel_edge_detection_serial(float input[M][N], float output[M][N]) {
     int desired_threads = 2; // Change this value as needed
    omp_set_num_threads(desired_threads);
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

void sobel_edge_detection_parallel(float input[M][N], float output[M][N]) {
    #pragma omp parallel for collapse(2)
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

int main() {
    float input[M][N];
    float output[M][N];

    char *input_filename = "image256x256.pgm";
    char *output_filename = "image-output256x25679.pgm";

    // Read input image
    pgmread(input_filename, input, M, N);

    printf("Width: %d\nHeight: %d\n", M, N);

    double start_time_serial = omp_get_wtime();

    // Sobel edge detection (Serial)
    sobel_edge_detection_serial(input, output);

    double end_time_serial = omp_get_wtime();

    printf("Serial Execution Time: %f seconds\n", end_time_serial - start_time_serial);

    // Clear output array for parallel execution
    memset(output, 0, sizeof(output));

    double start_time_parallel = omp_get_wtime();

    // Sobel edge detection (Parallel)
    sobel_edge_detection_parallel(input, output);

    double end_time_parallel = omp_get_wtime();

    printf("Parallel Execution Time: %f seconds\n", end_time_parallel - start_time_parallel);

    // Write output image
    pgmwrite(output_filename, output, M, N);
      int num_threads = omp_get_max_threads();
      printf("num_threads = %d",num_threads);

    return 0;
}
