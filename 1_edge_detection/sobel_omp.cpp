#include <stdlib.h>
#include <stdio.h>
#include <chrono>
#include <iostream>
#include <omp.h>
#include <math.h>

// #define DEBUG_PRINT

char sobel_h[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
char sobel_v[9] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
int filter_sz = 3;

void sobel_parallel_pragma(unsigned char *img, int rows, int cols, unsigned char *G_x)
{
    int grad_x, grad_y, grad;
    int k = filter_sz / 2;
    omp_set_num_threads(4);

    
    #pragma omp parallel
    {

        int i;

        #pragma omp for private(grad_x, grad_y, grad)
        for (i = k; i < rows; i++)
        {
            int j, m, n;
        
            for (j = k; j < cols; j++)
            {
                grad_x = 0, grad_y = 0;
                for (m = -k; m <= k; m++)
                {
                    for (n = -k; n <= k; n++)
                    {
                        grad_x += img[(i + m) * cols + (j + n)] * sobel_h[(m + k) * filter_sz + (n + k)];
                        grad_y += img[(i + m) * cols + (j + n)] * sobel_v[(m + k) * filter_sz + (n + k)];
                    }
                }
                
                grad = sqrt(grad_x*grad_x + grad_y*grad_y);

                if (grad > 255)
                    G_x[i * cols + j] = 255;
                else
                    G_x[i * cols + j] = grad;
            }
        }
    }

}

void sobel_parallel_spmd(unsigned char *img, int rows, int cols, unsigned char *G_x)
{

    int k = filter_sz / 2;
    int npart_c = 2;
    int npart_r = 2;
    int nproc = npart_r * npart_c;
    int blk_sz_r = rows / npart_r;
    int blk_sz_c = cols / npart_c;
    int nthreads;

    omp_set_num_threads(nproc);
    #pragma omp parallel
    {
        int i, j, m, n;

        int pid = omp_get_thread_num();

        if (pid == 0)
        {
            nthreads = omp_get_num_threads();
        }
        
        int blk_col = pid % npart_c;
        int blk_row = (pid - blk_col) / npart_r;

        int r_strt = blk_row * blk_sz_r;
        int r_end = r_strt + blk_sz_r;

        int c_strt = blk_col * blk_sz_c;
        int c_end = c_strt + blk_sz_c;

        int grad_x, grad_y, grad;

        #ifdef DEBUG_PRINT
            printf("row(%d): %d %d \n", pid, r_strt, r_end);
            printf("col(%d): %d %d \n", pid, c_strt, c_end);
        #endif

        for (i = r_strt + k; i < r_end - k; i++)
        {
            for (j = c_strt + k; j < c_end - k; j++)
            {
                grad_x = 0, grad_y = 0;
                for (m = -k; m <= k; m++)
                {
                    for (n = -k; n <= k; n++)
                    {
                        #ifdef DEBUG_PRINT
                            printf("%d: %d, %d \n", pid, (i + m)*cols + (j + n),  (m + k) * filter_sz + (n + k));
                        #endif

                        grad_x += img[(i + m) * cols + (j + n)] * sobel_h[(m + k) * filter_sz + (n + k)];
                        grad_y += img[(i + m) * cols + (j + n)] * sobel_v[(m + k) * filter_sz + (n + k)];
                    }
                }

                
                grad = sqrt(grad_x*grad_x + grad_y*grad_y);
                
                if (grad > 255)
                    G_x[i * cols + j] = 255;
                else
                    G_x[i * cols + j] = grad;
            }
        }
    }

    if (nthreads < nproc)
    {
        printf("Required no.of threads cannot be allocated. \n Please select smaller partitions.\n");
    }
}

void sobel_sequential(unsigned char *img, int rows, int cols, unsigned char *G_x)
{

    int i, j, m, n;
    int k = filter_sz / 2;
    int grad_x, grad_y, grad;

    for (i = k; i < rows; i++)
    {
        for (j = k; j < cols; j++)
        {
            grad_x = 0, grad_y = 0;
            for (m = -k; m <= k; m++)
            {
                for (n = -k; n <= k; n++)
                {
                    grad_x += img[(i + m) * cols + (j + n)] * sobel_h[(m + k) * filter_sz + (n + k)];
                    grad_y += img[(i + m) * cols + (j + n)] * sobel_v[(m + k) * filter_sz + (n + k)];
                }
            }
            
            grad = sqrt(grad_x*grad_x + grad_y*grad_y);
            if (grad > 255)
                G_x[i * cols + j] = 255;
            else
                G_x[i * cols + j] = grad;
        }
    }
    
}

// reads from raw image file and load into memory
unsigned char * readImg(const char *fname, int r, int c)
{
    FILE *fd = fopen(fname, "rb");
    if (fd == NULL)
    {
        printf("Unable to open %c", *fname);
        return NULL;
    }

    unsigned char *buff = (unsigned char *)malloc(sizeof(unsigned char) * r * c);
    int bytes_read = fread(buff, sizeof(unsigned char), r * c, fd);
    printf("No.of bytes read: %d\n", bytes_read);
    return buff;
}


// writes from memory into a raw image file
void writeImg(const char *fname, unsigned char *im, int r, int c)
{

    FILE *fd = fopen(fname, "wb");
    if (fd == NULL)
    {
        printf("Unable to open %c", *fname);
        exit(0);
    }
    fwrite(im, r * c, sizeof(unsigned char), fd);
}

int main(int argc, char* argv[])
{

    if(argc < 4) {
        printf("Invalid arguments \n");
        return 0;
    }
    char* filename = argv[1];
    int rows = atoi(argv[2]);
    int cols = atoi(argv[3]);

    unsigned char *img = readImg(filename, rows, cols);
    unsigned char *G_x = (unsigned char *)calloc(rows * cols, sizeof(unsigned char));

    std::chrono::time_point<std::chrono::system_clock> start, end;
    std::chrono::duration<double> elapsed_seconds;    

    start = std::chrono::system_clock::now();
    sobel_sequential(img, rows, cols, G_x);
    end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    std::cout << "sobel_sequential time: " << elapsed_seconds.count() << "s\n";

    start = std::chrono::system_clock::now();
    sobel_parallel_pragma(img, rows, cols, G_x);
    end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    std::cout << "sobel_parallel_pragma time: " << elapsed_seconds.count() << "s\n";

    start = std::chrono::system_clock::now();
    sobel_parallel_spmd(img, rows, cols, G_x);
    end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    std::cout << "sobel_parallel_spmd time: " << elapsed_seconds.count() << "s\n";

    writeImg("edge_img.bin", G_x, rows, cols);

}
