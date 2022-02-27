/*
 *  sad.c
 *  Created on: 29-Mar-2021
 *  Author: Goutham Bhatta
 */

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <omp.h>
#include <chrono>
#include <iostream>


//computes sad values for each block in the given image, wrt template(blk)
void findSadFrame_pragma(unsigned char* img, unsigned char* im_crop, int W, int H, int w, int h, int* out) {

	int i,j,m,n,sadValue;
	int nthrds = 4;
	
	// Error log: intialization was not correct
	int min_val[nthrds] = {INT_MAX, INT_MAX, INT_MAX, INT_MAX}; 
	int idx_i[nthrds] = {0, 0, 0, 0};
	int idx_j[nthrds] = {0, 0, 0, 0};

    omp_set_num_threads(nthrds);

	// int *sadMat = (int*)calloc(W*H, sizeof(int));

    #pragma omp parallel for private(j,m,n,sadValue)
	for(i=0; i<H-h; i++) {
		
		int tid = omp_get_thread_num();

		for(j=0; j<W-w; j++) {
			
            sadValue = 0;

            for(m=0; m<h; m++) {
                for(n=0; n<w; n++) {
                    sadValue += abs(img[(i+m)*W+(j+n)] - im_crop[m*w+n]);
                }
            }
			
			if (sadValue < min_val[tid]) {
				min_val[tid] = sadValue;
				idx_i[tid] = i;
                idx_j[tid] = j;
			}
		}
	}

	
	int minSad = min_val[0];
	int id=0;
	for(i=0; i<nthrds; i++)
	{
		// printf("%d, %d : %d\n", idx_i[i][0], idx_j[i][0], min_val[i][0]);
		if(min_val[i] < minSad) {
			minSad = min_val[i];
			id = i;
		}
	}

	out[0] = idx_i[id];
	out[1] = idx_j[id];
	out[2] = min_val[id];
	

}

//computes sad values for each block in the given image, wrt template(blk)
void findSadFrame(unsigned char* img, unsigned char* im_crop, int W, int H, int w, int h, int* out) {

	int i, j, m, n;
	int sadValue;
	int min_val = INT_MAX; int idx_i = 0, idx_j = 0;
    
    
	for(i=0; i<H-h; i++) {
		for(j=0; j<W-w; j++) {
			
            sadValue = 0;

            for(m=0; m<h; m++) {
                for(n=0; n<w; n++) {
                    sadValue += abs(img[(i+m)*W+(j+n)] - im_crop[m*w+n]);
                }
            }
			// printf("%d, ", sadValue);
			if (sadValue < min_val) {
				min_val = sadValue;
				idx_i = i;
                idx_j = j;
			}
		}
	}

	out[0] = idx_i;
	out[1] = idx_j;
	out[2] = min_val;

}

//loads the data from raw image file
unsigned char* readImg(const char* fname, int w, int h) {
	FILE* fd = fopen(fname, "rb");
	if (fd == NULL) {
		printf("Cannot open %s \n", fname);
		exit(0);
	}
	unsigned char* im = (unsigned char*) malloc(sizeof(unsigned char)*w*h);
	int bytes = fread(im, sizeof(unsigned char), w*h, fd );
	printf("No.of bytes read %d\n", bytes);
	return im;
}



int main(int argc, char** argv) {
	
	int W = 256, H = 256;
	int w = 30, h = 21;

	unsigned char*  im1 = readImg("./input/img.bin", W, H);
	unsigned char*  im2 = readImg("./input/img_crop.data", w, h);
	
	int output[3];

    std::chrono::time_point<std::chrono::system_clock> start, end;
    std::chrono::duration<double> elapsed_seconds;    
    
    start = std::chrono::system_clock::now();
    findSadFrame(im1, im2, W, H, w, h, output);
    end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    std::cout << "sad seq time: " << elapsed_seconds.count() << "s\n";	

	start = std::chrono::system_clock::now();
    findSadFrame_pragma(im1, im2, W, H, w, h, output);
    end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    std::cout << "sad pragma time: " << elapsed_seconds.count() << "s\n";

	printf("SAD in Im1 found at : %d %d, with value : %d\n", output[0], output[1], output[2]);
	return 0;
}

