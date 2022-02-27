/*
 *  interpolation.c
 *  Created on: 05-Apr-2021
 *  Author: Goutham Bhatta
 */

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <chrono>
#include <omp.h>

//writes the data into a raw image file
void writeImg(const char* fname, unsigned char* im, int r, int c) {

	FILE* fd = fopen(fname, "wb");
	if (fd == NULL) {
		printf("Unable to open %c", *fname);
		return;
	}
	int nbytes = fwrite(im, r*c, sizeof(unsigned char), fd);
	printf("No.of bytes written: %d\n", nbytes);
	
}


//rotates image clockwise by 90 deg
unsigned char* rotateImc90_ll(unsigned char* im, int r, int c) {
//	printf("Rotating image anti-clockwise 90 deg\n");
	int i, j;
	unsigned char* rotIm = (unsigned char*)malloc(r*c*sizeof(unsigned char));
	
	#pragma omp parallel for private(i,j)
	for(i=0;i<r;i++) {
		for(j=0; j<c; j++) {
			rotIm[(j*r)+(c-i)] = im[i*c+j];
		}
	}
	return rotIm;
}

//rotates image anti-clockwise by 90 deg
unsigned char* rotateImac90_ll(unsigned char* im, int r, int c) {
//	printf("Rotating image clockwise 90 deg\n");
	int i, j;
	unsigned char* rotIm = (unsigned char*)malloc(r*c*sizeof(unsigned char));
	
	#pragma omp parallel for private(i,j)
	for(i=0;i<r;i++) {
		for(j=0; j<c; j++) {
			rotIm[(r-j)*c+i] = im[i*c+j];
		}
	}
	return rotIm;
}

unsigned char* interp_ll(unsigned char* im, int r, int c) {

//	printf("Interpolates along x-axis without intrinsics");
	unsigned char* res = (unsigned char*)calloc(r*2*c, sizeof(unsigned char));
	int i,p,j,k;
	
	#pragma omp parallel for private(i,p,j,k)
	for(p=0; p<r; p++) {
		for(i=0, j=0, k=1; i<c; i++, j+=2, k+=2) {
			res[p*2*c+j] = im[p*c+i];
			res[p*2*c+k] = (im[p*c+i]+im[p*c+i+1])/2;
		}
	}

	return res;

}

unsigned char* rotateImc90(unsigned char* im, int r, int c) {
//	printf("Rotating image anti-clockwise 90 deg\n");
	int i, j;
	unsigned char* rotIm = (unsigned char*)malloc(r*c*sizeof(unsigned char));
	
	
	for(i=0;i<r;i++) {
		for(j=0; j<c; j++) {
			rotIm[(j*r)+(c-i)] = im[i*c+j];
		}
	}
	return rotIm;
}

unsigned char* rotateImac90(unsigned char* im, int r, int c) {
//	printf("Rotating image clockwise 90 deg\n");
	int i, j;
	unsigned char* rotIm = (unsigned char*)malloc(r*c*sizeof(unsigned char));
	
	for(i=0;i<r;i++) {
		for(j=0; j<c; j++) {
			rotIm[(r-j)*c+i] = im[i*c+j];
		}
	}
	return rotIm;
}

// without intrinsics (plain-interp)
unsigned char* interp(unsigned char* im, int r, int c) {

//	printf("Interpolates along x-axis without intrinsics");
	unsigned char* res = (unsigned char*)calloc(r*2*c, sizeof(unsigned char));
	int i,p,j,k;

	for(p=0; p<r; p++) {
		for(i=0, j=0, k=1; i<c; i++, j+=2, k+=2) {
			res[p*2*c+j] = im[p*c+i];
			res[p*2*c+k] = (im[p*c+i]+im[p*c+i+1])/2;
		}
	}

	return res;

}

//loads the data from raw image file
unsigned char* readImg(const char* fname, int r, int c) {
	FILE* fd = fopen(fname, "rb");
	if (fd == NULL) {
		printf("Unable to open %c", *fname);
		return NULL;
	}

	unsigned char* buff = (unsigned char*)malloc(sizeof(unsigned char)*r*c);
	int bytes_read = fread(buff, sizeof(unsigned char),  r*c,fd );
	printf("No.of bytes read: %d\n", bytes_read);
	return buff;
}



int main(void) {
	
	unsigned char *im, *res;
	
	std::chrono::time_point<std::chrono::system_clock> start, end;
    std::chrono::duration<double> elapsed_seconds;    

	omp_set_num_threads(4);

	//	read image files
	im = readImg("./input_and_output/img_256x256.raw", 256, 256);

	start = std::chrono::system_clock::now();
	res = interp(im, 256, 256);
	res = rotateImc90(res, 256, 512);
	res = interp(res, 512, 256);
	res = rotateImac90(res, 512, 512);
	end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    std::cout << "interpolation_seq time: " << elapsed_seconds.count() << "s\n";


	start = std::chrono::system_clock::now();
	res = interp_ll(im, 256, 256);
	res = rotateImc90_ll(res, 256, 512);
	res = interp_ll(res, 512, 256);
	res = rotateImac90_ll(res, 512, 512);
	end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    std::cout << "interpolation_llel time: " << elapsed_seconds.count() << "s\n";

	writeImg("./input_and_output/result_512x512.raw", res, 512, 512);
	return 0;
}
