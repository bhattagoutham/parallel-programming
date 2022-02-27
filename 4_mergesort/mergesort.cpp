#include <iostream>
#include <omp.h>
#include <string.h>
#include <stdlib.h>
#include <chrono>

int arr_size = 100000;
int nthrds = 6;

void merge(int* arr, int low, int mid, int high) {


    int n_elems = high - low + 1;    
    int *aux_arr = (int*) malloc(n_elems*sizeof(int));
    int i = low, j = mid+1, k = 0;

    while(i < mid+1 && j <= high) {
        
        if(arr[i] <= arr[j]) {
            aux_arr[k] = arr[i]; i++;
        } else {
            aux_arr[k] = arr[j]; j++;
        } 
        
        k++;
    }

    while (i < mid+1) {
        aux_arr[k] = arr[i]; k++; i++;
    }

    while(j <= high) {
        aux_arr[k] = arr[j]; j++; k++;
    }

    memcpy(&arr[low], aux_arr, sizeof(int)*n_elems);

    free(aux_arr);
    

}


void mergesort(int *arr, int low, int high) {

    if(low >= high) {
        return;
    }

    int mid = (low+high)/2;
    mergesort(arr, low, mid);
    mergesort(arr, mid+1, high);
    merge(arr, low, mid, high);

}

void mergesort_llel(int *arr, int low, int high) {

    if(low >= high) {
        return;
    }

    int mid = (low+high)/2;
    int size = high - low + 1;

    #pragma omp task shared(arr) if (size > arr_size/nthrds)
    mergesort_llel(arr, low, mid);
    
    #pragma omp task shared(arr) if (size > arr_size/nthrds)
    mergesort_llel(arr, mid+1, high);

    #pragma omp taskwait
    merge(arr, low, mid, high);

}

void disp(int* arr, int size) {
    int i;
    for(i=0;i<size;i++)
        printf("%d, ", arr[i]);
    printf("\n");
}

int main() {

    int size = arr_size;
    int arr[size];
    int i;

    /* Initialize the array in reverse order*/
    for(i=size; i>0; i--) {
        arr[size-i] = i;
    }
    

    std::chrono::time_point<std::chrono::system_clock> start, end;
    std::chrono::duration<double> elapsed_seconds;    
    
    start = std::chrono::system_clock::now();
    mergesort(arr, 0, size-1);
    end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    std::cout << "mergesort seq time: " << elapsed_seconds.count() << "s\n";

    
    omp_set_num_threads(nthrds);
    
    start = std::chrono::system_clock::now();
    #pragma omp parallel 
    {
        #pragma omp single 
        {
            mergesort_llel(arr, 0, size-1);
        }
    }

    end = std::chrono::system_clock::now();
    elapsed_seconds = end - start;
    std::cout << "mergesort llel time: " << elapsed_seconds.count() << "s\n";

    // disp(arr, size);


}
