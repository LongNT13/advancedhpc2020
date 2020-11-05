#include <stdio.h>
#include <include/labwork.h>
#include <cuda_runtime_api.h>
#include <omp.h>

#define ACTIVE_THREADS 4

int main(int argc, char **argv) {
    printf("USTH ICT Master 2018, Advanced Programming for HPC.\n");
    if (argc < 2) {
        printf("Usage: labwork <lwNum> <inputImage>\n");
        printf("   lwNum        labwork number\n");
        printf("   inputImage   the input file name, in JPEG format\n");
        return 0;
    }

    int lwNum = atoi(argv[1]);
    std::string inputFilename;

    // pre-initialize CUDA to avoid incorrect profiling
    printf("Warming up...\n");
    char *temp;
    cudaMalloc(&temp, 1024);

    Labwork labwork;
    if (lwNum != 2 ) {
        inputFilename = std::string(argv[2]);
        labwork.loadInputImage(inputFilename);
    }

    printf("Starting labwork %d\n", lwNum);
    Timer timer;
    timer.start();
    switch (lwNum) {
        case 1:
            labwork.labwork1_CPU();
            labwork.saveOutputImage("labwork2-cpu-out.jpg");
            printf("labwork 1 CPU ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            timer.start();
            labwork.labwork1_OpenMP();
            // printf("labwork 1 CPU OpenMP ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork2-openmp-out.jpg");
            break;
        case 2:
            labwork.labwork2_GPU();
            break;
        case 3:    
            labwork.labwork3_GPU();
            labwork.saveOutputImage("labwork3-gpu-out.jpg");
            break;
        case 4:
            labwork.labwork4_GPU();
            labwork.saveOutputImage("labwork4-gpu-out.jpg");
            break;
        case 5:
            labwork.labwork5_CPU();
            labwork.saveOutputImage("labwork5-cpu-out.jpg");
            labwork.labwork5_GPU(FALSE);
            labwork.saveOutputImage("labwork5-gpu-out.jpg");
            break;
        case 6:
            labwork.labwork6_GPU();
            labwork.saveOutputImage("labwork6-gpu-out.jpg");
            break;
        case 7:
            labwork.labwork7_GPU();
            printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork7-gpu-out.jpg");
            break;
        case 8:
            labwork.labwork8_GPU();
            printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork8-gpu-out.jpg");
            break;
        case 9:
            labwork.labwork9_GPU();
            printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork9-gpu-out.jpg");
            break;
        case 10:
            labwork.labwork10_GPU();
            printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork10-gpu-out.jpg");
            break;
    }
    printf("labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
}

void Labwork::loadInputImage(std::string inputFileName) {
    inputImage = jpegLoader.load(inputFileName);
}

void Labwork::saveOutputImage(std::string outputFileName) {
    jpegLoader.save(outputFileName, outputImage, inputImage->width, inputImage->height, 90);
}

void Labwork::labwork1_CPU() {
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    for (int j = 0; j < 100; j++) {     // let's do it 100 times, otherwise it's too fast!
        for (int i = 0; i < pixelCount; i++) {
            outputImage[i * 3] = (char) (((int) inputImage->buffer[i * 3] + (int) inputImage->buffer[i * 3 + 1] +
                                          (int) inputImage->buffer[i * 3 + 2]) / 3);
            outputImage[i * 3 + 1] = outputImage[i * 3];
            outputImage[i * 3 + 2] = outputImage[i * 3];
        }
    }
}

void Labwork::labwork1_OpenMP() {
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    
    double previousTime = 0;
    Timer timer;
    for(int noThreads = 0; noThreads < 500; noThreads++){
        omp_set_num_threads(noThreads);
        timer.start();
        for (int j = 0; j < 100; j++) {     // let's do it 100 times, otherwise it's too fast!
            #pragma omp parallel for
            for (int i = 0; i < pixelCount; i++) {
                outputImage[i * 3] = (char) (((int) inputImage->buffer[i * 3] + (int) inputImage->buffer[i * 3 + 1] +
                                              (int) inputImage->buffer[i * 3 + 2]) / 3);
                outputImage[i * 3 + 1] = outputImage[i * 3];
                outputImage[i * 3 + 2] = outputImage[i * 3];
            }
        }
        double currentTime = timer.getElapsedTimeInMilliSec();
        printf("labwork 1 CPU OpenMP with %d threads ellapsed %.1fms, currentThread : %f previousThread\n", noThreads, currentTime, (currentTime/previousTime));
        previousTime = currentTime;
    }

}

int getSPcores(cudaDeviceProp devProp) {
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major) {
        case 2: // Fermi
            if (devProp.minor == 1) cores = mp * 48;
            else cores = mp * 32;
            break;
        case 3: // Kepler
            cores = mp * 192;
            break;
        case 5: // Maxwell
            cores = mp * 128;
            break;
        case 6: // Pascal
            if (devProp.minor == 1) cores = mp * 128;
            else if (devProp.minor == 0) cores = mp * 64;
            else printf("Unknown device type\n");
            break;
        default:
            printf("Unknown device type\n");
            break;
    }
    return cores;
}

void Labwork::labwork2_GPU() {
    int nDevices = 0;
    // get all devices
    cudaGetDeviceCount(&nDevices);
    printf("Number total of GPU : %d\n\n", nDevices);
    for (int i = 0; i < nDevices; i++){
        // get informations from individual device
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        // something more here
        printf("Device name : %s\n", prop.name);
        // Core info: clock rate, core counts, multiprocessor count, wrap size
        printf("Core clock rate : %d\n", prop.clockRate);
        printf("Core Count : %d\n", getSPcores(prop));
        printf("Core multiprocessor count : %d\n", prop.multiProcessorCount);
        printf("Core wrap size : %d\n", prop.warpSize);

        // Memory info: clock rate, bus width and [optional] bandwidth
        printf("Memory clock rate : %d\n", prop.memoryClockRate);
        printf("Memory bus width : %d\n", prop.memoryBusWidth);
        printf("Memory bandwidth : %.0f\n", 2.0*prop.memoryClockRate*prop.memoryBusWidth/8);
        
        printf("\n");
    }
}

__global__ void grayScale(uchar3 *input, uchar3 *output){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    output[tid].x = (input[tid].x + input[tid].y +
    input[tid].z) / 3;
    output[tid].z = output[tid].y = output[tid].x;
}

void charToUchar3(char *input, uchar3 *output, int pixelCount){
    for(int i = 0; i < pixelCount; i++){
        output[i].x = input[i*3];
        output[i].y = input[i*3 + 1];
        output[i].z = input[i*3 + 2];
    }
}

void uchar3ToChar(uchar3 *input, char *output, int pixelCount){
    for(int i = 0; i < pixelCount; i++){
        output[i*3] = input[i].x;
        output[i*3 + 1] = input[i].y;
        output[i*3 + 2] = input[i].z;
    }
}

void Labwork::labwork3_GPU() {
    // Calculate number of pixels
    int pixelCount = inputImage->width * inputImage->height;
    
    // Allocate CUDA memory
    uchar3 *d_inputImage;
    uchar3 *d_grayImage;
    cudaMalloc(&d_inputImage, pixelCount * 3);
    cudaMalloc(&d_grayImage, pixelCount * 3);

    // Copy CUDA Memory from CPU to GPU
    cudaMemcpy(d_inputImage, inputImage->buffer, pixelCount * 3, cudaMemcpyHostToDevice);

    // Processing
    int blockSize = 256;
    int numBlock = pixelCount / blockSize;
    Timer t;
    t.start();
    grayScale<<<numBlock, blockSize>>>(d_inputImage, d_grayImage);
    cudaDeviceSynchronize();

    printf("time elapsed : %fms\n", t.getElapsedTimeInMilliSec());
    // Copy CUDA Memory from GPU to CPU
    uchar3 *outputGrayImage;
    outputGrayImage = (uchar3 *) malloc(pixelCount * 3);
    cudaMemcpy(outputGrayImage, d_grayImage, pixelCount * 3, cudaMemcpyDeviceToHost);

    // Cleaning
    cudaFree(d_grayImage);
    cudaFree(d_inputImage);
    outputImage = (char*) outputGrayImage;
}

__global__ void grayScale2D(uchar3 *input, uchar3 *output, int widthImage, int heightImage){
    int width = blockDim.x * gridDim.x;
    
    int tidX = threadIdx.x + blockIdx.x * blockDim.x;
    int tidY = threadIdx.y + blockIdx.y * blockDim.y;

    if (tidY * width + tidX < (widthImage * heightImage)) {
        output[tidY * width + tidX].x = (input[tidY * width + tidX].x + 
                                        input[tidY * width + tidX].y +
                                        input[tidY * width + tidX].z) / 3;
        output[tidY * width + tidX].z = output[tidY * width + tidX].y = output[tidY * width + tidX].x;       
    }
}

void Labwork::labwork4_GPU() {
    // Calculate number of pixels
    int pixelCount = inputImage->width * inputImage->height;
    
    // Allocate CUDA memory
    uchar3 *d_inputImage;
    uchar3 *d_grayImage;
    cudaMalloc(&d_inputImage, pixelCount * 3);
    cudaMalloc(&d_grayImage, pixelCount * 3);

    // Copy CUDA Memory from CPU to GPU
    cudaMemcpy(d_inputImage, inputImage->buffer, pixelCount * 3, cudaMemcpyHostToDevice);

    // Processing
    dim3 blockSize = dim3(32,32);
    dim3 gridSize = dim3(inputImage->width/blockSize.x, inputImage->height/blockSize.y);
    
    if (inputImage->width % blockSize.x != 0){
        gridSize.x += 1;
    }

    if (inputImage->height % blockSize.y != 0){
        gridSize.y += 1;
    }

    Timer t;
    t.start();
    grayScale2D<<<gridSize, blockSize>>>(d_inputImage, d_grayImage, inputImage->width, inputImage->height);
    cudaDeviceSynchronize();

    printf("time elapsed : %fms\n", t.getElapsedTimeInMilliSec());
    // Copy CUDA Memory from GPU to CPU
    uchar3 *outputGrayImage;
    outputGrayImage = (uchar3 *) malloc(pixelCount * 3);
    cudaMemcpy(outputGrayImage, d_grayImage, pixelCount * 3, cudaMemcpyDeviceToHost);

    // Cleaning
    cudaFree(d_grayImage);
    cudaFree(d_inputImage);
    outputImage = (char*) outputGrayImage;
}

void Labwork::labwork5_CPU() {
    int filter[] = {
        0, 0, 1, 2, 1, 0, 0,
        0, 3, 13, 22, 13, 3, 0,
        1, 13, 59, 97, 59, 13, 1, 
        2, 22, 97, 159, 97, 22, 2, 
        1, 13, 59, 97, 59, 13, 1, 
        0, 3, 13, 22, 13, 3, 0, 
        0, 0, 1, 2, 1, 0, 0,
    };
    int width = inputImage->width;
    int height = inputImage->height;

    labwork1_CPU();
    char* inputImageLab5 = outputImage;
    outputImage = static_cast<char *>(malloc(width * height * 3));

    for(int i = 3; i < height - 3; i++){
        for(int j = 3; j < width - 3; j++){
            int sum = 0;
            for(int iFilter = 0; iFilter < 7; iFilter++){
                for(int jFilter = 0; jFilter < 7; jFilter++){
                    sum += filter[iFilter*7 + jFilter] * inputImageLab5[((i + (iFilter - 3)) * width + (j + (jFilter - 3))) * 3];
                }
            }
            sum /= 1003;
            outputImage[(i * width + j) * 3] = sum;
            outputImage[(i * width + j) * 3 + 1] = sum;
            outputImage[(i * width + j) * 3 + 2] = sum;
        }
    }
}

__global__ void blurrImage(uchar3* input, uchar3* output){
    
}

void Labwork::labwork5_GPU(bool shared) {

}

void Labwork::labwork6_GPU() {
}

void Labwork::labwork7_GPU() {
}

void Labwork::labwork8_GPU() {
}

void Labwork::labwork9_GPU() {

}

void Labwork::labwork10_GPU(){
}
