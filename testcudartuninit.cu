// ======================================================================== //
// Copyright 2022 Louis Pisha                                               //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include <iostream>
#include <cstdio>
#include <cinttypes>
#include <thread>

#define checkCudaErrors(val) checkCudaErrors_( (val), #val, __LINE__ )
void checkCudaErrors_(cudaError_t result, char const *const func, int const line)
{
    if(result){
        std::cout << "CUDA call " << func << " failed line " << line << "\n";
        cudaDeviceReset();
        std::abort();
    }
}

const char * const memTypeNames[4] = {
    "cudaMemoryTypeUnregistered",
    "cudaMemoryTypeHost",
    "cudaMemoryTypeDevice",
    "cudaMemoryTypeManaged"
};
const char *getMemType(int i){
    if(i >= 0 && i < 4) return memTypeNames[i];
    return "error";
}

float *mem_d;

void threadfunc()
{
    int dev;
    checkCudaErrors(cudaGetDevice(&dev));
    printf("CUDA device is %d\n", dev);
    cudaPointerAttributes attrs;
    checkCudaErrors(cudaPointerGetAttributes(&attrs, mem_d));
    printf("type %s, device %d, devptr %016" PRIx64 ", hostptr %016" PRIx64 "\n",
        getMemType((int)attrs.type), attrs.device, attrs.devicePointer, attrs.hostPointer);
    printf("Setting CUDA device to 0\n");
    checkCudaErrors(cudaSetDevice(0));
    checkCudaErrors(cudaPointerGetAttributes(&attrs, mem_d));
    printf("type %s, device %d, devptr %016" PRIx64 ", hostptr %016" PRIx64 "\n",
        getMemType((int)attrs.type), attrs.device, attrs.devicePointer, attrs.hostPointer);
}


int main(int argc, char **argv)
{
    checkCudaErrors(cudaSetDevice(0));
    checkCudaErrors(cudaMalloc(&mem_d, 10000));
    
    std::thread th(threadfunc);
    th.join();
    std::cout << "Done\n";
}
