# TestCudaRTUninit

Reproducer for a CUDA driver bug where CUDA runtime state may be uninitialized
after a new thread starts.

The bug can be looked at one of two different ways:
1. `cudaPointerGetAttributes` depends on CUDA runtime state which may be
uninitialized when it is called, and it does not perform this initialization,
it returns incorrect results that memory is not accessible from the current
device, even though that memory actually is accessible.
2. `cudaGetDevice` incorrectly returns that device 0 is the current device,
but actually some uninitialized "device -1" is the current device.

Tested on Debian, Driver Version: 495.46, CUDA Version: 11.5 \
CUDA device 0: NVIDIA GeForce RTX 3080 \
CUDA device 1: NVIDIA GeForce RTX 1080

### To build

`nvcc testcudartuninit.c -o testcudartuninit`

### To run

```
>TestCudaRTUninit$ ./testcudartuninit
CUDA device is 0
type cudaMemoryTypeDevice, device 0, devptr 0000000000000000, hostptr 0000000000000000
Setting CUDA device to 0
type cudaMemoryTypeDevice, device 0, devptr 00007fadc7a00000, hostptr 0000000000000000
Done
>TestCudaRTUninit$
```
