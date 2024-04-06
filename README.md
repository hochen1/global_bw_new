# global_bw_new

A simplified version of clpeak to measure global memory bandwidth.

## Building

```console
mkdir build
cd build
cmake ..
cmake --build .
```

## Sample

```console
.\Debug\clpeak.exe
Device: Intel(R) UHD Graphics
 Driver version  :  31.0.101.4255  (Win64)
 Compute units   :  48
 Clock frequency :  1400  MHz
 device name: Intel(R) UHD Graphics
 driver_version:  31.0.101.4255
 compute_units:  48
 clock_frequency:  1400
 clock_frequency_unit: MHz
    Global memory bandwidth (GBPS)
     global_memory_bandwidth unit: gbps
       float   :  20.2407 
       float2  :  21.1909 
       float4  :  21.8038 
       float8  :  22.2734 
       float16 :  22.3987 
```

