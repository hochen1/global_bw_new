#include <common.h>


#define MSTRINGIFY(...) #__VA_ARGS__
#define FETCH_PER_WI 16
#define BUILD_OPTIONS " -cl-mad-enable "

#define PRINT(...) print(__VA_ARGS__)

template<typename T, typename... Args>
void print(T arg, Args... args) {
    std::cout << arg << " ";
}


static const std::string stringifiedKernels =
#include "global_bandwidth_kernels.cl"
    ;

float run_kernel(cl::CommandQueue &queue, cl::Kernel &kernel, cl::NDRange &globalSize, cl::NDRange &localSize, uint iters)
{
  bool useEventTimer = true;
  float timed = 0;

  // Dummy calls
  queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize);
  queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize);
  queue.finish();

  if (useEventTimer)
  {
    for (uint i = 0; i < iters; i++)
    {
      cl::Event timeEvent;

      queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize, NULL, &timeEvent);
      queue.finish();
      timed += timeInUS(timeEvent);
    }
  }
  else // std timer
  {
    Timer timer;

    timer.start();
    for (uint i = 0; i < iters; i++)
    {
      queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize);
      queue.flush();
    }
    queue.finish();
    timed = timer.stopAndTime();
  }

  return (timed / static_cast<float>(iters));
}

int runGlobalBandwidthTest(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo)
{
    bool forceTest = true;
    float timed_lo, timed_go, timed, gbps;
    cl::NDRange globalSize, localSize;
    float *arr = NULL;

    cl::Context ctx = queue.getInfo<CL_QUEUE_CONTEXT>();
    uint iters = devInfo.gloalBWIters;

    uint64_t maxItems = devInfo.maxAllocSize / sizeof(float) / 2;
    uint64_t numItems = roundToMultipleOf(maxItems, (devInfo.maxWGSize * FETCH_PER_WI * 16), devInfo.globalBWMaxSize);
    // std::cout << "num: " << numItems << std::endl; 
    try
    {
        arr = new float[numItems];
        populate(arr, numItems);

        PRINT(NEWLINE TAB TAB "Global memory bandwidth (GBPS)" NEWLINE);
        PRINT(TAB TAB "global_memory_bandwidth");
        PRINT("unit: gbps" NEWLINE);

        cl::Buffer inputBuf = cl::Buffer(ctx, CL_MEM_READ_ONLY, (numItems * sizeof(float)));
        cl::Buffer outputBuf = cl::Buffer(ctx, CL_MEM_WRITE_ONLY, (numItems * sizeof(float)));
        queue.enqueueWriteBuffer(inputBuf, CL_TRUE, 0, (numItems * sizeof(float)), arr);

        cl::Kernel kernel_v1_lo(prog, "global_bandwidth_v1_local_offset");
        kernel_v1_lo.setArg(0, inputBuf), kernel_v1_lo.setArg(1, outputBuf);

        cl::Kernel kernel_v2_lo(prog, "global_bandwidth_v2_local_offset");
        kernel_v2_lo.setArg(0, inputBuf), kernel_v2_lo.setArg(1, outputBuf);

        cl::Kernel kernel_v4_lo(prog, "global_bandwidth_v4_local_offset");
        kernel_v4_lo.setArg(0, inputBuf), kernel_v4_lo.setArg(1, outputBuf);

        cl::Kernel kernel_v8_lo(prog, "global_bandwidth_v8_local_offset");
        kernel_v8_lo.setArg(0, inputBuf), kernel_v8_lo.setArg(1, outputBuf);

        cl::Kernel kernel_v16_lo(prog, "global_bandwidth_v16_local_offset");
        kernel_v16_lo.setArg(0, inputBuf), kernel_v16_lo.setArg(1, outputBuf);

        cl::Kernel kernel_v1_go(prog, "global_bandwidth_v1_global_offset");
        kernel_v1_go.setArg(0, inputBuf), kernel_v1_go.setArg(1, outputBuf);

        cl::Kernel kernel_v2_go(prog, "global_bandwidth_v2_global_offset");
        kernel_v2_go.setArg(0, inputBuf), kernel_v2_go.setArg(1, outputBuf);

        cl::Kernel kernel_v4_go(prog, "global_bandwidth_v4_global_offset");
        kernel_v4_go.setArg(0, inputBuf), kernel_v4_go.setArg(1, outputBuf);

        cl::Kernel kernel_v8_go(prog, "global_bandwidth_v8_global_offset");
        kernel_v8_go.setArg(0, inputBuf), kernel_v8_go.setArg(1, outputBuf);

        cl::Kernel kernel_v16_go(prog, "global_bandwidth_v16_global_offset");
        kernel_v16_go.setArg(0, inputBuf), kernel_v16_go.setArg(1, outputBuf);

        localSize = devInfo.maxWGSize;

        ///////////////////////////////////////////////////////////////////////////
        // Vector width 1
        if (forceTest)
        {
            // log->print(TAB TAB TAB "float   : ");
            PRINT(TAB TAB TAB "float   : ");

            globalSize = numItems / FETCH_PER_WI;

            // Run 2 kind of bandwidth kernel
            // lo -- local_size offset - subsequent fetches at local_size offset
            // go -- global_size offset
            timed_lo = run_kernel(queue, kernel_v1_lo, globalSize, localSize, iters);
            timed_go = run_kernel(queue, kernel_v1_go, globalSize, localSize, iters);
            timed = (timed_lo < timed_go) ? timed_lo : timed_go;

            gbps = ((float)numItems * sizeof(float)) / timed / 1e3f;

            PRINT(gbps);
            PRINT(NEWLINE);
        }
        ///////////////////////////////////////////////////////////////////////////

        // Vector width 2
        if (forceTest)
        {
            // log->print(TAB TAB TAB "float2  : ");
            PRINT(TAB TAB TAB "float2  : ");

            globalSize = (numItems / 2 / FETCH_PER_WI);

            timed_lo = run_kernel(queue, kernel_v2_lo, globalSize, localSize, iters);
            timed_go = run_kernel(queue, kernel_v2_go, globalSize, localSize, iters);
            timed = (timed_lo < timed_go) ? timed_lo : timed_go;

            gbps = ((float)numItems * sizeof(float)) / timed / 1e3f;

            PRINT(gbps);
            PRINT(NEWLINE);
        }
        ///////////////////////////////////////////////////////////////////////////

        // Vector width 4
        if (forceTest)
        {
            // log->print(TAB TAB TAB "float4  : ");
            PRINT(TAB TAB TAB "float4  : ");

            globalSize = (numItems / 4 / FETCH_PER_WI);

            timed_lo = run_kernel(queue, kernel_v4_lo, globalSize, localSize, iters);
            timed_go = run_kernel(queue, kernel_v4_go, globalSize, localSize, iters);
            timed = (timed_lo < timed_go) ? timed_lo : timed_go;

            gbps = ((float)numItems * sizeof(float)) / timed / 1e3f;

            PRINT(gbps);
            PRINT(NEWLINE);
        }
        ///////////////////////////////////////////////////////////////////////////

        // Vector width 8
        if (forceTest)
        {
            PRINT(TAB TAB TAB "float8  : ");

            globalSize = (numItems / 8 / FETCH_PER_WI);

            timed_lo = run_kernel(queue, kernel_v8_lo, globalSize, localSize, iters);
            timed_go = run_kernel(queue, kernel_v8_go, globalSize, localSize, iters);
            timed = (timed_lo < timed_go) ? timed_lo : timed_go;

            gbps = ((float)numItems * sizeof(float)) / timed / 1e3f;
            
            PRINT(gbps);
            PRINT(NEWLINE);
        }
        ///////////////////////////////////////////////////////////////////////////

        // Vector width 16
        if (forceTest)
        {
            PRINT(TAB TAB TAB "float16 : ");

            globalSize = (numItems / 16 / FETCH_PER_WI);

            timed_lo = run_kernel(queue, kernel_v16_lo, globalSize, localSize, iters);
            timed_go = run_kernel(queue, kernel_v16_go, globalSize, localSize, iters);
            timed = (timed_lo < timed_go) ? timed_lo : timed_go;

            gbps = ((float)numItems * sizeof(float)) / timed / 1e3f;
            
            PRINT(gbps);
            PRINT(NEWLINE);
        }
        ///////////////////////////////////////////////////////////////////////////

        if (arr)
        {
            delete[] arr;
        }
    }
    catch (cl::Error &error)
    {
        UNUSED(error);
        PRINT("Error");

        if (arr)
        {
        delete[] arr;
        }
        return -1;
    }

    return 0;
}



int main(int argc, char **argv)
{
    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);


    for (size_t p = 0; p < platforms.size(); p++)
    {
        cl_context_properties cps[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[p])(), 0};

        cl::Context ctx(CL_DEVICE_TYPE_ALL, cps);
        vector<cl::Device> devices = ctx.getInfo<CL_CONTEXT_DEVICES>();
        cl::Program::Sources source(1, stringifiedKernels);
        cl::Program prog = cl::Program(ctx, source);

        for (size_t d = 0; d < devices.size(); d++)
        {

            device_info_t devInfo = getDeviceInfo(devices[d]);
            
            try
            {
                vector<cl::Device> dev = {devices[d]};
                prog.build(dev, BUILD_OPTIONS);
            }
            catch (cl::Error &error)
            {
                UNUSED(error);
                PRINT("Error");
                continue;
            }

            PRINT("Device: " + devInfo.deviceName + NEWLINE);
            PRINT("Driver version  : ");
            PRINT(devInfo.driverVersion);
            PRINT(" (" OS_NAME ")" NEWLINE);
            PRINT("Compute units   : ");
            PRINT(devInfo.numCUs);
            PRINT(NEWLINE);
            PRINT("Clock frequency : ");
            PRINT(devInfo.maxClockFreq);
            PRINT(" MHz" NEWLINE);
            PRINT("device name:");
            PRINT(devInfo.deviceName);
            PRINT(NEWLINE);
            PRINT("driver_version: ");
            PRINT(devInfo.driverVersion);
            PRINT(NEWLINE);
            PRINT("compute_units: ");
            PRINT(devInfo.numCUs);
            PRINT(NEWLINE);
            PRINT("clock_frequency: ");
            PRINT(devInfo.maxClockFreq);
            PRINT(NEWLINE);
            PRINT("clock_frequency_unit: MHz");
            // PRINT(TAB TAB "Build Log: ", prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[d]), NEWLINE NEWLINE);

            cl::CommandQueue queue = cl::CommandQueue(ctx, devices[d], CL_QUEUE_PROFILING_ENABLE);

            runGlobalBandwidthTest(queue, prog, devInfo);

            PRINT(NEWLINE);
        }
    }
    return 0;
}
