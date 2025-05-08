# xr_multi_gpu

![xr_multi_gpu](/doc/xr_multi_gpu.webp)

This sample focuses on multi-GPU side-by-side stereo rendering, as it is common for most XR HMDs. The frames can be presented either to an OpenXR runtime or – for easier testing and debugging – to a Win32 window.

## Introduction
XR workloads usually consist of multiple independent views which can be distributed easily across multiple physical devices. For frame presentation, these individual views must be transferred to the main physical device before they can be presented to the XR runtime, e.g. OpenXR.

With a modern graphics API like Vulkan, the developer has to carefully synchronize rendering with device to device image transfers and communication, while it is crucial – especially for XR – to not introduce unnecessary stalls and latency. Ideally, the rendering and transfer workloads should run in parallel. Vulkan offers a sophisticated execution model and a wide range of powerful and fine-grained resource synchronization techniques to achieve this in multiple ways. However, this wide range of possible solutions makes it difficult to choose and stick to one.

## The sample
The scene of the sample consists of an adjustable amount of tori. Their triangle count and fragment shader complexity can be adjusted to simulate different frame times.

Two modes are supported: a two GPU mode, each one rendering a single eye, and a four GPU mode, each one rendering one half of one eye. Additionally, both modes can be simulated on a single GPU which removes the necessity to set up a multi-GPU system before running the sample.

A single frame is rendered in 4 steps.
1. Each GPU uses a globally shared Vulkan graphics queue to independently render its own view onto a device local image and notifies a semaphore when it is done.
2. The first GPU waits for all semaphores to be signaled, including its own.
3. The first GPU uses a dedicated Vulkan transfer queue to issue device to device image transfers of the rendered images of all devices to the final swapchain image which resides inside the first GPU’s memory. A semaphore is signalled when all transfers are finished.
4. Another dedicated graphics queue is used to present the final image either to OpenXR or to a debugging window. Using a dedicated queue ensures that the primary graphics queue is free to start rendering the next frame early and does not have to wait until the current frame is presented.

### Usage
```
  xr_multi_gpu --help | -h
  xr_multi_gpu [--device-group <index>] [--simulate <count>] [--windowed [<width> <height>] | --monitor <index>] [--present-mode <string>] [--frame-time-log-interval <count>] [--trace-range <begin, end> [--trace-file <path>]] [--base-torus-tesselation <count>] [--base-torus-count <count>] [--torus-layer-count <count>]

Options:
  --help -h                            Show this text.
  --device-group <index>               Select the device group to use explicitly by its index. Only device groups of size 2 and 4 are allowed when not in simulated mode. If absent, the first compatible device group will be used.
  --simulate <count>                   Simulate multi-GPU rendering with <count> physical devices on a single one. All commands and resources will be executed and allocated on the first physical device of the selected device group; <count> must be 2 or 4.
  --windowed [<width> <height>]        Open a window of size <width> x <height> instead of using OpenXR; default: 1280 x 720
  --monitor <index>                    Open a fullscreen window on monitor <monitor index> instead of using OpenXR.
  --present-mode <string>              Set present mode for windowed and fullscreen rendering. Must be one of {fifo, fifoRelaxed, immediate, mailbox}; default: mailbox.
  --frame-time-log-interval <count>    Log the avg. frame time every <count> milliseconds to stdout.
  --trace-range <begin, end>           Enable CPU and GPU tracing of frames <begin> to <end>.
  --trace-file <path>                  Output file of tracing; default: ./trace.json
  --base-torus-tesselation <count>     The initial parametric surface subdivision of each torus will be 2 x <count> x <count>; default: 16
  --base-torus-count <count>           The number of tori per compass direction will be 2 x <count> x <count>; default: 5
  --torus-layer-count <count>          The number of layers per torus to sculpt its spikes; default: 8
```

### Controls
Torus count and details can be adjusted at runtime with the following keys.

|Property              |Decrease|Set to default|Increase|
|----------------------|--------|--------------|--------|
|Base torus count      |NUMPAD1 |NUMPAD4       |NUMPAD7 |
|Base torus tesselation|NUMPAD2 |NUMPAD5       |NUMPAD8 |
|Torus layer count     |NUMPAD3 |NUMPAD6       |NUMPAD9 |

When using a window instead of OpenXR, the camera can be moved with WASD keys and rotated by pressing the left mouse button on the window and dragging the mouse. ESC quits the applcation in both modes, windowed and OpenXR.

## Single GPU (baseline)
With a single GPU both, the frame time and latency are just the sum of the time it takes to render the image and the time it takes to present the image. Note, that with variable refresh rate the latter is usually neglectable. However, as most HMDs work with a fixed refresh rate, this call usually syncs with the scanout. XR applications are therefore advised to adjust their rendering loop and/or scene complexity to minimize the time between the present call and the display's scanout.

![Baseline](/doc/baseline.svg)

## Multi-GPU, weak scaling
In this sample, we refer to the naive apporach where a frame is rendered after the previous frame was presented as weak scaling. Compared to the baseline, in this optimal case the render time is divided by roughly the number of GPUs involved. However, an additional overhead is introduced by the time it takes to transfer all rendered portions of the final image from their rendering devices to the one that executes the scanout. This is the way how the OpenGL multiview extension works.

![Weak scaling](/doc/weak.svg)

The transfer time depends on the display's resolution. In case of a Meta Quest 3 and dual-GPU stereo rendering, this resolution is 2064 x 2208 per eye. Given that the app is rendering to a common 24 bpp RGB image, the amount of data to transfer would end up as ~17.4 MiB per frame. Many XR devices can also make use of the depth buffer which is commonly another 24 bpp, leading to ~35 MiB per frame. The following table compares the actual per frame transfer times depending on the PCIe generation.

|HMD         |single frame (RGB+D)|PCIe 4.0 (~25 GiB/s)|PCIe 5.0 (~50 GiB/s)|
|------------|--------------------| -------------------|--------------------|
|Meta Quest 3|~35 MiB             |1.4 ms              |0.7 ms              |
|Varjo XR-4  |~110 MiB            |4.4 ms              |2.2 ms              |

Note, that with two GPUs and stereo rendering the optimal case of cutting the rendering time in half is easily achievable as both eyes can see almost the same portion of the scene. Therefore, each GPU handles an identical rendering workload for a single eye. For more GPUs or more heterogeneous multi-view rendering the workload must be balanced more carefully.

## Multi-GPU Strong scaling
This sample demonstrates how to conceal the extra transfer time by initiating the rendering process of the next frame early, specifically right after all devices have completed rendering their portion of the final image. Typically, the transfer time is shorter than the rendering time, especially with current PCIe 5.0 speeds. With this optimization the frame time can be reduced by almost the number of devices, achieving an almost perfect multi-GPU scaling factor. The latency, however, will be the same as in the case of weak scaling.

![Strong scaling](/doc/strong.svg)

## Implementation details
The `Renderer.cpp` class implements a sophisticated synchronization mechanism between the graphics and transfer queues to achieve efficient multi-GPU rendering.

### Graphics Queue (Rendering Phase)
- Command Submission:
  - Each GPU independently renders its assigned view (e.g., left or right eye) using the **graphics queue**.
  - A **semaphore** is signaled (`m_renderDoneSemaphores[devIdx]`) when rendering is complete for each GPU.

- Synchronization:
  - Before rendering, the **transfer queue** releases ownership of the images (color and depth) to the **graphics queue** using `pipelineBarrier2` with `transferToGraphicsQueueFamilyBarriersEnd`.

- Pipeline Barrier:
  - After rendering, the ownership of the images is transferred back to the **transfer queue** using `pipelineBarrier2` with `graphicsToTransferQueueFamilyBarriersBegin`.

### Transfer Queue (Image Transfer Phase)
- Command Submission:
  - The **transfer queue** copies the rendered images from each GPU to the final composite image on the primary GPU using `copyImage2`.

- Synchronization:
  - The **graphics queue** signals the `m_renderDoneSemaphores` for each GPU, which the **transfer queue** waits on before starting the image transfer.

- Pipeline Barrier:
  - After the transfer is complete, the ownership of the composite image is returned to the **graphics queue** using `pipelineBarrier2` with `transferToGraphicsQueueFamilyBarriersBegin`.

### Graphics Queue (Final Presentation Phase)
- Command Submission:
  - The **graphics queue** prepares the final composite image for presentation to the XR runtime or debugging window.

- Synchronization:
  - The **transfer queue** signals the `m_transferDoneSemaphore` after completing the image transfer, which the **graphics queue** waits on before starting the final presentation.

- Pipeline Barrier:
  - The final composite image is transitioned to the desired layout for presentation using `pipelineBarrier2`.

### Semaphore Synchronization Logic

| Queue          | Semaphore                 | Purpose                                                                  |
|----------------|---------------------------|--------------------------------------------------------------------------|
| Graphics Queue | `m_renderDoneSemaphores`  | Signals when rendering is complete for each GPU.                         |
| Transfer Queue | `m_renderDoneSemaphores`  | Waits for rendering to complete before starting image transfer.          |
| Transfer Queue | `m_transferDoneSemaphore` | Signals when image transfer is complete.                                 |
| Graphics Queue | `m_transferDoneSemaphore` | Waits for image transfer to complete before starting final presentation. |
| Graphics Queue | `m_frameIndexSem`         | Tracks frame progress for synchronization across multiple frames.        |

## Known issues and limitations
At the time of release, Vulkan device groups with multiple NVIDIA devices are supported only on Windows systems. To enable this functionality, NVIDIA SLI must be activated within the NVIDIA Control Panel. We will update this sample once the feature becomes available on Linux.

Due to a known issue with the current NVIDIA drivers, Vulkan timestamp writes may produce inconsistent values when invoked from different physical devices. This inconsistency impacts the trace files generated using the `--trace-file` and `--trace-range` options. While we have implemented a temporary workaround to mitigate this issue, it is not entirely foolproof. Therefore, please exercise caution when interpreting the trace files.