/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "Renderer.hpp"

#include "App.hpp"
#include "Options.hpp"
#include "RenderTarget.hpp"

namespace xrmg {
static const std::filesystem::path g_pipelineCachePath = "./" SAMPLE_NAME ".pipeline-cache.bin";

static const std::vector<const char *> g_vulkanInstanceExtensions = {"VK_KHR_get_physical_device_properties2",
                                                                     "VK_KHR_surface"};
static const std::vector<const char *> g_vulkanDeviceExtensions = {
    "VK_KHR_dynamic_rendering",     "VK_KHR_get_memory_requirements2", "VK_KHR_swapchain",
    "VK_KHR_calibrated_timestamps",
#ifdef _WIN32
    "VK_KHR_external_memory_win32", "VK_KHR_external_fence_win32",
#endif
};

Renderer::Renderer(std::unique_ptr<UserInterface> p_userInterface) : m_userInterface(std::move(p_userInterface)) {
  vk::ApplicationInfo appInfo(SAMPLE_NAME, 1, nullptr, 0, VK_API_VERSION_1_4);
  vk::InstanceCreateInfo instanceCreateInfo({}, &appInfo, {}, g_vulkanInstanceExtensions);
  m_vkInstance = m_userInterface->createVkInstance(instanceCreateInfo);

  this->fillPhysicalDevices();
  this->createQueueFamilies();
  this->updateMainPhysicalDevice();
  this->createLogicalDevice();
  this->printVulkanMemoryProps();
  this->initPipelineCache();
  this->createMainRenderTargets();

  m_userInterface->initialize(*this, this->getGraphicsQueueFamilyIndex(), 1);
  if (!m_userInterface->getSwapchainImageReadySemaphore()) {
    m_swapchainImageReadySemaphore = m_vkDevice->createSemaphoreUnique({});
  }
}

Renderer::~Renderer() { this->savePipelineCache(); }

void Renderer::fillPhysicalDevices() {
  std::vector<vk::PhysicalDeviceGroupProperties> physicalDeviceGroupProps =
      m_vkInstance->enumeratePhysicalDeviceGroups();
  XRMG_ASSERT(!physicalDeviceGroupProps.empty(), "No device groups available.");
  std::optional<uint32_t> selectedDeviceGroupIndex = g_app->getOptions().devGroupIndex;
  XRMG_INFO("Device groups:");
  for (uint32_t devGroupIndex = 0; devGroupIndex < physicalDeviceGroupProps.size(); ++devGroupIndex) {
    const vk::PhysicalDeviceGroupProperties &groupProps = physicalDeviceGroupProps[devGroupIndex];
    if (!selectedDeviceGroupIndex && (groupProps.physicalDeviceCount == 2 || groupProps.physicalDeviceCount == 4 ||
                                      g_app->getOptions().simulatedPhysicalDeviceCount.has_value())) {
      selectedDeviceGroupIndex = devGroupIndex;
    }
    XRMG_INFO("{}[{}] device count: {}",
              selectedDeviceGroupIndex && selectedDeviceGroupIndex.value() == devGroupIndex ? ">" : " ", devGroupIndex,
              groupProps.physicalDeviceCount);
    for (uint32_t devIndex = 0; devIndex < groupProps.physicalDeviceCount; ++devIndex) {
      vk::PhysicalDeviceProperties devProps = groupProps.physicalDevices[devIndex].getProperties();
      XRMG_INFO("  {}─╴{}", devIndex == groupProps.physicalDeviceCount - 1 ? "└" : "├", devProps.deviceName.data());
    }
  }
  XRMG_ASSERT(selectedDeviceGroupIndex,
              "No compatible device group found. Only groups of size 2 or 4 are supported when not in simulated mode.");
  XRMG_ASSERT(selectedDeviceGroupIndex.value() < physicalDeviceGroupProps.size(), "Invalid device group index: {}",
              selectedDeviceGroupIndex.value());
  XRMG_INFO("Selected device group: {}", selectedDeviceGroupIndex.value());
  const vk::PhysicalDeviceGroupProperties &devGroup = physicalDeviceGroupProps[selectedDeviceGroupIndex.value()];
  m_vkPhysicalDevices = {devGroup.physicalDevices.data(),
                         devGroup.physicalDevices.data() + devGroup.physicalDeviceCount};
}

void Renderer::createQueueFamilies() {
  std::vector<vk::QueueFamilyProperties> queueFamilyProps = m_vkPhysicalDevices.front().getQueueFamilyProperties();
  auto graphicsQueueIt =
      std::find_if(queueFamilyProps.begin(), queueFamilyProps.end(), [](const vk::QueueFamilyProperties &p_props) {
        return (p_props.queueFlags & vk::QueueFlagBits::eGraphics) && 2 <= p_props.queueCount;
      });
  XRMG_ASSERT(graphicsQueueIt != queueFamilyProps.end(), "No graphics capable queue available.");
  auto transferQueueIt =
      std::find_if(queueFamilyProps.begin(), queueFamilyProps.end(), [&](const vk::QueueFamilyProperties &p_props) {
        return &(*graphicsQueueIt) != &p_props && p_props.queueFlags & vk::QueueFlagBits::eTransfer;
      });
  XRMG_ASSERT(transferQueueIt != queueFamilyProps.end(), "No dedicated transfer capable queue available.");
  m_graphicsQueueFamily =
      std::make_unique<VulkanQueueFamily>(static_cast<uint32_t>(graphicsQueueIt - queueFamilyProps.begin()));
  m_transferQueueFamily =
      std::make_unique<VulkanQueueFamily>(static_cast<uint32_t>(transferQueueIt - queueFamilyProps.begin()));

  XRMG_INFO("Queue families of first physical device:");
  for (size_t i = 0; i < queueFamilyProps.size(); ++i) {
    std::string mark;
    if (i == m_graphicsQueueFamily->getIndex()) {
      mark = " [selected for graphics]";
    } else if (i == m_transferQueueFamily->getIndex()) {
      mark = " [selected for transfer]";
    }
    XRMG_INFO("{}╴{:2d} queues: {}\033[96m{}\033[0m", i == queueFamilyProps.size() - 1 ? "└" : "├",
              queueFamilyProps[i].queueCount, vk::to_string(queueFamilyProps[i].queueFlags), mark);
  }

  // We only take the first physical device into account to determine graphics and transfer queue families. If they
  // don't match with those of the other physical devices we can't continue.
  for (uint32_t devIdx = 1; devIdx < m_vkPhysicalDevices.size(); ++devIdx) {
    queueFamilyProps = m_vkPhysicalDevices[devIdx].getQueueFamilyProperties();
    XRMG_ASSERT(m_graphicsQueueFamily->getIndex() < queueFamilyProps.size(),
                "From the first physical device {} was selected as the graphics queue family but physical device {} "
                "doesn't have such queue family.",
                m_graphicsQueueFamily->getIndex(), devIdx);
    XRMG_ASSERT(queueFamilyProps[m_graphicsQueueFamily->getIndex()].queueFlags & vk::QueueFlagBits::eGraphics,
                "Physical device {}'s queue family {} does not have the graphics bit set.", devIdx,
                m_graphicsQueueFamily->getIndex());
    XRMG_ASSERT(m_transferQueueFamily->getIndex() < queueFamilyProps.size(),
                "From the first physical device {} was selected as the transfer queue family but physical device {} "
                "doesn't have such queue family.",
                m_transferQueueFamily->getIndex(), devIdx);
    XRMG_ASSERT(queueFamilyProps[m_transferQueueFamily->getIndex()].queueFlags & vk::QueueFlagBits::eTransfer,
                "Physical device {}'s queue family {} does not have the transfer bit set.", devIdx,
                m_transferQueueFamily->getIndex());
  }
}

void Renderer::updateMainPhysicalDevice() {
  // Not all physical devices can present to the swapchain. We need to find the one that can.
  std::optional<uint32_t> mainPhysicalDeviceIndex = m_userInterface->queryMainPhysicalDevice(
      m_vkInstance.get(), m_graphicsQueueFamily->getIndex(), static_cast<uint32_t>(m_vkPhysicalDevices.size()),
      &m_vkPhysicalDevices.front());
  if (!mainPhysicalDeviceIndex) {
    XRMG_WARN("UserInterface did not provide a main physical device.");
    return;
  } else if (mainPhysicalDeviceIndex.value() != 0) {
    XRMG_INFO("Swapping physical devices 0 and {} of selected device group.", mainPhysicalDeviceIndex.value());
    std::swap(m_vkPhysicalDevices[0], m_vkPhysicalDevices[mainPhysicalDeviceIndex.value()]);
  }
  if (g_app->getOptions().simulatedPhysicalDeviceCount) {
    m_vkPhysicalDevices.erase(m_vkPhysicalDevices.begin() + 1, m_vkPhysicalDevices.end());
  }
}

void Renderer::createLogicalDevice() {
  // The logical device may be created with a single physical device if the app runs in simulated mode.
  std::vector<float> graphicsQueuePriorities = {1.0f, 1.0f};
  std::vector<float> transferQueuePriorities = {1.0f};
  std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos = {
      {{}, m_graphicsQueueFamily->getIndex(), graphicsQueuePriorities},
      {{}, m_transferQueueFamily->getIndex(), transferQueuePriorities}};
  std::vector<const char *> deviceExtensions = m_userInterface->getNeededDeviceExtensions();
  deviceExtensions.insert(deviceExtensions.end(), g_vulkanDeviceExtensions.begin(), g_vulkanDeviceExtensions.end());
  vk::StructureChain deviceCreateInfoChain(
      vk::DeviceCreateInfo({}, queueCreateInfos, {}, deviceExtensions, nullptr),
      vk::DeviceGroupDeviceCreateInfo(m_vkPhysicalDevices), vk::PhysicalDeviceDynamicRenderingFeatures(true),
      vk::PhysicalDeviceTimelineSemaphoreFeatures(true), vk::PhysicalDeviceSynchronization2Features(true));
  m_vkDevice = m_vkPhysicalDevices.front().createDeviceUnique(deviceCreateInfoChain.get());

  m_graphicsQueueFamily->allocateCommandBuffers(m_vkDevice.get(), MAX_QUEUED_FRAMES, 10);
  m_renderQueue = m_vkDevice->getQueue(m_graphicsQueueFamily->getIndex(), 0);
  m_presentQueue = m_vkDevice->getQueue(m_graphicsQueueFamily->getIndex(), 1);

  m_transferQueueFamily->allocateCommandBuffers(m_vkDevice.get(), MAX_QUEUED_FRAMES, 10);
  m_transferQueue = m_vkDevice->getQueue(m_transferQueueFamily->getIndex(), 0);

  m_deviceMaskAll =
      g_app->getOptions().simulatedPhysicalDeviceCount.has_value() ? 0b1 : (1 << this->getPhysicalDeviceCount()) - 1;

  vk::StructureChain<vk::SemaphoreCreateInfo, vk::SemaphoreTypeCreateInfo> timelineSemaphoreCreateInfo(
      {}, {vk::SemaphoreType::eTimeline});
  m_frameIndexSem = m_vkDevice->createSemaphoreUnique(timelineSemaphoreCreateInfo.get());
  m_renderDoneSemaphores.resize(this->getPhysicalDeviceCount());
  for (vk::UniqueSemaphore &sem : m_renderDoneSemaphores) {
    sem = m_vkDevice->createSemaphoreUnique({});
  }
  m_transferDoneSemaphore = m_vkDevice->createSemaphoreUnique(timelineSemaphoreCreateInfo.get());
}

void Renderer::createMainRenderTargets() {
  m_resolutionPerPhysicalDevice = m_userInterface->getResolutionPerEye();
  if (this->getPhysicalDeviceCount() == 4) {
    m_resolutionPerPhysicalDevice.height /= 2;
  }
  for (uint32_t devIdx = 0; devIdx < this->getPhysicalDeviceCount(); ++devIdx) {
    m_renderTargets.emplace_back(*this, devIdx);
  }
}

void Renderer::initPipelineCache() {
  std::vector<char> cacheData;
  if (std::filesystem::is_regular_file(g_pipelineCachePath)) {
    std::ifstream cache(g_pipelineCachePath, std::ios::binary | std::ios::ate);
    if (cache) {
      cacheData.resize(cache.tellg());
      cache.seekg(std::ios::beg);
      cache.read(cacheData.data(), cacheData.size());
      if (!cache) {
        XRMG_ERROR("Failed to read from vulkan pipeline cache file: {}", g_pipelineCachePath.string());
      } else {
        XRMG_INFO("Vulkan pipeline cache loaded from file: {}", g_pipelineCachePath.string());
      }
    } else {
      XRMG_WARN("Failed to open vulkan pipeline cache file: {}", g_pipelineCachePath.string());
    }
  } else {
    XRMG_INFO("Vulkan pipeline cache file not found: {}", g_pipelineCachePath.string());
  }
  vk::PipelineCacheCreateInfo pipelineCacheCreateInfo({}, cacheData.size(), cacheData.data());
  m_pipelineCache = m_vkDevice->createPipelineCacheUnique(pipelineCacheCreateInfo);
}

void Renderer::savePipelineCache() noexcept {
  if (m_vkDevice && m_pipelineCache) {
    std::ofstream cache(g_pipelineCachePath, std::ios::binary);
    if (cache) {
      std::vector<uint8_t> cacheData = m_vkDevice->getPipelineCacheData(m_pipelineCache.get());
      if (!cache.write(reinterpret_cast<const char *>(cacheData.data()), cacheData.size())) {
        XRMG_WARN("Failed to write vulkan pipeline cache to file: {}", g_pipelineCachePath.string());
      }
    } else {
      XRMG_WARN("Failed to open vulkan pipeline cache file for writing: {}", g_pipelineCachePath.string());
    }
  }
}

void Renderer::nextFrame(Scene &p_scene) {
  UserInterface::FrameInfo frameInfo;
  {
    XRMG_SCOPED_INSTRUMENT("begin frame");
    frameInfo = m_userInterface->beginFrame();
  }
  if (!frameInfo.predictedDisplayTimeNanos) {
    XRMG_SCOPED_INSTRUMENT("end frame");
    m_userInterface->endFrame(m_presentQueue);
    return;
  }
  float deltaMillis = m_lastPredictedDisplayTimeNanos
                          ? 1e-6f * static_cast<float>(frameInfo.predictedDisplayTimeNanos.value() -
                                                       m_lastPredictedDisplayTimeNanos.value())
                          : 0.0f;
  m_lastPredictedDisplayTimeNanos = frameInfo.predictedDisplayTimeNanos;
  m_runtimeMillis += deltaMillis;
  {
    XRMG_SCOPED_INSTRUMENT("scene update");
    p_scene.update(deltaMillis);
  }
  {
    XRMG_SCOPED_INSTRUMENT("user interface update");
    m_userInterface->update(deltaMillis);
  }
  {
    // The frame rendering can start before we aquire the swapchain image. Depending on the OpenXR runtime, acquiring
    // the swapchain image may block all graphics queue operations until the last frame's scanout finishes. That's why
    // we do it at the latest possible time.
    XRMG_SCOPED_INSTRUMENT("render frame");
    if (MAX_QUEUED_FRAMES <= m_frameIndex) {
      XRMG_SCOPED_INSTRUMENT("wait for frame index value");
      uint64_t waitValue = m_frameIndex - MAX_QUEUED_FRAMES + 1;
      vk::Result waitResult = m_vkDevice.get().waitSemaphores({{}, m_frameIndexSem.get(), waitValue}, UINT64_MAX);
      XRMG_ASSERT(waitResult == vk::Result::eSuccess, "Wait failed.");
    }
    m_graphicsQueueFamily->reset(m_vkDevice.get());
    m_transferQueueFamily->reset(m_vkDevice.get());
    this->renderFrame(p_scene);
  }

  UserInterface::FrameRenderTargets frt;
  {
    // With the rendering commands submitted, we can now acquire the swapchain image.
    XRMG_SCOPED_INSTRUMENT("acquire swap chain images");
    frt = m_userInterface->acquireSwapchainImages(m_vkDevice.get());
    XRMG_INFO_ONCE("Depth buffer transfers: {}", XRMG_BOOL_TO_STRING(frt.depthImage));
  }
  if (!m_userInterface->getSwapchainImageReadySemaphore()) {
    vk::SemaphoreSubmitInfo swapchainImageReadySignal(m_swapchainImageReadySemaphore.get(), 0,
                                                      vk::PipelineStageFlagBits2::eAllCommands, 0);
    m_presentQueue.submit2(vk::SubmitInfo2({}, {}, {}, swapchainImageReadySignal));
  }
  {
    XRMG_SCOPED_INSTRUMENT("build final frame");
    this->buildFinalFrame(frt);
  }
  {
    XRMG_SCOPED_INSTRUMENT("release swap chain image");
    m_userInterface->releaseSwapchainImage();
  }
  {
    XRMG_SCOPED_INSTRUMENT("end frame");
    m_userInterface->endFrame(m_presentQueue);
  }
  ++m_frameIndex;
}

void Renderer::renderFrame(Scene &p_scene) {
  std::vector<vk::SubmitInfo2> graphicsSubmits;
  std::vector<vk::SemaphoreSubmitInfo> semaphoreWaits(2 * this->getPhysicalDeviceCount());
  uint32_t nextSemWaitIdx = 0;
  std::vector<vk::CommandBufferSubmitInfo> graphicsCmdBufferSubmits(this->getPhysicalDeviceCount());
  std::vector<vk::SemaphoreSubmitInfo> semaphoreSignals(this->getPhysicalDeviceCount());

  for (uint32_t devIdx = 0; devIdx < this->getPhysicalDeviceCount(); ++devIdx) {
    RenderTarget &rt = m_renderTargets[devIdx];
    vk::Image rtColorImage = rt.getColorResource(m_frameIndex).getImage();
    vk::ImageView rtColorImageView = rt.getColorResource(m_frameIndex).getImageView();
    vk::Image rtDepthImage = rt.getDepthResource(m_frameIndex).getImage();
    vk::ImageView rtDepthImageView = rt.getDepthResource(m_frameIndex).getImageView();

    vk::CommandBuffer cmdBuffer = m_graphicsQueueFamily->nextCommandBuffer();
    cmdBuffer.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    if (devIdx == 0 && m_frameIndex == 0) {
      g_app->getProfiler().resetQueryPool(cmdBuffer);
    }
    g_app->getProfiler().pushDurationBegin(std::format("render device {}", devIdx), m_frameIndex, devIdx, cmdBuffer);
    uint32_t srcQueueFamilyIndex =
        m_frameIndex < MAX_QUEUED_FRAMES ? m_graphicsQueueFamily->getIndex() : m_transferQueueFamily->getIndex();
    // Before rendering, we need to transfer the color and depth images from the transfer queue family to the graphics
    // queue family.
    std::vector<vk::ImageMemoryBarrier2> transferToGraphicsQueueFamilyBarriersEnd = {
        {
            vk::PipelineStageFlagBits2::eAllCommands,
            vk::AccessFlagBits2::eNone,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            vk::AccessFlagBits2::eColorAttachmentWrite,
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eColorAttachmentOptimal,
            srcQueueFamilyIndex,
            m_graphicsQueueFamily->getIndex(),
            rtColorImage,
            {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        },
        {
            vk::PipelineStageFlagBits2::eAllCommands,
            vk::AccessFlagBits2::eNone,
            vk::PipelineStageFlagBits2::eEarlyFragmentTests,
            vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eDepthAttachmentOptimal,
            srcQueueFamilyIndex,
            m_graphicsQueueFamily->getIndex(),
            rtDepthImage,
            {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1},
        },
    };
    cmdBuffer.pipelineBarrier2({{}, {}, {}, transferToGraphicsQueueFamilyBarriersEnd});
    auto eye = static_cast<StereoProjection::Eye>(devIdx % 2);
    if (g_app->getOptions().swapEyes) {
      eye = static_cast<StereoProjection::Eye>(1 - static_cast<int32_t>(eye));
    }
    StereoProjection proj = m_userInterface->getCurrentFrameProjection(eye);
    Rect2Df eyeViewport = proj.relativeViewport;
    Rect2Df deviceViewport = this->getDeviceViewport(devIdx);
    eyeViewport.x -= deviceViewport.x / deviceViewport.width;
    eyeViewport.y -= deviceViewport.y / deviceViewport.height;
    eyeViewport.width /= deviceViewport.width;
    eyeViewport.height /= deviceViewport.height;
    vk::Viewport vp(eyeViewport.x * static_cast<float>(this->getResolutionPerPhysicalDevice().width),
                    eyeViewport.y * static_cast<float>(this->getResolutionPerPhysicalDevice().height),
                    eyeViewport.width * static_cast<float>(this->getResolutionPerPhysicalDevice().width),
                    eyeViewport.height * static_cast<float>(this->getResolutionPerPhysicalDevice().height), 0.0f, 1.0f);
    p_scene.render(devIdx, cmdBuffer, rtColorImageView, rtDepthImageView, {{0, 0}, m_resolutionPerPhysicalDevice}, vp,
                   m_userInterface->getCurrentFrameView(eye), proj.projectionMatrix);
    // After rendering, we need to transfer the color and depth images from the graphics queue family to the transfer
    // queue family.
    std::vector<vk::ImageMemoryBarrier2> graphicsToTransferQueueFamilyBarriersBegin = {
        {
            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            vk::AccessFlagBits2::eColorAttachmentWrite,
            vk::PipelineStageFlagBits2::eTransfer,
            vk::AccessFlagBits2::eTransferRead,
            vk::ImageLayout::eColorAttachmentOptimal,
            vk::ImageLayout::eTransferSrcOptimal,
            m_graphicsQueueFamily->getIndex(),
            m_transferQueueFamily->getIndex(),
            rtColorImage,
            {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        },
        {
            vk::PipelineStageFlagBits2::eLateFragmentTests,
            vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
            vk::PipelineStageFlagBits2::eTransfer,
            vk::AccessFlagBits2::eTransferRead,
            vk::ImageLayout::eDepthAttachmentOptimal,
            vk::ImageLayout::eTransferSrcOptimal,
            m_graphicsQueueFamily->getIndex(),
            m_transferQueueFamily->getIndex(),
            rtDepthImage,
            {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1},
        },
    };
    cmdBuffer.pipelineBarrier2({{}, {}, {}, graphicsToTransferQueueFamilyBarriersBegin});
    g_app->getProfiler().pushDurationEnd(cmdBuffer);
    cmdBuffer.end();

    graphicsCmdBufferSubmits[devIdx] = vk::CommandBufferSubmitInfo(cmdBuffer, this->deviceIndexToDeviceMask(devIdx));
    semaphoreSignals[devIdx] = vk::SemaphoreSubmitInfo(m_renderDoneSemaphores[devIdx].get(), 0,
                                                       vk::PipelineStageFlagBits2::eAllCommands, devIdx);

    uint32_t semWaitCount = 0;
    vk::SemaphoreSubmitInfo *semWaits = &semaphoreWaits[nextSemWaitIdx];
    if (MAX_QUEUED_FRAMES <= m_frameIndex) {
      semaphoreWaits[nextSemWaitIdx++] =
          vk::SemaphoreSubmitInfo(m_frameIndexSem.get(), m_frameIndex - MAX_QUEUED_FRAMES + 1,
                                  vk::PipelineStageFlagBits2::eAllCommands, devIdx);
      ++semWaitCount;
    }
    graphicsSubmits.emplace_back(vk::SubmitInfo2({}, semWaitCount, semWaits, 1, &graphicsCmdBufferSubmits[devIdx], 1,
                                                 &semaphoreSignals[devIdx]));
  }
  m_renderQueue.submit2(graphicsSubmits);
}

void Renderer::buildFinalFrame(const UserInterface::FrameRenderTargets &p_renderTargets) {
  // The invidual render targets are transferred to the swapchain image. The swapchain image and depth image are then
  // prepared to be returned to the user interface by transitioning them to their desired layout.
  std::vector<vk::SemaphoreSubmitInfo> renderDoneWaits(this->getPhysicalDeviceCount() + 1);
  renderDoneWaits.back() = {m_swapchainImageReadySemaphore ? m_swapchainImageReadySemaphore.get()
                                                           : m_userInterface->getSwapchainImageReadySemaphore(),
                            0, vk::PipelineStageFlagBits2::eAllCommands, 0};
  std::vector<vk::ImageMemoryBarrier2> graphicsToTransferQueueFamilyBarriersEnd = {
      {vk::PipelineStageFlagBits2::eAllCommands,
       vk::AccessFlagBits2::eNone,
       vk::PipelineStageFlagBits2::eTransfer,
       vk::AccessFlagBits2::eTransferWrite,
       vk::ImageLayout::eUndefined,
       vk::ImageLayout::eTransferDstOptimal,
       VK_QUEUE_FAMILY_IGNORED,
       VK_QUEUE_FAMILY_IGNORED,
       p_renderTargets.colorImage,
       {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}}};
  std::vector<vk::ImageMemoryBarrier2> transferToGraphicsQueueFamilyBarriersBegin = {
      {vk::PipelineStageFlagBits2::eTransfer,
       vk::AccessFlagBits2::eTransferWrite,
       vk::PipelineStageFlagBits2::eAllCommands,
       vk::AccessFlagBits2::eMemoryRead,
       vk::ImageLayout::eTransferDstOptimal,
       p_renderTargets.desiredColorImageLayoutOnRelease,
       m_transferQueueFamily->getIndex(),
       m_graphicsQueueFamily->getIndex(),
       p_renderTargets.colorImage,
       {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}}};
  if (p_renderTargets.depthImage) {
    graphicsToTransferQueueFamilyBarriersEnd.emplace_back(vk::ImageMemoryBarrier2(
        vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eNone, vk::PipelineStageFlagBits2::eTransfer,
        vk::AccessFlagBits2::eTransferWrite, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, p_renderTargets.depthImage,
        {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1}));
    transferToGraphicsQueueFamilyBarriersBegin.emplace_back(
        vk::ImageMemoryBarrier2(vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
                                vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryRead,
                                vk::ImageLayout::eTransferDstOptimal, p_renderTargets.desiredDepthImageLayoutOnRelease,
                                m_transferQueueFamily->getIndex(), m_graphicsQueueFamily->getIndex(),
                                p_renderTargets.depthImage, {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1}));
  }

  std::vector<vk::ImageCopy2> colorRegions(this->getPhysicalDeviceCount());
  std::vector<vk::ImageCopy2> depthRegions(this->getPhysicalDeviceCount());
  std::vector<vk::CopyImageInfo2> copyImageInfos;
  for (uint32_t devIdx = 0; devIdx < this->getPhysicalDeviceCount(); ++devIdx) {
    vk::Image rtColorImage = m_renderTargets[devIdx].getColorResource(m_frameIndex).getImage();
    vk::Image rtDepthImage = m_renderTargets[devIdx].getDepthResource(m_frameIndex).getImage();

    renderDoneWaits[devIdx] = {m_renderDoneSemaphores[devIdx].get(), 0, vk::PipelineStageFlagBits2::eAllCommands, 0};
    graphicsToTransferQueueFamilyBarriersEnd.emplace_back(vk::ImageMemoryBarrier2(
        vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead,
        vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eTransferSrcOptimal,
        m_graphicsQueueFamily->getIndex(), m_transferQueueFamily->getIndex(), rtColorImage,
        {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}));
    graphicsToTransferQueueFamilyBarriersEnd.emplace_back(vk::ImageMemoryBarrier2(
        vk::PipelineStageFlagBits2::eLateFragmentTests, vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
        vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead,
        vk::ImageLayout::eDepthAttachmentOptimal, vk::ImageLayout::eTransferSrcOptimal,
        m_graphicsQueueFamily->getIndex(), m_transferQueueFamily->getIndex(), rtDepthImage,
        {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1}));

    vk::Offset3D dstOffset((devIdx % 2) * m_resolutionPerPhysicalDevice.width,
                           (devIdx / 2) * m_resolutionPerPhysicalDevice.height);
    colorRegions[devIdx] = vk::ImageCopy2(
        {vk::ImageAspectFlagBits::eColor, 0, 0, 1}, {0, 0, 0}, {vk::ImageAspectFlagBits::eColor, 0, 0, 1}, dstOffset,
        {m_resolutionPerPhysicalDevice.width, m_resolutionPerPhysicalDevice.height, 1});
    copyImageInfos.emplace_back(rtColorImage, vk::ImageLayout::eTransferSrcOptimal, p_renderTargets.colorImage,
                                vk::ImageLayout::eTransferDstOptimal, colorRegions[devIdx]);

    if (p_renderTargets.depthImage) {
      depthRegions[devIdx] = vk::ImageCopy2(
          {vk::ImageAspectFlagBits::eDepth, 0, 0, 1}, {0, 0, 0}, {vk::ImageAspectFlagBits::eDepth, 0, 0, 1}, dstOffset,
          {m_resolutionPerPhysicalDevice.width, m_resolutionPerPhysicalDevice.height, 1});
      copyImageInfos.emplace_back(rtDepthImage, vk::ImageLayout::eTransferSrcOptimal, p_renderTargets.depthImage,
                                  vk::ImageLayout::eTransferDstOptimal, depthRegions[devIdx]);
    }

    transferToGraphicsQueueFamilyBarriersBegin.emplace_back(vk::ImageMemoryBarrier2(
        vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead,
        vk::PipelineStageFlagBits2::eEarlyFragmentTests, vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
        vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eDepthAttachmentOptimal,
        m_transferQueueFamily->getIndex(), m_graphicsQueueFamily->getIndex(), rtDepthImage,
        {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1}));
    transferToGraphicsQueueFamilyBarriersBegin.emplace_back(vk::ImageMemoryBarrier2(
        vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eColorAttachmentOptimal,
        m_transferQueueFamily->getIndex(), m_graphicsQueueFamily->getIndex(), rtColorImage,
        {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}));
  }

  std::vector<VkCommandBufferSubmitInfo> transferCmdBufferSubmits(this->getPhysicalDeviceCount());
  vk::CommandBuffer transferCmdBuffer = m_transferQueueFamily->nextCommandBuffer();
  transferCmdBuffer.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
  g_app->getProfiler().pushDurationBegin("transfer", m_frameIndex, 0, transferCmdBuffer);
  transferCmdBuffer.pipelineBarrier2({{}, {}, {}, graphicsToTransferQueueFamilyBarriersEnd});
  uint32_t c = 0;
  for (const auto &copyImageInfo : copyImageInfos) {
    g_app->getProfiler().pushDurationBegin(std::format("tranfer {}", c++), m_frameIndex, 0, transferCmdBuffer);
    transferCmdBuffer.copyImage2(copyImageInfo);
    g_app->getProfiler().pushDurationEnd(transferCmdBuffer);
  }
  transferCmdBuffer.pipelineBarrier2({{}, {}, {}, transferToGraphicsQueueFamilyBarriersBegin});
  g_app->getProfiler().pushDurationEnd(transferCmdBuffer);
  transferCmdBuffer.end();
  vk::CommandBufferSubmitInfo transferCmdBufferSubmit(transferCmdBuffer, this->getDeviceMaskFirst());
  vk::SemaphoreSubmitInfo transferDoneSignalAndWait(m_transferDoneSemaphore.get(), m_frameIndex + 1,
                                                    vk::PipelineStageFlagBits2::eAllCommands, 0);
  m_transferQueue.submit2(vk::SubmitInfo2({}, renderDoneWaits, transferCmdBufferSubmit, transferDoneSignalAndWait));

  vk::CommandBuffer finalCmdBuffer = m_graphicsQueueFamily->nextCommandBuffer();
  finalCmdBuffer.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
  std::vector<vk::ImageMemoryBarrier2> finalBarriers = {{vk::PipelineStageFlagBits2::eTransfer,
                                                         vk::AccessFlagBits2::eTransferWrite,
                                                         vk::PipelineStageFlagBits2::eAllCommands,
                                                         vk::AccessFlagBits2::eMemoryRead,
                                                         vk::ImageLayout::eTransferDstOptimal,
                                                         p_renderTargets.desiredColorImageLayoutOnRelease,
                                                         m_transferQueueFamily->getIndex(),
                                                         m_graphicsQueueFamily->getIndex(),
                                                         p_renderTargets.colorImage,
                                                         {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}}};
  if (p_renderTargets.depthImage) {
    finalBarriers.emplace_back(
        vk::ImageMemoryBarrier2(vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
                                vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryRead,
                                vk::ImageLayout::eTransferDstOptimal, p_renderTargets.desiredDepthImageLayoutOnRelease,
                                m_transferQueueFamily->getIndex(), m_graphicsQueueFamily->getIndex(),
                                p_renderTargets.depthImage, {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1}));
  }
  finalCmdBuffer.pipelineBarrier2({{}, {}, {}, finalBarriers});
  g_app->getProfiler().pushInstant("finalize", m_frameIndex, 0, finalCmdBuffer);
  finalCmdBuffer.end();
  vk::CommandBufferSubmitInfo finalCmdBufferSubmit(finalCmdBuffer, this->getDeviceMaskFirst());
  std::vector<vk::SemaphoreSubmitInfo> finalSignals = {
      {m_frameIndexSem.get(), m_frameIndex + 1, vk::PipelineStageFlagBits2::eAllCommands, 0}};
  if (vk::Semaphore frameReadySem = m_userInterface->getFrameReadySemaphore(); frameReadySem) {
    finalSignals.emplace_back(frameReadySem, 0, vk::PipelineStageFlagBits2::eAllCommands, 0);
  }
  m_presentQueue.submit2(vk::SubmitInfo2({}, transferDoneSignalAndWait, finalCmdBufferSubmit, finalSignals));
}

Rect2Df Renderer::getDeviceViewport(uint32_t p_physicalDeviceIndex) const {
  XRMG_ASSERT(p_physicalDeviceIndex < this->getPhysicalDeviceCount(),
              "Phyiscal device index ({}) must be less than the number of physical device ({})", p_physicalDeviceIndex,
              this->getPhysicalDeviceCount());
  if (this->getPhysicalDeviceCount() == 2) {
    // Each device renders the full viewport of the left and right eye, respectively.
    return {.x = 0.0f, .y = 0.0f, .width = 1.0f, .height = 1.0f};
  } else if (this->getPhysicalDeviceCount() == 4) {
    // Device 0 and 2 render the left eye, device 1 and 3 render the right eye.
    return p_physicalDeviceIndex < 2 ? Rect2Df{.x = 0.0f, .y = 0.0f, .width = 1.0f, .height = 0.5f}
                                     : Rect2Df{.x = 0.0f, .y = 0.5f, .width = 1.0f, .height = 0.5f};
  }
  XRMG_FATAL("{} physical devices not supported.", this->getPhysicalDeviceCount());
  return {};
}

uint32_t Renderer::getPhysicalDeviceCount() const {
  return g_app->getOptions().simulatedPhysicalDeviceCount.value_or(static_cast<uint32_t>(m_vkPhysicalDevices.size()));
}

vk::PhysicalDevice Renderer::getPhysicalDevice(uint32_t p_index) const {
  return g_app->getOptions().simulatedPhysicalDeviceCount.has_value() ? m_vkPhysicalDevices.front()
                                                                      : m_vkPhysicalDevices[p_index];
}

uint32_t Renderer::deviceIndexToDeviceMask(uint32_t p_index) const {
  return g_app->getOptions().simulatedPhysicalDeviceCount.has_value() ? 0b1 : 1 << p_index;
}

std::optional<uint32_t> Renderer::queryCompatibleMemoryTypeIndex(uint32_t p_physicalDeviceIndex,
                                                                 vk::MemoryPropertyFlags p_propertyFlags,
                                                                 std::optional<uint32_t> p_filterMemTypeBits) const {
  vk::PhysicalDeviceMemoryProperties memProps =
      m_vkPhysicalDevices[g_app->getOptions().simulatedPhysicalDeviceCount.has_value() ? 0 : p_physicalDeviceIndex]
          .getMemoryProperties();
  std::optional<uint32_t> candidate;
  for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
    if (!p_filterMemTypeBits || ((1 << i) & p_filterMemTypeBits.value())) {
      if (memProps.memoryTypes[i].propertyFlags == p_propertyFlags) {
        return i;
      } else if ((memProps.memoryTypes[i].propertyFlags & p_propertyFlags) == p_propertyFlags) {
        candidate = i;
      }
    }
  }
  return candidate;
}

void Renderer::printVulkanMemoryProps() const {
  XRMG_INFO("Physical devices memory heaps");
  for (uint32_t devIdx = 0; devIdx < m_vkPhysicalDevices.size(); ++devIdx) {
    vk::PhysicalDeviceMemoryProperties memProps = m_vkPhysicalDevices[devIdx].getMemoryProperties();
    XRMG_INFO(" [{}] {}; heap count: {}, memory type count: {}", devIdx,
              m_vkPhysicalDevices[devIdx].getProperties().deviceName.data(), memProps.memoryHeapCount,
              memProps.memoryTypeCount);
    for (uint32_t i = 0; i < memProps.memoryHeapCount; ++i) {
      bool lastHeap = i == memProps.memoryHeapCount - 1;
      bool deviceLocal = memProps.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal ? true : false;
      bool multiInstance = memProps.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eMultiInstance ? true : false;
      XRMG_INFO("  {}─╴[{}] size: {},\tdevice local: {}, multi instance: {}", lastHeap ? "└" : "├", i,
                formatByteSize(memProps.memoryHeaps[i].size).c_str(), XRMG_BOOL_TO_STRING(deviceLocal),
                XRMG_BOOL_TO_STRING(multiInstance));
      for (uint32_t j = 0; j < memProps.memoryTypeCount; ++j) {
        if (memProps.memoryTypes[j].heapIndex == i) {
          bool lastMemType = std::find_if(memProps.memoryTypes + j + 1, memProps.memoryTypes + memProps.memoryTypeCount,
                                          [i](const vk::MemoryType &p_type) { return p_type.heapIndex == i; }) ==
                             memProps.memoryTypes + memProps.memoryTypeCount;
          XRMG_INFO("  {}   {}─╴memory type index: {}, property flags: {}", lastHeap ? " " : "│",
                    lastMemType ? "└" : "├", j, vk::to_string(memProps.memoryTypes[j].propertyFlags).c_str());
        }
      }
    }
  }
}
} // namespace xrmg