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
#pragma once
#include "xrmg.hpp"

#include "UserInterface.hpp"
#include "VulkanQueueFamily.hpp"

namespace xrmg {
class RenderTarget;
class Scene;

class Renderer {
public:
  Renderer(std::unique_ptr<class UserInterface> p_userInterface);
  ~Renderer();

  vk::Instance vkInstance() const { return *m_vkInstance; }
  vk::Device vkDevice() const { return *m_vkDevice; }
  vk::PipelineCache getPipelineCache() const { return m_pipelineCache.get(); }
  uint32_t getPhysicalDeviceCount() const;
  vk::PhysicalDevice getPhysicalDevice(uint32_t p_index) const;
  uint32_t deviceIndexToDeviceMask(uint32_t p_index) const;
  uint32_t getDeviceMaskAll() const { return m_deviceMaskAll; }
  uint32_t getDeviceMaskFirst() const { return m_deviceMaskFirst; }
  const vk::Extent2D &getResolutionPerPhysicalDevice() const { return m_resolutionPerPhysicalDevice; }
  uint32_t getGraphicsQueueFamilyIndex() const { return m_graphicsQueueFamily->getIndex(); }
  std::optional<uint32_t> queryCompatibleMemoryTypeIndex(uint32_t p_physicalDeviceIndex,
                                                         vk::MemoryPropertyFlags p_propertyFlags,
                                                         std::optional<uint32_t> p_filterMemTypeBits = {}) const;
  float getRuntimeMillis() const { return m_runtimeMillis; }
  uint64_t getCurrentFrameIndex() const { return m_frameIndex; }
  void nextFrame(Scene &p_scene);
  void waitIdle() const { m_vkDevice->waitIdle(); }

private:
  vk::Extent2D m_resolutionPerPhysicalDevice;

  vk::UniqueInstance m_vkInstance;
  std::vector<vk::PhysicalDevice> m_vkPhysicalDevices;
  uint32_t m_deviceMaskAll;
  uint32_t m_deviceMaskFirst = 0b0001;
  vk::UniqueDevice m_vkDevice;
  std::unique_ptr<VulkanQueueFamily> m_graphicsQueueFamily;
  std::unique_ptr<VulkanQueueFamily> m_transferQueueFamily;
  vk::Queue m_renderQueue;
  vk::Queue m_transferQueue;
  vk::Queue m_presentQueue;
  vk::UniquePipelineCache m_pipelineCache;

  uint64_t m_frameIndex = 0;
  vk::UniqueSemaphore m_frameIndexSem;
  std::vector<vk::UniqueSemaphore> m_renderDoneSemaphores;
  vk::UniqueSemaphore m_swapchainImageReadySemaphore;
  vk::UniqueSemaphore m_transferDoneSemaphore;

  std::unique_ptr<UserInterface> m_userInterface;
  float m_runtimeMillis = 0.0f;
  std::optional<uint64_t> m_lastPredictedDisplayTimeNanos;
  std::vector<RenderTarget> m_renderTargets;

  void fillPhysicalDevices();
  void createQueueFamilies();
  void updateMainPhysicalDevice();
  void createLogicalDevice();
  void initPipelineCache();
  void savePipelineCache() noexcept;
  void createMainRenderTargets();

  void renderFrame(Scene &p_scene);
  void buildFinalFrame(const UserInterface::FrameRenderTargets &p_renderTargets);

  void printVulkanMemoryProps() const;
  Rect2Df getDeviceViewport(std::uint32_t p_physicalDeviceIndex) const;
};
} // namespace xrmg