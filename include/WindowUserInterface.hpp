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

#include "UserInputSink.hpp"
#include "UserInterface.hpp"
#include "Window.hpp"

#include <unordered_map>

namespace xrmg {
class WindowUserInterface : public UserInterface, public UserInputSink {
public:
  WindowUserInterface(Window &p_window) : m_window(p_window) { m_window.pushUserInputSink(*this); }

  vk::Extent2D getResolutionPerEye() override;
  vk::UniqueInstance createVkInstance(const vk::InstanceCreateInfo &p_createInfo) override;
  std::optional<uint32_t> queryMainPhysicalDevice(vk::Instance p_vkInstance, uint32_t p_queueFamilyIndex,
                                                  uint32_t p_candidateCount, vk::PhysicalDevice *p_candidates) override;
  std::vector<const char *> getNeededDeviceExtensions() override { return {}; }
  void initialize(const Renderer &p_renderer, uint32_t p_presentQueueFamilyIndex,
                  uint32_t p_presentQueueIndex) override;
  void update(float p_millis) override;
  FrameInfo beginFrame() override;
  Mat4x4f getCurrentFrameView(StereoProjection::Eye p_eye) override;
  StereoProjection getCurrentFrameProjection(StereoProjection::Eye p_eye) override;
  FrameRenderTargets acquireSwapchainImages(vk::Device p_device) override;
  vk::Semaphore getSwapchainImageReadySemaphore() override;
  void releaseSwapchainImage() override;
  vk::Semaphore getFrameReadySemaphore() override { return m_frameReadySemaphore.get(); }
  void endFrame(vk::Queue p_presentGraphicsQueue) override;
  float getAspectRatioPerEye();

  bool onKeyDown(int32_t p_key) override;
  bool onKeyUp(int32_t p_key) override;
  bool onMouseMove(int32_t p_deltaX, int32_t p_deltaY) override;
  bool onWheelMove(int32_t p_delta) override;

private:
  Window &m_window;
  std::optional<uint32_t> m_currentSwapchainImageIndex;
  std::array<vk::UniqueSemaphore, MAX_QUEUED_FRAMES> m_swapchainImageReadySemaphores;
  size_t m_swapchainImageReadySemaphoreIndex = 0;
  vk::UniqueSemaphore m_frameReadySemaphore;
  bool m_fast = true;
  float m_ipd = 0.065f;
  float m_projectionPlaneDistance = 10.0f;
  std::unordered_map<StereoProjection::Eye, StereoProjection> m_projections;
  float m_runtimeMillis = 0.0f;
  Vec3f m_camMoveDir = {};
  Vec3f m_camPos = {0.0f, 3.0f, 0.0f};
  Angle m_camPitch = {};
  Angle m_camYaw = Angle::deg(180.0f);
  std::optional<std::chrono::high_resolution_clock::time_point> m_lastBeginFrame;

  void printSurfaceCapabilities(uint32_t p_presentQueueFamilyIndex, uint32_t p_physicalDeviceCount,
                                vk::PhysicalDevice *p_physicalDevices) const;
  void buildProjections();
};
} // namespace xrmg