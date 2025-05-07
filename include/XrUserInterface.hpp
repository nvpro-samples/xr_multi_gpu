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

#include <openxr/openxr.h>

namespace xrmg {
class XrUserInterface : public UserInterface {
public:
  static const uint32_t VIEW_COUNT = 2;

  XrUserInterface(bool p_enableCoreValidation);
  ~XrUserInterface();

  vk::Extent2D getResolutionPerEye() override { return m_resolutionPerEye; }
  vk::UniqueInstance createVkInstance(const vk::InstanceCreateInfo &p_createInfo) override;
  std::optional<uint32_t> queryMainPhysicalDevice(vk::Instance p_vkInstance, uint32_t p_queueFamilyIndex,
                                                  uint32_t p_candidateCount, vk::PhysicalDevice *p_candidates) override;
  std::vector<const char *> getNeededDeviceExtensions() override;
  void initialize(const Renderer &p_renderer, uint32_t p_presentQueueFamilyIndex,
                  uint32_t p_presentQueueIndex) override;
  void update(float p_millis) override {}
  FrameInfo beginFrame() override;
  Mat4x4f getCurrentFrameView(StereoProjection::Eye p_eye) override;
  StereoProjection getCurrentFrameProjection(StereoProjection::Eye p_eye) override;
  FrameRenderTargets acquireSwapchainImages(vk::Device p_device) override;
  vk::Semaphore getSwapchainImageReadySemaphore() override { return {}; }
  void releaseSwapchainImage() override;
  vk::Semaphore getFrameReadySemaphore() override { return {}; }
  void endFrame(vk::Queue p_presentGraphicsQueue) override;

private:
  enum class SwapchainImageState {
    UNTOUCHED,
    ACQUIRED,
    RELEASED,
  };

  struct Swapchain {
    XrSwapchain swapchain;
    std::vector<vk::Image> images;
  };

  XrInstance m_instance = nullptr;
  XrSystemId m_systemId;
  vk::Extent2D m_resolutionPerEye;
  vk::PhysicalDevice m_mainPhysicalDevice;
  XrSession m_session = nullptr;
  Swapchain m_colorSwapchain;
  Swapchain m_depthSwapchain;
  XrSpace m_space;
  XrSessionState m_sessionState = XR_SESSION_STATE_UNKNOWN;
  XrTime m_currentFramePredictedDisplayTime;
  std::array<XrView, VIEW_COUNT> m_locatedViews;
  SwapchainImageState m_swapchainImageState;

  void handleEvents();
  void handle(XrEventDataSessionStateChanged &p_evt);
  Swapchain createSwapchain(const XrSwapchainCreateInfo &p_createInfo) const;
};
} // namespace xrmg