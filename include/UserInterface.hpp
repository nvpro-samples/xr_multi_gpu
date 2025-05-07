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

#include "Matrix.hpp"
#include "StereoProjection.hpp"

namespace xrmg {
class Renderer;

class UserInterface {
public:
  struct FrameInfo {
    std::optional<uint64_t> predictedDisplayTimeNanos;
  };

  struct FrameRenderTargets {
    vk::Image colorImage;
    vk::ImageLayout desiredColorImageLayoutOnRelease;
    vk::Image depthImage;
    vk::ImageLayout desiredDepthImageLayoutOnRelease;
  };

  virtual ~UserInterface() {}

  virtual vk::Extent2D getResolutionPerEye() = 0;
  virtual vk::UniqueInstance createVkInstance(const vk::InstanceCreateInfo &p_createInfo) = 0;
  virtual std::optional<uint32_t> queryMainPhysicalDevice(vk::Instance p_vkInstance, uint32_t p_queueFamilyIndex,
                                                          uint32_t p_candidateCount,
                                                          vk::PhysicalDevice *p_candidates) = 0;
  virtual std::vector<const char *> getNeededDeviceExtensions() = 0;
  virtual void initialize(const Renderer &p_renderer, uint32_t p_presentQueueFamilyIndex,
                          uint32_t p_presentQueueIndex) = 0;
  virtual void update(float p_millis) = 0;
  virtual FrameInfo beginFrame() = 0;
  virtual Mat4x4f getCurrentFrameView(StereoProjection::Eye p_eye) = 0;
  virtual StereoProjection getCurrentFrameProjection(StereoProjection::Eye p_eye) = 0;
  virtual FrameRenderTargets acquireSwapchainImages(vk::Device p_device) = 0;
  virtual vk::Semaphore getSwapchainImageReadySemaphore() = 0;
  virtual void releaseSwapchainImage() = 0;
  virtual vk::Semaphore getFrameReadySemaphore() = 0;
  virtual void endFrame(vk::Queue p_presentGraphicsQueue) = 0;
};
} // namespace xrmg
