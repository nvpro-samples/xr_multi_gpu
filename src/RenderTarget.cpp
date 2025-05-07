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
#include "RenderTarget.hpp"

#include "Renderer.hpp"

namespace xrmg {
RenderTarget::RenderTarget(const Renderer &p_renderer, uint32_t p_physicalDeviceIndex) {
  vk::ImageCreateInfo colorImageCreateInfo(
      {}, vk::ImageType::e2D, g_renderFormat, vk::Extent3D(p_renderer.getResolutionPerPhysicalDevice(), 1), 1, 1,
      vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal,
      vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc, vk::SharingMode::eExclusive, {},
      vk::ImageLayout::eUndefined);
  vk::ImageViewCreateInfo colorImageViewCreateInfo({}, {}, vk::ImageViewType::e2D, g_renderFormat, {},
                                                   {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
  vk::ImageCreateInfo depthImageCreateInfo(
      {}, vk::ImageType::e2D, g_depthFormat, vk::Extent3D(p_renderer.getResolutionPerPhysicalDevice(), 1), 1, 1,
      vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal,
      vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eTransferSrc,
      vk::SharingMode::eExclusive, {}, vk::ImageLayout::eUndefined);
  vk::ImageViewCreateInfo depthImageViewCreateInfo({}, {}, vk::ImageViewType::e2D, g_depthFormat, {},
                                                   {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1});
  for (uint32_t i = 0; i < MAX_QUEUED_FRAMES; ++i) {
    m_colorResources.emplace_back(p_renderer, p_physicalDeviceIndex, colorImageCreateInfo, colorImageViewCreateInfo);
    m_depthResources.emplace_back(p_renderer, p_physicalDeviceIndex, depthImageCreateInfo, depthImageViewCreateInfo);
  }
}

VulkanImageResource &RenderTarget::getColorResource(uint64_t p_frameIndex) {
  return m_colorResources[p_frameIndex % MAX_QUEUED_FRAMES];
}

const VulkanImageResource &RenderTarget::getColorResource(uint64_t p_frameIndex) const {
  return m_colorResources[p_frameIndex % MAX_QUEUED_FRAMES];
}

VulkanImageResource &RenderTarget::getDepthResource(uint64_t p_frameIndex) {
  return m_depthResources[p_frameIndex % MAX_QUEUED_FRAMES];
}

const VulkanImageResource &RenderTarget::getDepthResource(uint64_t p_frameIndex) const {
  return m_depthResources[p_frameIndex % MAX_QUEUED_FRAMES];
}
} // namespace xrmg