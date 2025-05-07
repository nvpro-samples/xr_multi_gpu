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
#include "VulkanImageResource.hpp"

#include "Renderer.hpp"

namespace xrmg {
VulkanImageResource::VulkanImageResource(const Renderer &p_renderer, uint32_t p_physicalDeviceIndex,
                                         const vk::ImageCreateInfo &p_imageCreateInfo,
                                         std::optional<vk::ImageViewCreateInfo> p_imageViewCreateInfo) {
  m_image = p_renderer.vkDevice().createImageUnique(p_imageCreateInfo);
  vk::MemoryRequirements memReqs = p_renderer.vkDevice().getImageMemoryRequirements(m_image.get());
  std::optional<uint32_t> memTypeIdx = p_renderer.queryCompatibleMemoryTypeIndex(
      p_physicalDeviceIndex, vk::MemoryPropertyFlagBits::eDeviceLocal, memReqs.memoryTypeBits);
  XRMG_ASSERT(memTypeIdx.has_value(), "No compatible memory type found.");
  vk::MemoryAllocateFlagsInfo alllocateFlagsInfo(vk::MemoryAllocateFlagBits::eDeviceMask,
                                                 p_renderer.deviceIndexToDeviceMask(p_physicalDeviceIndex));

  vk::MemoryAllocateInfo allocateInfo(memReqs.size, memTypeIdx.value(), &alllocateFlagsInfo);
  m_memory = p_renderer.vkDevice().allocateMemoryUnique(allocateInfo);
  vk::BindImageMemoryInfo bindInfo = {m_image.get(), m_memory.get()};
  p_renderer.vkDevice().bindImageMemory2(bindInfo);
  vk::ImageAspectFlags imageAspectFlags;
  if (p_imageViewCreateInfo) {
    vk::ImageViewCreateInfo imageViewCreateInfo = p_imageViewCreateInfo.value();
    imageViewCreateInfo.setImage(m_image.get());
    m_imageView = p_renderer.vkDevice().createImageViewUnique(imageViewCreateInfo);
  }
}
} // namespace xrmg