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
#include "VulkanQueueFamily.hpp"

namespace xrmg {
VulkanQueueFamily::VulkanQueueFamily(uint32_t p_queueFamilyIndex)
    : m_queueFamilyIndex(p_queueFamilyIndex), m_nextCommandBufferIndex(0) {}

void VulkanQueueFamily::allocateCommandBuffers(vk::Device p_device, uint32_t p_commandPoolCount,
                                               uint32_t p_commandBufferCountPerPool) {
  m_poolsAndBuffers.resize(p_commandPoolCount);
  for (CommandPoolAndBuffers &poolAndBuffers : m_poolsAndBuffers) {
    poolAndBuffers.pool = p_device.createCommandPoolUnique({{}, m_queueFamilyIndex});
    poolAndBuffers.buffers = p_device.allocateCommandBuffersUnique(
        {poolAndBuffers.pool.get(), vk::CommandBufferLevel::ePrimary, p_commandBufferCountPerPool});
  }
  m_currentCommandPoolIndex = (p_commandPoolCount - 1);
  m_nextCommandBufferIndex = 0;
}

void VulkanQueueFamily::reset(vk::Device p_device) {
  m_currentCommandPoolIndex = (m_currentCommandPoolIndex + 1) % m_poolsAndBuffers.size();
  p_device.resetCommandPool(m_poolsAndBuffers[m_currentCommandPoolIndex].pool.get());
  m_nextCommandBufferIndex = 0;
}

vk::CommandBuffer VulkanQueueFamily::nextCommandBuffer() {
  std::vector<vk::UniqueCommandBuffer> &buffers = m_poolsAndBuffers[m_currentCommandPoolIndex].buffers;
  XRMG_ASSERT(m_nextCommandBufferIndex < buffers.size(), "No more command buffers available.");
  return buffers[m_nextCommandBufferIndex++].get();
}
} // namespace xrmg
