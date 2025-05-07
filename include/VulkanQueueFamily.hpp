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

namespace xrmg {
class VulkanQueueFamily {
public:
  VulkanQueueFamily(uint32_t p_queueFamilyIndex);

  void allocateCommandBuffers(vk::Device p_device, uint32_t p_commandPoolCount, uint32_t p_commandBufferCountPerPool);
  uint32_t getIndex() const { return m_queueFamilyIndex; }
  void reset(vk::Device p_device);
  vk::CommandBuffer nextCommandBuffer();

private:
  struct CommandPoolAndBuffers {
    vk::UniqueCommandPool pool;
    std::vector<vk::UniqueCommandBuffer> buffers;
  };

  uint32_t m_queueFamilyIndex;
  std::vector<CommandPoolAndBuffers> m_poolsAndBuffers;
  uint32_t m_currentCommandPoolIndex;
  uint32_t m_nextCommandBufferIndex;
};
} // namespace xrmg