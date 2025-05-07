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
class Renderer;

class VulkanImageResource {
public:
  VulkanImageResource(const Renderer &p_renderer, uint32_t p_physicalDeviceIndex,
                      const vk::ImageCreateInfo &p_imageCreateInfo,
                      std::optional<vk::ImageViewCreateInfo> p_imageViewCreateInfo = {});

  const vk::Image &getImage() const { return m_image.get(); }
  const vk::ImageView &getImageView() const { return m_imageView.get(); }

private:
  vk::UniqueDeviceMemory m_memory;
  vk::UniqueImage m_image;
  vk::UniqueImageView m_imageView;
};
} // namespace  xrmg
