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

namespace xrmg {
class Renderer;

class TriangleMesh {
public:
  struct Vertex {
    Vec3f pos;
    Vec3f normal;
    Vec2f tex;
  };

  typedef std::pair<std::vector<vk::VertexInputBindingDescription>, std::vector<vk::VertexInputAttributeDescription>>
      VertexDescription;

  static TriangleMesh createUnitCube(const Renderer &p_renderer, uint32_t p_maxInstances, size_t p_sizePerInstance);
  static TriangleMesh createPlaneXZ(const Renderer &p_renderer, uint32_t p_maxInstances, size_t p_sizePerInstance);
  static TriangleMesh createTorusXY(const Renderer &p_renderer, uint32_t p_maxInstances, size_t p_sizePerInstance,
                                    uint32_t p_subdivisionCount, float p_minorRadius, float p_majorRadius);

  TriangleMesh(const Renderer &p_renderer, uint32_t p_maxInstances, size_t p_sizePerInstance, uint32_t p_vertexCount,
               const Vertex *p_vertices, uint32_t p_indexCount = 0, const uint32_t *p_indices = nullptr);

  bool hasIndices() const { return m_indexCount != 0; }
  bool isUploaded() const { return m_uploaded; }
  void upload(vk::Device p_vkDevice, uint32_t p_queueFamilyIndex, uint32_t p_deviceMask);
  uint32_t getMaxInstances() const { return m_maxInstances; }
  vk::Buffer getInstanceBuffer() const { return m_instanceBuffer.get(); }
  void bind(vk::CommandBuffer p_cmdBuffer) const;
  void draw(vk::CommandBuffer p_cmdBuffer, uint32_t p_instanceCount = 1, uint32_t p_firstInstance = 0) const;

private:
  bool m_uploaded = false;
  uint32_t m_vertexCount;
  uint32_t m_indexCount;
  uint32_t m_maxInstances;
  vk::UniqueDeviceMemory m_uploadMem;
  vk::UniqueBuffer m_uploadBuffer;
  vk::UniqueBuffer m_vertexBuffer;
  vk::UniqueBuffer m_instanceBuffer;
  vk::UniqueBuffer m_indexBuffer;
  vk::UniqueDeviceMemory m_geoMem;
};
} // namespace xrmg