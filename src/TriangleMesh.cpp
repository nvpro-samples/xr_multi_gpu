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
#include "TriangleMesh.hpp"

#include "App.hpp"

#define ALIGN(_value, _alignment) ((((_value) + (_alignment) - 1) / (_alignment)) * (_alignment))
#define PRIMITIVE_RESTART 0xffffffff

namespace xrmg {
TriangleMesh TriangleMesh::createUnitCube(const Renderer &p_renderer, uint32_t p_maxInstances,
                                          size_t p_sizePerInstance) {
  std::vector<Vertex> vertices = {
      {{-0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 0.0f}},
      {{+0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 0.0f}},
      {{-0.5f, +0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f}},
      {{+0.5f, +0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 1.0f}},

      {{+0.5f, +0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}},
      {{-0.5f, +0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}},
      {{+0.5f, -0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
      {{-0.5f, -0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},

      {{+0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
      {{+0.5f, -0.5f, +0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
      {{+0.5f, +0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},
      {{+0.5f, +0.5f, +0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},

      {{-0.5f, +0.5f, +0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
      {{-0.5f, +0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
      {{-0.5f, -0.5f, +0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},
      {{-0.5f, -0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},

      {{-0.5f, +0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
      {{+0.5f, +0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
      {{-0.5f, +0.5f, +0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}},
      {{+0.5f, +0.5f, +0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}},

      {{+0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
      {{-0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}},
      {{+0.5f, -0.5f, +0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}},
      {{-0.5f, -0.5f, +0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}},
  };
  std::vector<uint32_t> indices = {
      0,  1,  2,  3,  PRIMITIVE_RESTART, 4,  5,  6,  7,  PRIMITIVE_RESTART, 8,  9,  10, 11, PRIMITIVE_RESTART,
      12, 13, 14, 15, PRIMITIVE_RESTART, 16, 17, 18, 19, PRIMITIVE_RESTART, 20, 21, 22, 23,
  };
  return {p_renderer,        p_maxInstances,
          p_sizePerInstance, static_cast<uint32_t>(vertices.size()),
          vertices.data(),   static_cast<uint32_t>(indices.size()),
          indices.data()};
}

TriangleMesh TriangleMesh::createTorusXY(const Renderer &p_renderer, uint32_t p_maxInstances, size_t p_sizePerInstance,
                                         uint32_t p_subdivisionCount, float p_minorRadius, float p_majorRadius) {
  std::vector<Vertex> vertices;
  vertices.reserve((p_subdivisionCount + 1) * (p_subdivisionCount + 1));
  std::vector<uint32_t> indices;
  for (uint32_t i = 0; i <= p_subdivisionCount; ++i) {
    for (uint32_t j = 0; j <= p_subdivisionCount; ++j) {
      Vec2f tex = {.x = static_cast<float>(i) / static_cast<float>(p_subdivisionCount),
                   .y = static_cast<float>(j) / static_cast<float>(p_subdivisionCount)};
      float phi = 2.0f * static_cast<float>(M_PI) * tex.x;
      float theta = 2.0f * static_cast<float>(M_PI) * tex.y;
      Vec3f pos = {
          .x = (p_majorRadius + p_minorRadius * std::sinf(theta)) * std::cosf(phi),
          .y = (p_majorRadius + p_minorRadius * std::sinf(theta)) * std::sinf(phi),
          .z = p_minorRadius * std::cosf(theta),
      };
      Vec3f normal = {
          .x = std::cosf(phi) * std::sinf(theta),
          .y = std::sinf(phi) * std::sinf(theta),
          .z = std::cosf(theta),
      };
      tex.y /= 16.0f;
      vertices.emplace_back(pos, normal, tex);
      if (i != p_subdivisionCount) {
        indices.emplace_back(i * (p_subdivisionCount + 1) + j);
        indices.emplace_back((i + 1) * (p_subdivisionCount + 1) + j);
      }
    }
    if (i != p_subdivisionCount) {
      indices.emplace_back(PRIMITIVE_RESTART);
    }
  }
  return TriangleMesh(p_renderer, p_maxInstances, p_sizePerInstance, static_cast<uint32_t>(vertices.size()),
                      vertices.data(), static_cast<uint32_t>(indices.size()), indices.data());
}

TriangleMesh TriangleMesh::createPlaneXZ(const Renderer &p_renderer, uint32_t p_maxInstances,
                                         size_t p_sizePerInstance) {
  std::vector<Vertex> vertices = {
      {{-1.0f, 0.0f, -1.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
      {{+1.0f, 0.0f, -1.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
      {{-1.0f, 0.0f, +1.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}},
      {{+1.0f, 0.0f, +1.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}},
  };
  std::vector<uint32_t> indices = {0, 1, 2, 3};
  return {p_renderer,        p_maxInstances,
          p_sizePerInstance, static_cast<uint32_t>(vertices.size()),
          vertices.data(),   static_cast<uint32_t>(indices.size()),
          indices.data()};
}

TriangleMesh::TriangleMesh(const Renderer &p_renderer, uint32_t p_maxInstances, size_t p_sizePerInstance,
                           uint32_t p_vertexCount, const Vertex *p_vertices, uint32_t p_indexCount,
                           const uint32_t *p_indices)
    : m_vertexCount(p_vertexCount), m_indexCount(p_indexCount), m_maxInstances(p_maxInstances) {
  std::optional<uint32_t> uploadMemTypeIdx =
      p_renderer.queryCompatibleMemoryTypeIndex(0, vk::MemoryPropertyFlagBits::eHostVisible);
  vk::DeviceSize uploadMemSize = m_vertexCount * sizeof(Vertex) + m_indexCount * sizeof(uint32_t);
  m_uploadMem = p_renderer.vkDevice().allocateMemoryUnique({uploadMemSize, uploadMemTypeIdx.value()});
  char *mapped = reinterpret_cast<char *>(p_renderer.vkDevice().mapMemory(m_uploadMem.get(), 0, uploadMemSize));
  memcpy(mapped, p_vertices, m_vertexCount * sizeof(Vertex));
  if (this->hasIndices()) {
    memcpy(mapped + m_vertexCount * sizeof(Vertex), p_indices, m_indexCount * sizeof(uint32_t));
  }
  p_renderer.vkDevice().unmapMemory(m_uploadMem.get());
  m_uploadBuffer = p_renderer.vkDevice().createBufferUnique(
      {{}, uploadMemSize, vk::BufferUsageFlagBits::eTransferSrc, vk::SharingMode::eExclusive});
  p_renderer.vkDevice().bindBufferMemory(m_uploadBuffer.get(), m_uploadMem.get(), 0);

  vk::BufferCreateInfo vertexBufferCreateInfo(
      {}, m_vertexCount * sizeof(Vertex),
      vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst, vk::SharingMode::eExclusive);
  m_vertexBuffer = p_renderer.vkDevice().createBufferUnique(vertexBufferCreateInfo);
  vk::MemoryRequirements vertexBufferMemReqs = p_renderer.vkDevice().getBufferMemoryRequirements(m_vertexBuffer.get());
  std::optional<uint32_t> vertexBufferMemTypeIdx = p_renderer.queryCompatibleMemoryTypeIndex(
      0, vk::MemoryPropertyFlagBits::eDeviceLocal, vertexBufferMemReqs.memoryTypeBits);
  XRMG_ASSERT(vertexBufferMemTypeIdx, "No memory type for vertex buffer available.");

  vk::DeviceSize geoBufferSize = vertexBufferMemReqs.size;
  vk::DeviceSize indexBufferOffset = 0;
  vk::DeviceSize instanceBufferOffset = 0;

  if (this->hasIndices()) {
    vk::BufferCreateInfo indexBufferCreateInfo(
        {}, m_indexCount * sizeof(uint32_t),
        vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst, vk::SharingMode::eExclusive);
    m_indexBuffer = p_renderer.vkDevice().createBufferUnique(indexBufferCreateInfo);
    vk::MemoryRequirements indexBufferMemReqs = p_renderer.vkDevice().getBufferMemoryRequirements(m_indexBuffer.get());
    std::optional<uint32_t> indexBufferMemTypeIdx = p_renderer.queryCompatibleMemoryTypeIndex(
        0, vk::MemoryPropertyFlagBits::eDeviceLocal, indexBufferMemReqs.memoryTypeBits);
    XRMG_ASSERT(indexBufferMemTypeIdx, "No memory type for index buffer available.");
    XRMG_ASSERT(vertexBufferMemTypeIdx.value() == indexBufferMemTypeIdx.value(),
                "Memory types of index and vertex buffer do not match.");

    indexBufferOffset = ALIGN(geoBufferSize, indexBufferMemReqs.alignment);
    geoBufferSize = indexBufferOffset + indexBufferMemReqs.size;
  }

  vk::BufferCreateInfo instanceBufferCreateInfo(
      {}, p_maxInstances * p_sizePerInstance,
      vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst, vk::SharingMode::eExclusive, {});
  m_instanceBuffer = p_renderer.vkDevice().createBufferUnique(instanceBufferCreateInfo);
  vk::MemoryRequirements instanceBufferMemReqs =
      p_renderer.vkDevice().getBufferMemoryRequirements(m_instanceBuffer.get());
  std::optional<uint32_t> instanceBufferMemTypeIndex = p_renderer.queryCompatibleMemoryTypeIndex(
      0, vk::MemoryPropertyFlagBits::eDeviceLocal, instanceBufferMemReqs.memoryTypeBits);
  XRMG_ASSERT(instanceBufferMemTypeIndex, "No memory type for instance buffer available.");
  XRMG_ASSERT(vertexBufferMemTypeIdx.value() == instanceBufferMemTypeIndex.value(),
              "Memory types of instance and vertex buffer do not match.");

  instanceBufferOffset = ALIGN(geoBufferSize, instanceBufferMemReqs.alignment);
  geoBufferSize = instanceBufferOffset + instanceBufferMemReqs.size;

  m_geoMem = p_renderer.vkDevice().allocateMemoryUnique({geoBufferSize, vertexBufferMemTypeIdx.value()});
  p_renderer.vkDevice().bindBufferMemory(m_vertexBuffer.get(), m_geoMem.get(), 0);
  if (this->hasIndices()) {
    p_renderer.vkDevice().bindBufferMemory(m_indexBuffer.get(), m_geoMem.get(), indexBufferOffset);
  }
  p_renderer.vkDevice().bindBufferMemory(m_instanceBuffer.get(), m_geoMem.get(), instanceBufferOffset);
}

void TriangleMesh::bind(vk::CommandBuffer p_cmdBuffer) const {
  XRMG_WARN_UNLESS(m_uploaded, "Binding triangle mesh before it was uploaded.");
  p_cmdBuffer.bindVertexBuffers(0, {m_vertexBuffer.get(), m_instanceBuffer.get()}, {0, 0});
  if (this->hasIndices()) {
    p_cmdBuffer.bindIndexBuffer(m_indexBuffer.get(), 0, vk::IndexType::eUint32);
  }
}

void TriangleMesh::draw(vk::CommandBuffer p_cmdBuffer, uint32_t p_instanceCount, uint32_t p_firstInstance) const {
  XRMG_WARN_UNLESS(m_uploaded, "Drawing triangle mesh before it was uploaded.");
  if (this->hasIndices()) {
    p_cmdBuffer.drawIndexed(m_indexCount, p_instanceCount, 0, 0, p_firstInstance);
  } else {
    p_cmdBuffer.draw(m_vertexCount, p_instanceCount, 0, p_firstInstance);
  }
}

void TriangleMesh::upload(vk::Device p_vkDevice, uint32_t p_queueFamilyIndex, uint32_t p_deviceMask) {
  XRMG_WARN_IF(m_uploaded, "Triangle mesh already uploaded.");
  vk::UniqueCommandPool cmdPool = p_vkDevice.createCommandPoolUnique({{}, p_queueFamilyIndex});
  vk::UniqueCommandBuffer cmdBuffer =
      std::move(p_vkDevice.allocateCommandBuffersUnique({cmdPool.get(), vk::CommandBufferLevel::ePrimary, 1}).front());

  vk::DeviceSize vbSize = m_vertexCount * sizeof(Vertex);
  vk::DeviceSize ibSize = m_indexCount * sizeof(uint32_t);
  std::vector<vk::BufferMemoryBarrier2> preUploadBarriers = {
      {vk::PipelineStageFlagBits2::eNone, vk::AccessFlagBits2::eNone, vk::PipelineStageFlagBits2::eTransfer,
       vk::AccessFlagBits2::eTransferWrite, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, m_vertexBuffer.get(), 0,
       vbSize},
  };
  std::vector<vk::BufferMemoryBarrier2> postUploadBarriers = {
      {vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
       vk::PipelineStageFlagBits2::eVertexAttributeInput, vk::AccessFlagBits2::eVertexAttributeRead,
       VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, m_vertexBuffer.get(), 0, vbSize}};
  if (this->hasIndices()) {
    preUploadBarriers.emplace_back(vk::PipelineStageFlagBits2::eNone, vk::AccessFlagBits2::eNone,
                                   vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
                                   VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, m_indexBuffer.get(), 0, ibSize);
    postUploadBarriers.emplace_back(vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
                                    vk::PipelineStageFlagBits2::eIndexInput, vk::AccessFlagBits2::eIndexRead,
                                    VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, m_indexBuffer.get(), 0, ibSize);
  }

  cmdBuffer->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
  cmdBuffer->pipelineBarrier2(vk::DependencyInfo({}, {}, preUploadBarriers));
  cmdBuffer->copyBuffer(m_uploadBuffer.get(), m_vertexBuffer.get(), {{0, 0, vbSize}});
  if (this->hasIndices()) {
    cmdBuffer->copyBuffer(m_uploadBuffer.get(), m_indexBuffer.get(), {{vbSize, 0, ibSize}});
  }
  cmdBuffer->pipelineBarrier2(vk::DependencyInfo({}, {}, postUploadBarriers));
  cmdBuffer->end();
  vk::CommandBufferSubmitInfo cmdBufferSubmit(cmdBuffer.get(), p_deviceMask);
  p_vkDevice.getQueue(p_queueFamilyIndex, 0).submit2(vk::SubmitInfo2({}, {}, cmdBufferSubmit));
  p_vkDevice.getQueue(p_queueFamilyIndex, 0).waitIdle();
  m_uploaded = true;
}
} // namespace xrmg