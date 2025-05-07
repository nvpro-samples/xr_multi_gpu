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

#include "Instance.hpp"
#include "Renderer.hpp"
#include "TriangleMesh.hpp"

#include <functional>

namespace xrmg {
class Scene {
public:
  typedef std::function<TriangleMesh(const Renderer &p_renderer, uint32_t p_maxInstances, size_t p_sizePerInstance)>
      TriangleMeshCreator;
  typedef uint16_t TriangleMeshIndex;
  typedef uint32_t TriangleMeshInstanceIndex;

  static const uint32_t MAX_BASE_TORUS_COUNT = 64;
  static const uint32_t MAX_TORUS_LAYER_COUNT = 16;

  Scene(const Renderer &p_renderer);

  void update(float p_millis);
  void render(uint32_t p_physicalDeviceIndex, vk::CommandBuffer p_cmdBuffer, vk::ImageView p_colorDest,
              vk::ImageView p_depthDest, const vk::Rect2D &p_renderArea, const vk::Viewport &p_viewport,
              const Mat4x4f &p_view, const Mat4x4f &p_projection);

  TriangleMeshIndex pushTriangleMesh(TriangleMeshCreator p_creator, uint32_t p_maxInstances);
  TriangleMeshInstanceIndex pushTriangleMeshInstance(TriangleMeshIndex p_triangleMeshIndex,
                                                     const Mat4x4f &p_modelToWorld = Mat4x4f::IDENTITY);
  void pushFurryTriangleMeshInstances(TriangleMeshIndex p_triangleMeshIndex, uint32_t p_layerCount,
                                      float p_maxExtrusion, const Mat4x4f &p_modelToWorld = Mat4x4f::IDENTITY);
  Instance &getTriangleMeshIntance(TriangleMeshIndex p_triangleMeshIndex, TriangleMeshInstanceIndex p_instanceIndex);
  std::pair<TriangleMeshIndex, TriangleMeshInstanceIndex> pushTriangleMeshSingleInstance(TriangleMeshCreator p_creator,
                                                                                         const Mat4x4f &p_localToGlobal,
                                                                                         uint32_t p_maxInstances = 1);
  void updateProjectionPlane(const Mat4x4f &p_cameraPose, Angle p_verticalFov, float p_aspectRatio,
                             float p_projectionPlaneDistance);
  void clearTriangleMeshInstances(TriangleMeshIndex p_triMeshIndex);
  void buildCage(uint32_t p_baseTorusTesselationCount, uint32_t p_baseTorusCount, uint32_t p_torusLayerCount);

private:
  template <typename T> struct MemPoolAllocation {
    uint32_t count;
    T *elements = nullptr;
    size_t memOffset = 0;
  };

  struct VulkanMemPool {
    size_t size = 0;
    vk::UniqueDeviceMemory memory = {};
    size_t freePtr = 0;
    char *mapped = nullptr;

    template <typename T> MemPoolAllocation<T> allocate(uint32_t p_count);
  };

  struct TriangleMeshContainer {
    TriangleMesh triMesh;
    bool enabled = true;
    uint32_t instanceCount = 0;
    std::array<MemPoolAllocation<Instance>, MAX_QUEUED_FRAMES> instances = {};
  };

  const Renderer &m_renderer;
  vk::UniquePipelineLayout m_pipelineLayout;
  vk::UniquePipeline m_pipeline;
  std::vector<TriangleMeshContainer> m_triangleMeshes;
  VulkanMemPool m_uploadMemPool;
  vk::UniqueBuffer m_uploadBuffer;
  uint32_t m_currentBufferIndex;
  std::pair<TriangleMeshIndex, TriangleMeshInstanceIndex> m_projectionPlane;
  std::unordered_map<uint32_t, TriangleMeshIndex> m_torusLods;

  void createChainMailPlane(TriangleMeshIndex p_torusMeshIndex, uint32_t p_horizontalTorusCount,
                            uint32_t p_verticalTorusCount, uint32_t p_layerCount, float p_maxExtrusion,
                            const Mat4x4f &p_transform);
  TriangleMeshIndex getTorusMeshIndex(uint32_t p_baseTesselationCount);
};
} // namespace xrmg
