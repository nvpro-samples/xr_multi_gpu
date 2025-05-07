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
#include "Scene.hpp"

#include "App.hpp"

#include "shaders.hpp"

namespace xrmg {
static const std::vector<vk::VertexInputAttributeDescription> g_vertexInputAttributeDescs = {
    {0, 0, vk::Format::eR32G32B32Sfloat, offsetof(TriangleMesh::Vertex, pos)},
    {1, 0, vk::Format::eR32G32B32Sfloat, offsetof(TriangleMesh::Vertex, normal)},
    {2, 0, vk::Format::eR32G32Sfloat, offsetof(TriangleMesh::Vertex, tex)},
    {3, 1, vk::Format::eR32G32B32A32Sfloat, offsetof(Instance, modelToWorld.rows[0])},
    {4, 1, vk::Format::eR32G32B32A32Sfloat, offsetof(Instance, modelToWorld.rows[1])},
    {5, 1, vk::Format::eR32G32B32A32Sfloat, offsetof(Instance, modelToWorld.rows[2])},
    {6, 1, vk::Format::eR32G32B32A32Sfloat, offsetof(Instance, modelToWorld.rows[3])},
    {7, 1, vk::Format::eR32G32B32A32Sfloat, offsetof(Instance, modelToWorldIT.rows[0])},
    {8, 1, vk::Format::eR32G32B32A32Sfloat, offsetof(Instance, modelToWorldIT.rows[1])},
    {9, 1, vk::Format::eR32G32B32A32Sfloat, offsetof(Instance, modelToWorldIT.rows[2])},
    {10, 1, vk::Format::eR32G32B32A32Sfloat, offsetof(Instance, modelToWorldIT.rows[3])},
    {11, 1, vk::Format::eR32Uint, offsetof(Instance, colorHint)},
    {12, 1, vk::Format::eR32Sfloat, offsetof(Instance, relativeExtrusion)},
    {13, 1, vk::Format::eR32Sfloat, offsetof(Instance, absoluteExtrusion)},
};

static const std::vector<vk::VertexInputBindingDescription> g_vertexInputBindingDescs = {
    {0, sizeof(TriangleMesh::Vertex), vk::VertexInputRate::eVertex},
    {1, sizeof(Instance), vk::VertexInputRate::eInstance},
};

struct Camera {
  Mat4x4f view;
  Mat4x4f projection;
};

const uint32_t MAX_TORUS_INSTANCE_COUNT =
    8 * Scene::MAX_BASE_TORUS_COUNT * Scene::MAX_BASE_TORUS_COUNT * Scene::MAX_TORUS_LAYER_COUNT;

Scene::Scene(const Renderer &p_renderer) : m_renderer(p_renderer), m_currentBufferIndex(0) {
  vk::UniqueShaderModule layeredMeshModule = p_renderer.vkDevice().createShaderModuleUnique({{}, g_layeredMeshSrc});
  std::vector<vk::PipelineShaderStageCreateInfo> stages = {
      {{}, vk::ShaderStageFlagBits::eVertex, layeredMeshModule.get(), "main"},
      {{}, vk::ShaderStageFlagBits::eFragment, layeredMeshModule.get(), "main"}};

  vk::PipelineVertexInputStateCreateInfo vertexInputState({}, g_vertexInputBindingDescs, g_vertexInputAttributeDescs);
  vk::PipelineInputAssemblyStateCreateInfo inputAssemblyState({}, vk::PrimitiveTopology::eTriangleStrip, true);
  vk::PipelineViewportStateCreateInfo viewportState({}, 1, nullptr, 1, nullptr);
  vk::PipelineRasterizationStateCreateInfo rasterizationState(
      {}, false, false, vk::PolygonMode::eFill, vk::CullModeFlagBits::eNone, vk::FrontFace::eCounterClockwise, false,
      0.0f, 0.0f, 0.0f, 1.0f);
  vk::PipelineMultisampleStateCreateInfo multisampleState({}, vk::SampleCountFlagBits::e1);
  vk::PipelineDepthStencilStateCreateInfo depthStencilState({}, true, true, vk::CompareOp::eLess, false, false);
  vk::PipelineColorBlendAttachmentState blendAttachment(false);
  blendAttachment.setColorWriteMask(vk::FlagTraits<vk::ColorComponentFlagBits>::allFlags);
  vk::PipelineColorBlendStateCreateInfo colorBlendState({}, false, {}, blendAttachment);
  std::vector<vk::DynamicState> dynamicStates = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
  vk::PipelineDynamicStateCreateInfo dynamicState({}, dynamicStates);
  std::vector<vk::DescriptorSetLayoutBinding> bindings = {};
  vk::UniqueDescriptorSetLayout setLayout = p_renderer.vkDevice().createDescriptorSetLayoutUnique({{}, bindings});
  std::vector<vk::PushConstantRange> pushConstantRanges = {{vk::ShaderStageFlagBits::eVertex, 0, sizeof(Camera)}};
  m_pipelineLayout = p_renderer.vkDevice().createPipelineLayoutUnique({{}, setLayout.get(), pushConstantRanges});
  vk::StructureChain pipelineCreateChain(
      vk::GraphicsPipelineCreateInfo({}, stages, &vertexInputState, &inputAssemblyState, nullptr, &viewportState,
                                     &rasterizationState, &multisampleState, &depthStencilState, &colorBlendState,
                                     &dynamicState, m_pipelineLayout.get()),
      vk::PipelineRenderingCreateInfo(0, g_renderFormat, g_depthFormat));
  auto [createPipelineResult, pipeline] =
      p_renderer.vkDevice().createGraphicsPipelineUnique(p_renderer.getPipelineCache(), pipelineCreateChain.get());
  XRMG_ASSERT(createPipelineResult == vk::Result::eSuccess, "Pipeline creation failed.");
  m_pipeline = std::move(pipeline);

  std::optional<uint32_t> uploadMemTypeIndex = p_renderer.queryCompatibleMemoryTypeIndex(
      0, vk::MemoryPropertyFlagBits::eHostCached | vk::MemoryPropertyFlagBits::eHostCoherent |
             vk::MemoryPropertyFlagBits::eHostVisible);
  XRMG_ASSERT(uploadMemTypeIndex, "No host cached, visible, and coherent memory type for upload buffers available.");
  m_uploadMemPool.size = 2ull << 30;
  m_uploadMemPool.memory =
      p_renderer.vkDevice().allocateMemoryUnique({m_uploadMemPool.size, uploadMemTypeIndex.value()});
  m_uploadMemPool.mapped =
      reinterpret_cast<char *>(p_renderer.vkDevice().mapMemory(m_uploadMemPool.memory.get(), 0, m_uploadMemPool.size));
  m_uploadBuffer = p_renderer.vkDevice().createBufferUnique(
      {{}, m_uploadMemPool.size, vk::BufferUsageFlagBits::eTransferSrc, vk::SharingMode::eExclusive, {}});
  p_renderer.vkDevice().bindBufferMemory(m_uploadBuffer.get(), m_uploadMemPool.memory.get(), 0);

  this->pushTriangleMeshSingleInstance(&TriangleMesh::createPlaneXZ, Mat4x4f::createScaling(4.0f));
  m_projectionPlane = this->pushTriangleMeshSingleInstance(TriangleMesh::createPlaneXZ, Mat4x4f::IDENTITY);

  XRMG_INFO("Max torus instance count: {}", MAX_TORUS_INSTANCE_COUNT);
}

Scene::TriangleMeshIndex Scene::getTorusMeshIndex(uint32_t p_baseTesselationCount) {
  if (auto it = m_torusLods.find(p_baseTesselationCount); it != m_torusLods.end()) {
    return it->second;
  }
  float minorRadius = 0.05f;
  float majorRadius = 0.45f;
  TriangleMeshIndex idx = this->pushTriangleMesh(
      [&](const Renderer &p_renderer, uint32_t p_maxInstances, size_t p_sizePerInstance) {
        return TriangleMesh::createTorusXY(p_renderer, p_maxInstances, p_sizePerInstance, p_baseTesselationCount,
                                           minorRadius, majorRadius);
      },
      MAX_TORUS_INSTANCE_COUNT);
  return m_torusLods.emplace(p_baseTesselationCount, idx).first->second;
}

void Scene::buildCage(uint32_t p_baseTorusTesselationCount, uint32_t p_baseTorusCount, uint32_t p_torusLayerCount) {
  uint32_t torusCount = 8 * p_baseTorusCount * p_baseTorusCount * p_torusLayerCount;
  uint32_t triangleCount = torusCount * 4 * p_baseTorusTesselationCount * p_baseTorusTesselationCount;
  std::string triangleCountStr =
      triangleCount < 1000000 ? std::to_string(triangleCount) : std::format("{}M", triangleCount / 1000000);
  XRMG_INFO("base torus tesselation: {}, base torus count: {}, torus layer count: {} -> {} instances, {} triangles",
            p_baseTorusTesselationCount, p_baseTorusCount, p_torusLayerCount, torusCount, triangleCountStr);
  for (auto &[baseTesselationCount, torusMeshIndex] : m_torusLods) {
    m_triangleMeshes[torusMeshIndex].instanceCount = 0;
  }
  TriangleMeshIndex torusMeshIndex = this->getTorusMeshIndex(p_baseTorusTesselationCount);
  float torusMaxExtrusion = 0.03f;
  for (uint32_t i = 0; i < 4; ++i) {
    float scaling = 8.0f / static_cast<float>(p_baseTorusCount);
    Mat4x4f transform = Mat4x4f::createRotationY(static_cast<float>(i) * Angle::deg(90.0f)) *
                        Mat4x4f::createTranslation(0.0f, 4.0f, 4.0f) * Mat4x4f::createScaling(scaling);
    this->createChainMailPlane(torusMeshIndex, p_baseTorusCount, 2 * p_baseTorusCount, p_torusLayerCount,
                               torusMaxExtrusion, transform);
  }
}

void Scene::createChainMailPlane(TriangleMeshIndex p_torusMeshIndex, uint32_t p_horizontalTorusCount,
                                 uint32_t p_verticalTorusCount, uint32_t p_layerCount, float p_maxExtrusion,
                                 const Mat4x4f &p_transform) {
  for (uint32_t i = 0; i < p_horizontalTorusCount; ++i) {
    for (uint32_t j = 0; j < p_verticalTorusCount; ++j) {
      float x = static_cast<float>(i) - 0.5f * static_cast<float>(p_horizontalTorusCount - 1);
      float y = static_cast<float>(j) - 0.5f * static_cast<float>(p_verticalTorusCount - 1);
      Mat4x4f finalTransform =
          p_transform * Mat4x4f::createTranslation(x + 0.5f * static_cast<float>(j % 2), 0.5f * y, 0.0f) *
          Mat4x4f::createRotation({}, (2.0f * static_cast<float>(j % 2) - 1.0f) * Angle::deg(20.0f), {});
      this->pushFurryTriangleMeshInstances(p_torusMeshIndex, p_layerCount, p_maxExtrusion, finalTransform);
    }
  }
}

void Scene::updateProjectionPlane(const Mat4x4f &p_cameraPose, Angle p_verticalFov, float p_aspectRatio,
                                  float p_projectionPlaneDistance) {
  float scaleY = 0.99f * (0.5f * p_verticalFov).tan() * p_projectionPlaneDistance;
  float scaleX = p_aspectRatio * scaleY;
  this->getTriangleMeshIntance(m_projectionPlane.first, m_projectionPlane.second)
      .setTransform(p_cameraPose * Mat4x4f::createTranslation(0.0f, 0.0f, -p_projectionPlaneDistance) *
                    Mat4x4f::createRotationX(Angle::deg(90.0f)) * Mat4x4f::createScaling(scaleX, 1.0f, scaleY));
}

template <typename T> Scene::MemPoolAllocation<T> Scene::VulkanMemPool::allocate(uint32_t p_count) {
  XRMG_ASSERT(freePtr + p_count * sizeof(T) <= size, "Out of mem pool memory.");
  auto elements = reinterpret_cast<T *>(mapped + freePtr);
  size_t memOffset = freePtr;
  freePtr += p_count * sizeof(T);
  return {.count = p_count, .elements = elements, .memOffset = memOffset};
}

Scene::TriangleMeshIndex Scene::pushTriangleMesh(TriangleMeshCreator p_creator, uint32_t p_maxInstances) {
  XRMG_ASSERT(p_maxInstances < 1u << 24, "Max instances ({}) must be less than {}.", p_maxInstances, 1u << 24);
  TriangleMeshContainer &triMeshContainer = m_triangleMeshes.emplace_back(
      TriangleMeshContainer{.triMesh = p_creator(m_renderer, p_maxInstances, sizeof(Instance))});
  for (uint32_t i = 0; i < MAX_QUEUED_FRAMES; ++i) {
    triMeshContainer.instances[i] = m_uploadMemPool.allocate<Instance>(p_maxInstances);
  }
  triMeshContainer.triMesh.upload(m_renderer.vkDevice(), m_renderer.getGraphicsQueueFamilyIndex(),
                                  m_renderer.getDeviceMaskAll());
  return static_cast<TriangleMeshIndex>(m_triangleMeshes.size() - 1);
}

Scene::TriangleMeshInstanceIndex Scene::pushTriangleMeshInstance(TriangleMeshIndex p_triangleMeshIndex,
                                                                 const Mat4x4f &p_modelToWorld) {
  XRMG_ASSERT(p_triangleMeshIndex < m_triangleMeshes.size(),
              "Triangle mesh index({}) must be less than number of triangle meshes ({}).", p_triangleMeshIndex,
              m_triangleMeshes.size());
  XRMG_ASSERT(m_triangleMeshes[p_triangleMeshIndex].instanceCount <
                  m_triangleMeshes[p_triangleMeshIndex].triMesh.getMaxInstances(),
              "Too many instances.");
  auto instanceIdx = static_cast<TriangleMeshInstanceIndex>(m_triangleMeshes[p_triangleMeshIndex].instanceCount++);
  this->getTriangleMeshIntance(p_triangleMeshIndex,
                               instanceIdx) = {.modelToWorld = p_modelToWorld,
                                               .modelToWorldIT = p_modelToWorld.invert().transpose(),
                                               .colorHint = p_triangleMeshIndex ^ instanceIdx};
  return instanceIdx;
}

void Scene::pushFurryTriangleMeshInstances(TriangleMeshIndex p_triangleMeshIndex, uint32_t p_layerCount,
                                           float p_maxExtrusion, const Mat4x4f &p_modelToWorld) {
  uint32_t colorHint = static_cast<uint32_t>(rand());
  for (uint32_t k = 0; k < p_layerCount; ++k) {
    TriangleMeshInstanceIndex instanceIdx = this->pushTriangleMeshInstance(p_triangleMeshIndex, p_modelToWorld);
    Instance &instance = this->getTriangleMeshIntance(p_triangleMeshIndex, instanceIdx);
    instance.colorHint = colorHint;
    instance.relativeExtrusion = static_cast<float>(k) / static_cast<float>(p_layerCount);
    instance.absoluteExtrusion = p_maxExtrusion * instance.relativeExtrusion;
  }
}

Instance &Scene::getTriangleMeshIntance(TriangleMeshIndex p_triangleMeshIndex,
                                        TriangleMeshInstanceIndex p_instanceIndex) {
  XRMG_ASSERT(p_triangleMeshIndex < m_triangleMeshes.size(),
              "Triangle mesh index({}) must be less than number of triangle meshes ({}).", p_triangleMeshIndex,
              m_triangleMeshes.size());
  XRMG_ASSERT(p_instanceIndex < m_triangleMeshes[p_triangleMeshIndex].instanceCount,
              "Triangle mesh instance index ({}) must be less than number of instances of triangle mesh ({}).",
              p_instanceIndex, m_triangleMeshes[p_triangleMeshIndex].instanceCount);
  return m_triangleMeshes[p_triangleMeshIndex].instances[m_currentBufferIndex].elements[p_instanceIndex];
}

std::pair<Scene::TriangleMeshIndex, Scene::TriangleMeshInstanceIndex>
Scene::pushTriangleMeshSingleInstance(TriangleMeshCreator p_creator, const Mat4x4f &p_localToGlobal,
                                      uint32_t p_maxInstances) {
  TriangleMeshIndex triMeshIdx = this->pushTriangleMesh(p_creator, p_maxInstances);
  TriangleMeshInstanceIndex instanceIdx = this->pushTriangleMeshInstance(triMeshIdx, p_localToGlobal);
  return {triMeshIdx, instanceIdx};
}

void Scene::clearTriangleMeshInstances(TriangleMeshIndex p_triMeshIndex) {
  m_triangleMeshes[p_triMeshIndex].instanceCount = 0;
}

void Scene::update(float p_millis) {
  uint32_t prevBufferIndex = m_currentBufferIndex;
  m_currentBufferIndex = (m_currentBufferIndex + 1) % MAX_QUEUED_FRAMES;
  for (TriangleMeshContainer &triMeshContainer : m_triangleMeshes) {
    if (triMeshContainer.enabled && triMeshContainer.instanceCount != 0) {
      memcpy(triMeshContainer.instances[m_currentBufferIndex].elements,
             triMeshContainer.instances[prevBufferIndex].elements, triMeshContainer.instanceCount * sizeof(Instance));
    }
  }
  m_triangleMeshes[m_projectionPlane.first].enabled = g_app->getOptions().renderProjectionPlane;
}

void Scene::render(uint32_t p_physicalDeviceIndex, vk::CommandBuffer p_cmdBuffer, vk::ImageView p_colorDest,
                   vk::ImageView p_depthDest, const vk::Rect2D &p_renderArea, const vk::Viewport &p_viewport,
                   const Mat4x4f &p_view, const Mat4x4f &p_projection) {
  std::vector<vk::BufferMemoryBarrier2> preUploadBarriers;
  if (g_app->getCurrentFrameIndex() == 0) {
    preUploadBarriers.emplace_back(vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eHostWrite,
                                   vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead,
                                   VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, m_uploadBuffer.get(), 0,
                                   m_uploadMemPool.size);
  }
  std::vector<vk::BufferMemoryBarrier2> postUploadBarriers;
  for (TriangleMeshContainer &triMeshContainer : m_triangleMeshes) {
    if (triMeshContainer.enabled && triMeshContainer.instanceCount != 0) {
      preUploadBarriers.emplace_back(
          vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eNone, vk::PipelineStageFlagBits2::eTransfer,
          vk::AccessFlagBits2::eTransferWrite, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
          triMeshContainer.triMesh.getInstanceBuffer(), 0, triMeshContainer.instanceCount * sizeof(Instance));
      postUploadBarriers.emplace_back(vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
                                      vk::PipelineStageFlagBits2::eVertexInput,
                                      vk::AccessFlagBits2::eVertexAttributeRead, VK_QUEUE_FAMILY_IGNORED,
                                      VK_QUEUE_FAMILY_IGNORED, triMeshContainer.triMesh.getInstanceBuffer(), 0,
                                      triMeshContainer.instanceCount * sizeof(Instance));
    }
  }
  p_cmdBuffer.pipelineBarrier2({{}, {}, preUploadBarriers});
  for (TriangleMeshContainer &triMeshContainer : m_triangleMeshes) {
    if (triMeshContainer.enabled && triMeshContainer.instanceCount != 0) {
      p_cmdBuffer.copyBuffer(m_uploadBuffer.get(), triMeshContainer.triMesh.getInstanceBuffer(),
                             vk::BufferCopy(triMeshContainer.instances[m_currentBufferIndex].memOffset, 0,
                                            triMeshContainer.instanceCount * sizeof(Instance)));
    }
  }
  p_cmdBuffer.pipelineBarrier2({{}, {}, postUploadBarriers});

  vk::RenderingAttachmentInfo colorAttachment(
      p_colorDest, vk::ImageLayout::eColorAttachmentOptimal, vk::ResolveModeFlagBits::eNone, nullptr,
      vk::ImageLayout::eUndefined, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, g_clearValues);
  vk::RenderingAttachmentInfo depthAttachment(p_depthDest, vk::ImageLayout::eDepthAttachmentOptimal, {}, {}, {},
                                              vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore,
                                              vk::ClearValue(vk::ClearDepthStencilValue(1.0f, 0)));
  p_cmdBuffer.beginRendering(vk::RenderingInfo({}, p_renderArea, 1, 0, colorAttachment, &depthAttachment, nullptr));
  p_cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, m_pipeline.get());
  p_cmdBuffer.setViewport(0, p_viewport);
  p_cmdBuffer.setScissor(0, p_renderArea);
  p_cmdBuffer.pushConstants(m_pipelineLayout.get(), vk::ShaderStageFlagBits::eVertex, offsetof(Camera, view),
                            sizeof(Mat4x4f), &p_view);
  p_cmdBuffer.pushConstants(m_pipelineLayout.get(), vk::ShaderStageFlagBits::eVertex, offsetof(Camera, projection),
                            sizeof(Mat4x4f), &p_projection);
  for (TriangleMeshContainer &triMeshContainer : m_triangleMeshes) {
    if (triMeshContainer.enabled && triMeshContainer.instanceCount != 0) {
      triMeshContainer.triMesh.bind(p_cmdBuffer);
      triMeshContainer.triMesh.draw(p_cmdBuffer, triMeshContainer.instanceCount);
    }
  }
  p_cmdBuffer.endRendering();
}
} // namespace xrmg
