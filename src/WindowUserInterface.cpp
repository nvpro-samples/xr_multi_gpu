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
#include "WindowUserInterface.hpp"

#include "App.hpp"
#include "Scene.hpp"

#include "shaders.hpp"

#define CROSS_EYE_COMPATIBLE_RENDERING

namespace xrmg {
const Angle VERTICAL_FOV = Angle::deg(60.0f);

vk::UniqueInstance WindowUserInterface::createVkInstance(const vk::InstanceCreateInfo &p_createInfo) {
  std::vector<const char *> instanceLayers(p_createInfo.ppEnabledLayerNames,
                                           p_createInfo.ppEnabledLayerNames + p_createInfo.enabledLayerCount);

  std::vector<const char *> instanceExtensions(
      p_createInfo.ppEnabledExtensionNames, p_createInfo.ppEnabledExtensionNames + p_createInfo.enabledExtensionCount);
  instanceExtensions.emplace_back("VK_KHR_surface");
  instanceExtensions.emplace_back("VK_KHR_get_surface_capabilities2");
#ifdef _WIN32
  instanceExtensions.emplace_back("VK_KHR_win32_surface");
#endif

  return vk::createInstanceUnique({
      p_createInfo.flags,
      p_createInfo.pApplicationInfo,
      instanceLayers,
      instanceExtensions,
      p_createInfo.pNext,
  });
}

std::optional<uint32_t> WindowUserInterface::queryMainPhysicalDevice(vk::Instance p_vkInstance,
                                                                     uint32_t p_presentQueueFamilyIndex,
                                                                     uint32_t p_candidateCount,
                                                                     vk::PhysicalDevice *p_candidates) {
  m_window.createSurface(p_vkInstance);
  this->printSurfaceCapabilities(p_presentQueueFamilyIndex, p_candidateCount, p_candidates);
  auto findIt = std::find_if(p_candidates, p_candidates + p_candidateCount, [&](const auto &dev) {
    return dev.getSurfaceSupportKHR(p_presentQueueFamilyIndex, m_window.getVulkanSurface());
  });
  XRMG_ASSERT(findIt != p_candidates + p_candidateCount,
              "No physical device of selected device group supports the window's surface.");
  return static_cast<uint32_t>(findIt - p_candidates);
}

void WindowUserInterface::initialize(const Renderer &p_renderer, uint32_t p_presentQueueFamilyIndex,
                                     uint32_t p_presentQueueIndex) {
  m_window.createSwapchain(p_renderer.vkDevice(), g_app->getOptions().swapchainFormat,
                           g_app->getOptions().swapchainImageCount, g_app->getOptions().presentMode);
  for (vk::UniqueSemaphore &sem : m_swapchainImageReadySemaphores) {
    sem = p_renderer.vkDevice().createSemaphoreUnique({});
  }
  m_frameReadySemaphore = p_renderer.vkDevice().createSemaphoreUnique({});
  this->buildProjections();
}

vk::Extent2D WindowUserInterface::getResolutionPerEye() {
  return {m_window.getSwapchainImageSize().width / 2, m_window.getSwapchainImageSize().height};
}

UserInterface::FrameInfo WindowUserInterface::beginFrame() {
  auto now = std::chrono::high_resolution_clock::now();
  if (!m_lastBeginFrame) {
    m_lastBeginFrame = now;
  }
  FrameInfo frameInfo = {.predictedDisplayTimeNanos = m_lastBeginFrame.value().time_since_epoch().count()};
  m_lastBeginFrame = now;
  return frameInfo;
}

WindowUserInterface::FrameRenderTargets WindowUserInterface::acquireSwapchainImages(vk::Device p_device) {
  m_currentSwapchainImageIndex = m_window.acquireNextImageIndex(p_device, this->getSwapchainImageReadySemaphore());
  return {m_window.getSwapchainImage(m_currentSwapchainImageIndex.value()), vk::ImageLayout::ePresentSrcKHR, {}, {}};
}

vk::Semaphore WindowUserInterface::getSwapchainImageReadySemaphore() {
  return m_swapchainImageReadySemaphores[m_swapchainImageReadySemaphoreIndex].get();
}

void WindowUserInterface::releaseSwapchainImage() {
  m_swapchainImageReadySemaphoreIndex = (m_swapchainImageReadySemaphoreIndex + 1) % MAX_QUEUED_FRAMES;
}

void WindowUserInterface::update(float p_millis) {
  if (!g_app->isPaused()) {
    m_runtimeMillis += p_millis;
  }
  if (m_camMoveDir.x != 0.0f || m_camMoveDir.y != 0.0f || m_camMoveDir.z != 0.0f) {
    m_camPos += Mat4x4f::createRotation({}, m_camPitch, m_camYaw)
                    .transformDir((m_fast ? 1e-2f : 1e-3f) * p_millis * m_camMoveDir.normalized());
  }
}

bool WindowUserInterface::onKeyDown(int32_t p_key) {
#ifdef _WIN32
  switch (p_key) {
  case VirtualKey::PAUSE: g_app->togglePaused(); break;
  case VirtualKey::ESCAPE: g_app->discontinue(); break;
  case 'W': m_camMoveDir.z = -1.0f; break;
  case 'S': m_camMoveDir.z = 1.0f; break;
  case 'A': m_camMoveDir.x = -1.0f; break;
  case 'D': m_camMoveDir.x = 1.0f; break;
  case VirtualKey::SPACE: m_camMoveDir.y = 1.0f; break;
  case 'C': m_camMoveDir.y = -1.0f; break;
  case VirtualKey::SHIFT: m_fast = false; break;
  case VirtualKey::ADD:
    m_ipd += 0.005f;
    this->buildProjections();
    break;
  case VirtualKey::SUBTRACT:
    m_ipd = std::max(0.0f, m_ipd - 0.005f);
    this->buildProjections();
    break;
  default: return false;
  }
  return true;
#else
  return false;
#endif
}

bool WindowUserInterface::onKeyUp(int32_t p_key) {
#ifdef _WIN32
  switch (p_key) {
  case 'W': m_camMoveDir.z = 0.0f; break;
  case 'S': m_camMoveDir.z = 0.0f; break;
  case 'A': m_camMoveDir.x = 0.0f; break;
  case 'D': m_camMoveDir.x = 0.0f; break;
  case VirtualKey::SPACE: m_camMoveDir.y = 0.0f; break;
  case 'C': m_camMoveDir.y = 0.0f; break;
  case VirtualKey::SHIFT: m_fast = true; break;
  default: return false;
  }
  return true;
#else
  return false;
#endif
}

bool WindowUserInterface::onMouseMove(int32_t p_deltaX, int32_t p_deltaY) {
  m_camPitch =
      std::clamp(m_camPitch + Angle::rad(1e-3f * static_cast<float>(-p_deltaY)), Angle::deg(-90.0f), Angle::deg(90.0f));
  m_camYaw += Angle::rad(1e-3f * static_cast<float>(-p_deltaX));
  return true;
}

bool WindowUserInterface::onWheelMove(int32_t p_delta) {
  m_projectionPlaneDistance = std::clamp(
      m_projectionPlaneDistance + static_cast<float>(p_delta) / static_cast<float>(WHEEL_DELTA) * 0.2f, 0.2f, 10.0f);
  this->buildProjections();
  return true;
}

void WindowUserInterface::buildProjections() {
  XRMG_INFO("IPD: {:.3f}, projection plane distance: {:.1f}", m_ipd, m_projectionPlaneDistance);
  float aspectRatio = this->getAspectRatioPerEye();
  m_projections[StereoProjection::Eye::LEFT] = StereoProjection::create(
      StereoProjection::Eye::LEFT, m_ipd, m_projectionPlaneDistance, VERTICAL_FOV, aspectRatio, 1e-2f, 1e2f);
  m_projections[StereoProjection::Eye::RIGHT] = StereoProjection::create(
      StereoProjection::Eye::RIGHT, m_ipd, m_projectionPlaneDistance, VERTICAL_FOV, aspectRatio, 1e-2f, 1e2f);
}

Mat4x4f WindowUserInterface::getCurrentFrameView(StereoProjection::Eye p_eye) {
  Mat4x4f cameraPose = Mat4x4f::createTranslation(m_camPos) * Mat4x4f::createRotation({}, m_camPitch, m_camYaw);
  if (!g_app->isPaused()) {
    Mat4x4f rotation = Mat4x4f::createRotation({}, Angle::deg(-30.0f), Angle::deg(45.0f * 1e-4f * m_runtimeMillis));
    Mat4x4f translation = Mat4x4f::createTranslation(0.0f, 2.0f, 12.0f);
    cameraPose = rotation * translation;
  }

  g_app->getScene().updateProjectionPlane(cameraPose, VERTICAL_FOV, this->getAspectRatioPerEye(),
                                          m_projectionPlaneDistance);

  Mat4x4f eyeTranslation = StereoProjection::createStereoEyeTranslation(p_eye, m_ipd);
  return (cameraPose * eyeTranslation).invert();
}

StereoProjection WindowUserInterface::getCurrentFrameProjection(StereoProjection::Eye p_eye) {
  return m_projections[p_eye];
}

void WindowUserInterface::endFrame(vk::Queue p_presentGraphicsQueue) {
  m_window.present(p_presentGraphicsQueue, m_currentSwapchainImageIndex.value(), m_frameReadySemaphore.get());
}

float WindowUserInterface::getAspectRatioPerEye() {
  return static_cast<float>(this->getResolutionPerEye().width) / static_cast<float>(this->getResolutionPerEye().height);
}

void WindowUserInterface::printSurfaceCapabilities(uint32_t p_presentQueueFamilyIndex, uint32_t p_physicalDeviceCount,
                                                   vk::PhysicalDevice *p_physicalDevices) const {
  XRMG_INFO("Physical devices surface capabilities");
  for (uint32_t devIdx = 0; devIdx < p_physicalDeviceCount; ++devIdx) {
    vk::PhysicalDevice dev = p_physicalDevices[devIdx];
    if (dev.getSurfaceSupportKHR(p_presentQueueFamilyIndex, m_window.getVulkanSurface())) {
      vk::PhysicalDeviceSurfaceInfo2KHR surfaceInfo(m_window.getVulkanSurface());
      auto caps = dev.getSurfaceCapabilities2KHR(surfaceInfo);
      auto formats = dev.getSurfaceFormats2KHR(surfaceInfo);
      auto presentModes = dev.getSurfacePresentModesKHR(m_window.getVulkanSurface());
      XRMG_INFO(" [{}] {}; image count: {{{},...,{}}}, current extent: {}x{}", devIdx,
                dev.getProperties().deviceName.data(), caps.surfaceCapabilities.minImageCount,
                caps.surfaceCapabilities.maxImageCount, caps.surfaceCapabilities.currentExtent.width,
                caps.surfaceCapabilities.currentExtent.height);
      XRMG_INFO("  ├─╴Supported formats:{}", formats.empty() ? " none" : "");
      for (uint32_t i = 0; i < formats.size(); ++i) {
        XRMG_INFO("  │  {}─╴{}, {}", i == formats.size() - 1 ? "└" : "├",
                  vk::to_string(formats[i].surfaceFormat.colorSpace), vk::to_string(formats[i].surfaceFormat.format));
      }
      XRMG_INFO("  └─╴Supported present modes:", presentModes.empty() ? " none" : "");
      for (uint32_t i = 0; i < presentModes.size(); ++i) {
        XRMG_INFO("     {}─╴{}", i == presentModes.size() - 1 ? "└" : "├", vk::to_string(presentModes[i]));
      }
    } else {
      XRMG_INFO(" [{}] not supported", devIdx);
    }
  }
}
} // namespace xrmg