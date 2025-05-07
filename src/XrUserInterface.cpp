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
#include "XrUserInterface.hpp"

#include "App.hpp"

#include <thread>

namespace xrmg {
template <typename R, typename... Args>
void setXrFunction(XrInstance p_instance, const std::string &p_fnName, R (*&p_dst)(Args...)) {
  PFN_xrVoidFunction fn;
  XrInstance m_instance = p_instance;
  XRMG_ASSERT_XR(xrGetInstanceProcAddr(p_instance, p_fnName.c_str(), &fn));
  p_dst = (R (*)(Args...))fn;
}

#define XRMG_SET_XR_FUNCTION(_fn) setXrFunction(m_instance, #_fn, _fn)

PFN_xrGetVulkanGraphicsRequirements2KHR xrGetVulkanGraphicsRequirements2KHR;
PFN_xrGetVulkanGraphicsDevice2KHR xrGetVulkanGraphicsDevice2KHR;
PFN_xrCreateVulkanInstanceKHR xrCreateVulkanInstanceKHR;

XrUserInterface::XrUserInterface(bool p_enableCoreValidation) : m_resolutionPerEye(0, 0) {
  std::vector<const char *> enabledLayers = {};
  if (p_enableCoreValidation) {
    enabledLayers.emplace_back("XR_APILAYER_LUNARG_core_validation");
  }
  std::vector<const char *> enabledExtensions = {"XR_KHR_vulkan_enable2"};
  XrInstanceCreateInfo instanceCreateInfo = {.type = XR_TYPE_INSTANCE_CREATE_INFO,
                                             .applicationInfo = {.applicationName = SAMPLE_NAME,
                                                                 .applicationVersion = 1,
                                                                 .engineName = SAMPLE_NAME,
                                                                 .engineVersion = 1,
                                                                 .apiVersion = XR_API_VERSION_1_0},
                                             .enabledApiLayerCount = static_cast<uint32_t>(enabledLayers.size()),
                                             .enabledApiLayerNames = enabledLayers.data(),
                                             .enabledExtensionCount = static_cast<uint32_t>(enabledExtensions.size()),
                                             .enabledExtensionNames = enabledExtensions.data()};
  XrResult r = xrCreateInstance(&instanceCreateInfo, &m_instance);
  XRMG_ASSERT(r == XrResult::XR_SUCCESS, "OpenXR instance creation failed.");
  XRMG_SET_XR_FUNCTION(xrGetVulkanGraphicsRequirements2KHR);
  XRMG_SET_XR_FUNCTION(xrGetVulkanGraphicsDevice2KHR);
  XRMG_SET_XR_FUNCTION(xrCreateVulkanInstanceKHR);
  XrSystemGetInfo systemGetInfo = {.type = XR_TYPE_SYSTEM_GET_INFO, .formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY};
  XRMG_ASSERT_XR(xrGetSystem(m_instance, &systemGetInfo, &m_systemId));
  XrSystemProperties sysProps = {.type = XR_TYPE_SYSTEM_PROPERTIES};
  XRMG_ASSERT_XR(xrGetSystemProperties(m_instance, m_systemId, &sysProps));
  XRMG_INFO("XR system name: {} (0x{:x}), max swapchain size: {}x{} (max layers: {}), position tracking: {}, "
            "orientation tracking: {}",
            sysProps.systemName, sysProps.vendorId, sysProps.graphicsProperties.maxSwapchainImageWidth,
            sysProps.graphicsProperties.maxSwapchainImageHeight, sysProps.graphicsProperties.maxLayerCount,
            XRMG_BOOL_TO_STRING(sysProps.trackingProperties.positionTracking),
            XRMG_BOOL_TO_STRING(sysProps.trackingProperties.orientationTracking));
  uint32_t viewConfigurationCount;
  XRMG_ASSERT_XR(xrEnumerateViewConfigurations(m_instance, m_systemId, 0, &viewConfigurationCount, nullptr));
  XRMG_ASSERT(viewConfigurationCount != 0, "No XR view configurations supported.");
  std::vector<XrViewConfigurationType> viewConfiurationTypes(viewConfigurationCount);
  XRMG_ASSERT_XR(xrEnumerateViewConfigurations(m_instance, m_systemId, viewConfigurationCount, &viewConfigurationCount,
                                               viewConfiurationTypes.data()));

  std::stringstream log;
  log << "XR view configurations" << std::endl;
  for (uint32_t i = 0; i < viewConfigurationCount; ++i) {
    log << (i == viewConfigurationCount - 1 ? "└╴" : "├╴");
    switch (viewConfiurationTypes[i]) {
    case XR_VIEW_CONFIGURATION_TYPE_PRIMARY_MONO: log << "primary mono"; break;
    case XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO: log << "primary stereo"; break;
    case XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO_WITH_FOVEATED_INSET:
      log << "primary stereo with foveated inset";
      break;
    case XR_VIEW_CONFIGURATION_TYPE_SECONDARY_MONO_FIRST_PERSON_OBSERVER_MSFT:
      log << "secondary mono first person observer msft";
      break;
    }
    XrViewConfigurationProperties viewConfProps = {.type = XR_TYPE_VIEW_CONFIGURATION_PROPERTIES};
    XRMG_ASSERT_XR(xrGetViewConfigurationProperties(m_instance, m_systemId, viewConfiurationTypes[i], &viewConfProps));
    log << std::format(", fovMutable: {}, pNext: {}", XRMG_BOOL_TO_STRING(viewConfProps.fovMutable),
                       XRMG_BOOL_TO_STRING(viewConfProps.next))
        << std::endl;
    uint32_t viewCount;
    XRMG_ASSERT_XR(
        xrEnumerateViewConfigurationViews(m_instance, m_systemId, viewConfiurationTypes[i], 0, &viewCount, nullptr));
    std::vector<XrViewConfigurationView> views(viewCount, {.type = XR_TYPE_VIEW_CONFIGURATION_VIEW});
    XRMG_ASSERT_XR(xrEnumerateViewConfigurationViews(m_instance, m_systemId, viewConfiurationTypes[i], viewCount,
                                                     &viewCount, views.data()));
    XRMG_WARN_UNLESS(viewConfiurationTypes[i] != XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO || viewCount == 2,
                     "Expected exactly two views for primary stereo view confugration type, but got {}", viewCount);
    for (uint32_t j = 0; j < viewCount; ++j) {
      XrViewConfigurationView view = views[j];
      log << "  " << (j == viewCount - 1 ? "└╴" : "├╴")
          << std::format(
                 "view {}, max [width: {}, height: {}, samples: {}], recommended [width: {}, height: {}, samples: {}]",
                 j, view.maxImageRectWidth, view.maxImageRectHeight, view.maxSwapchainSampleCount,
                 view.recommendedImageRectWidth, view.recommendedImageRectHeight, view.recommendedSwapchainSampleCount)
          << std::endl;
      if (viewConfiurationTypes[i] == XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO) {
        if (j == 0) {
          m_resolutionPerEye = vk::Extent2D(view.recommendedImageRectWidth, view.recommendedImageRectHeight);
        } else if (j == 1) {
          XRMG_WARN_UNLESS(m_resolutionPerEye.width == view.recommendedImageRectWidth &&
                               m_resolutionPerEye.height == view.recommendedImageRectHeight,
                           "Recommended image rect sizes differ between views.");
        }
      }
    }
  }
  XRMG_INFO("{}", log.str());
  XRMG_ASSERT(std::find(viewConfiurationTypes.begin(), viewConfiurationTypes.end(),
                        XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO) != viewConfiurationTypes.end(),
              "Primary stereo view configuration type is not supported.");
  if (g_app->getOptions().xrResolutionPerEye) {
    m_resolutionPerEye = g_app->getOptions().xrResolutionPerEye.value();
    XRMG_INFO("Overriding resolution per eye to: {}x{}", m_resolutionPerEye.width, m_resolutionPerEye.height);
  }
}

XrUserInterface::~XrUserInterface() {
  if (m_session) {
    xrDestroySession(m_session);
  }
  if (m_colorSwapchain.swapchain) {
    xrDestroySwapchain(m_colorSwapchain.swapchain);
  }
  if (m_depthSwapchain.swapchain) {
    xrDestroySwapchain(m_depthSwapchain.swapchain);
  }
  if (m_instance) {
    xrDestroyInstance(m_instance);
  }
}

vk::UniqueInstance XrUserInterface::createVkInstance(const vk::InstanceCreateInfo &p_createInfo) {
  XrGraphicsRequirementsVulkan2KHR graphicsRequirements = {.type = XR_TYPE_GRAPHICS_REQUIREMENTS_VULKAN2_KHR};
  XRMG_ASSERT_XR(xrGetVulkanGraphicsRequirements2KHR(m_instance, m_systemId, &graphicsRequirements));
  XRMG_INFO("min: {}.{}.{}, max: {}.{}.{}", XR_VERSION_MAJOR(graphicsRequirements.minApiVersionSupported),
            XR_VERSION_MINOR(graphicsRequirements.minApiVersionSupported),
            XR_VERSION_PATCH(graphicsRequirements.minApiVersionSupported),
            XR_VERSION_MAJOR(graphicsRequirements.maxApiVersionSupported),
            XR_VERSION_MINOR(graphicsRequirements.maxApiVersionSupported),
            XR_VERSION_PATCH(graphicsRequirements.maxApiVersionSupported));

  VkInstanceCreateInfo rawCreateInfo = p_createInfo;
  XrVulkanInstanceCreateInfoKHR xrVkInstanceCreateInfo = {
      .type = XR_TYPE_VULKAN_INSTANCE_CREATE_INFO_KHR,
      .systemId = m_systemId,
      .pfnGetInstanceProcAddr = &vkGetInstanceProcAddr,
      .vulkanCreateInfo = &rawCreateInfo,
  };
  VkInstance vkInstance;
  VkResult vkResult;
  XRMG_ASSERT_XR(xrCreateVulkanInstanceKHR(m_instance, &xrVkInstanceCreateInfo, &vkInstance, &vkResult));
  XRMG_ASSERT_VK(vkResult);
  return vk::UniqueInstance(vkInstance);
}

std::optional<uint32_t> XrUserInterface::queryMainPhysicalDevice(vk::Instance p_vkInstance, uint32_t p_queueFamilyIndex,
                                                                 uint32_t p_candidateCount,
                                                                 vk::PhysicalDevice *p_candidates) {
  XrVulkanGraphicsDeviceGetInfoKHR getInfo = {
      .type = XR_TYPE_VULKAN_GRAPHICS_DEVICE_GET_INFO_KHR,
      .systemId = m_systemId,
      .vulkanInstance = p_vkInstance,
  };
  VkPhysicalDevice physicalDevice;
  XRMG_ASSERT_XR(xrGetVulkanGraphicsDevice2KHR(m_instance, &getInfo, &physicalDevice));
  auto findIt = std::find(p_candidates, p_candidates + p_candidateCount, physicalDevice);
  XRMG_ASSERT(findIt != p_candidates + p_candidateCount, "No compatible main physical device found for OpenXR.");
  m_mainPhysicalDevice = *findIt;
  return static_cast<uint32_t>(findIt - p_candidates);
}

std::vector<const char *> XrUserInterface::getNeededDeviceExtensions() { return {"VK_KHR_external_memory_win32"}; }

void XrUserInterface::initialize(const Renderer &p_renderer, uint32_t p_presentQueueFamilyIndex,
                                 uint32_t p_presentQueueIndex) {
  std::fill(m_locatedViews.begin(), m_locatedViews.end(), XrView{.type = XR_TYPE_VIEW});

  XrGraphicsBindingVulkanKHR graphicsBindingVulkan = {.type = XR_TYPE_GRAPHICS_BINDING_VULKAN_KHR,
                                                      .instance = p_renderer.vkInstance(),
                                                      .physicalDevice = m_mainPhysicalDevice,
                                                      .device = p_renderer.vkDevice(),
                                                      .queueFamilyIndex = p_presentQueueFamilyIndex,
                                                      .queueIndex = p_presentQueueIndex};
  XrSessionCreateInfo sessionCreateInfo = {
      .type = XR_TYPE_SESSION_CREATE_INFO, .next = &graphicsBindingVulkan, .systemId = m_systemId};
  XRMG_ASSERT_XR(xrCreateSession(m_instance, &sessionCreateInfo, &m_session));

  do {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    this->handleEvents();
  } while (m_sessionState != XR_SESSION_STATE_READY);
  XrSessionBeginInfo sessionBeginInfo = {.type = XR_TYPE_SESSION_BEGIN_INFO,
                                         .primaryViewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO};
  XRMG_ASSERT_XR(xrBeginSession(m_session, &sessionBeginInfo));

  XrSwapchainCreateInfo swapchainCreateInfo = {
      .type = XR_TYPE_SWAPCHAIN_CREATE_INFO,
      .createFlags = 0,
      .usageFlags = XR_SWAPCHAIN_USAGE_TRANSFER_DST_BIT | XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT,
      .format = static_cast<uint64_t>(g_renderFormat),
      .sampleCount = 1,
      .width = 2 * m_resolutionPerEye.width,
      .height = m_resolutionPerEye.height,
      .faceCount = 1,
      .arraySize = 1,
      .mipCount = 1,
  };
  m_colorSwapchain = this->createSwapchain(swapchainCreateInfo);
  swapchainCreateInfo.usageFlags =
      XR_SWAPCHAIN_USAGE_TRANSFER_DST_BIT | XR_SWAPCHAIN_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
  swapchainCreateInfo.format = static_cast<uint64_t>(g_depthFormat);
  m_depthSwapchain = this->createSwapchain(swapchainCreateInfo);

  XrReferenceSpaceCreateInfo spaceCreateInfo = {.type = XR_TYPE_REFERENCE_SPACE_CREATE_INFO,
                                                .referenceSpaceType = XR_REFERENCE_SPACE_TYPE_STAGE,
                                                .poseInReferenceSpace = {.orientation = {.w = 1.0f}, .position = {}}};
  XRMG_ASSERT_XR(xrCreateReferenceSpace(m_session, &spaceCreateInfo, &m_space));
}

XrUserInterface::Swapchain XrUserInterface::createSwapchain(const XrSwapchainCreateInfo &p_createInfo) const {
  XrSwapchain swapchain;
  XRMG_ASSERT_XR(xrCreateSwapchain(m_session, &p_createInfo, &swapchain));
  uint32_t swapchainImageCount;
  XRMG_ASSERT_XR(xrEnumerateSwapchainImages(swapchain, 0, &swapchainImageCount, nullptr));
  std::vector<XrSwapchainImageVulkanKHR> images(swapchainImageCount, {.type = XR_TYPE_SWAPCHAIN_IMAGE_VULKAN_KHR});
  XRMG_ASSERT_XR(xrEnumerateSwapchainImages(swapchain, swapchainImageCount, &swapchainImageCount,
                                            (XrSwapchainImageBaseHeader *)images.data()));
  std::vector<vk::Image> vkImages(swapchainImageCount);
  for (uint32_t i = 0; i < swapchainImageCount; ++i) {
    vkImages[i] = images[i].image;
  }
  return {swapchain, vkImages};
}

XrUserInterface::FrameInfo XrUserInterface::beginFrame() {
  this->handleEvents();
  XRMG_ASSERT(m_sessionState != XR_SESSION_STATE_LOSS_PENDING && m_sessionState != XR_SESSION_STATE_EXITING &&
                  m_sessionState != XR_SESSION_STATE_STOPPING,
              "Unexpected session state: 0x{:x}", static_cast<int32_t>(m_sessionState));

  XrFrameWaitInfo frameWaitInfo = {.type = XR_TYPE_FRAME_WAIT_INFO};
  XrFrameState frameState = {.type = XR_TYPE_FRAME_STATE};
  {
    XRMG_SCOPED_INSTRUMENT("xrWaitFrame");
    XRMG_ASSERT_XR(xrWaitFrame(m_session, &frameWaitInfo, &frameState));
  }
  m_currentFramePredictedDisplayTime = frameState.predictedDisplayTime;
  XrViewLocateInfo viewLocateInfo = {.type = XR_TYPE_VIEW_LOCATE_INFO,
                                     .viewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO,
                                     .displayTime = frameState.predictedDisplayTime,
                                     .space = m_space};
  XrViewState viewState = {.type = XR_TYPE_VIEW_STATE};
  uint32_t viewCount;
  XRMG_ASSERT_XR(xrLocateViews(m_session, &viewLocateInfo, &viewState, 0, &viewCount, nullptr));
  XRMG_ASSERT(viewCount == VIEW_COUNT, "Expected view count: {}, actual: {}", VIEW_COUNT, viewCount);
  XRMG_ASSERT_XR(xrLocateViews(m_session, &viewLocateInfo, &viewState, viewCount, &viewCount, m_locatedViews.data()));
  {
    XRMG_SCOPED_INSTRUMENT("xrBeginFrame");
    XRMG_ASSERT_XR(xrBeginFrame(m_session, nullptr));
  }
  m_swapchainImageState = SwapchainImageState::UNTOUCHED;
  return frameState.shouldRender ? FrameInfo{.predictedDisplayTimeNanos = frameState.predictedDisplayTime}
                                 : FrameInfo{};
}

Mat4x4f XrUserInterface::getCurrentFrameView(StereoProjection::Eye p_eye) {
  const XrQuaternionf &q = m_locatedViews[p_eye == StereoProjection::Eye::LEFT ? 0 : 1].pose.orientation;
  const XrVector3f &p = m_locatedViews[p_eye == StereoProjection::Eye::LEFT ? 0 : 1].pose.position;
  return Mat4x4f::createRotation(-q.x, -q.y, -q.z, q.w) * Mat4x4f::createTranslation(-p.x, -p.y, -p.z);
}

StereoProjection XrUserInterface::getCurrentFrameProjection(StereoProjection::Eye p_eye) {
  const XrFovf &fov = m_locatedViews[p_eye == StereoProjection::Eye::LEFT ? 0 : 1].fov;
  Angle hFov = Angle::rad(fov.angleRight - fov.angleLeft);
  Angle vFov = Angle::rad(fov.angleUp - fov.angleDown);
  XRMG_WARN_IF(hFov.rad() < 0.0f || vFov.rad() < 0.0f, "Image flipping not yet supported.");
  return StereoProjection::create(Angle::rad(fov.angleLeft), Angle::rad(fov.angleRight), Angle::rad(fov.angleUp),
                                  Angle::rad(fov.angleDown), 1e-2f, 1e2f);
}

XrUserInterface::FrameRenderTargets XrUserInterface::acquireSwapchainImages(vk::Device p_device) {
  XrSwapchainImageAcquireInfo acquireInfo = {.type = XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
  uint32_t colorSwapchainImageIndex;
  uint32_t depthSwapchainImageIndex;
  XRMG_ASSERT_XR(xrAcquireSwapchainImage(m_colorSwapchain.swapchain, &acquireInfo, &colorSwapchainImageIndex));
  XRMG_ASSERT_XR(xrAcquireSwapchainImage(m_depthSwapchain.swapchain, &acquireInfo, &depthSwapchainImageIndex));
  XrSwapchainImageWaitInfo waitInfo = {.type = XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO, .timeout = XR_INFINITE_DURATION};
  XRMG_ASSERT_XR(xrWaitSwapchainImage(m_colorSwapchain.swapchain, &waitInfo));
  XRMG_ASSERT_XR(xrWaitSwapchainImage(m_depthSwapchain.swapchain, &waitInfo));
  m_swapchainImageState = SwapchainImageState::ACQUIRED;
  return {m_colorSwapchain.images[colorSwapchainImageIndex], vk::ImageLayout::eColorAttachmentOptimal,
          m_depthSwapchain.images[depthSwapchainImageIndex], vk::ImageLayout::eDepthStencilAttachmentOptimal};
}

void XrUserInterface::releaseSwapchainImage() {
  XrSwapchainImageReleaseInfo releaseInfo = {.type = XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
  XRMG_ASSERT_XR(xrReleaseSwapchainImage(m_colorSwapchain.swapchain, &releaseInfo));
  XRMG_ASSERT_XR(xrReleaseSwapchainImage(m_depthSwapchain.swapchain, &releaseInfo));
  m_swapchainImageState = SwapchainImageState::RELEASED;
}

void XrUserInterface::endFrame(vk::Queue p_presentGraphicsQueue) {
  XRMG_ERROR_IF(m_sessionState == XR_SESSION_STATE_LOSS_PENDING || m_sessionState == XR_SESSION_STATE_EXITING ||
                    m_sessionState == XR_SESSION_STATE_STOPPING,
                "Illegal call to endFrame.");

  std::vector<XrCompositionLayerProjectionView> projectionViews(VIEW_COUNT);
  std::vector<XrCompositionLayerDepthInfoKHR> depthViews(VIEW_COUNT);
  for (uint32_t i = 0; i < VIEW_COUNT; ++i) {
    XrRect2Di imageRect = {
        .offset = {static_cast<int32_t>((i % 2) * m_resolutionPerEye.width), 0},
        .extent = {static_cast<int32_t>(m_resolutionPerEye.width), static_cast<int32_t>(m_resolutionPerEye.height)}};
    depthViews[i] = {
        .type = XR_TYPE_COMPOSITION_LAYER_DEPTH_INFO_KHR,
        .subImage = {.swapchain = m_depthSwapchain.swapchain, .imageRect = imageRect, .imageArrayIndex = 0},
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
        .nearZ = 1e-2f,
        .farZ = 1e2f};
    projectionViews[i] = {
        .type = XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW,
        .next = &depthViews[i],
        .pose = m_locatedViews[i].pose,
        .fov = m_locatedViews[i].fov,
        .subImage = {.swapchain = m_colorSwapchain.swapchain, .imageRect = imageRect, .imageArrayIndex = 0}};
  }

  XrCompositionLayerProjection layer = {.type = XR_TYPE_COMPOSITION_LAYER_PROJECTION,
                                        .layerFlags = XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT,
                                        .space = m_space,
                                        .viewCount = static_cast<uint32_t>(projectionViews.size()),
                                        .views = projectionViews.data()};
  std::vector<XrCompositionLayerBaseHeader *> layers = {reinterpret_cast<XrCompositionLayerBaseHeader *>(&layer)};
  XrFrameEndInfo frameEndInfo = {.type = XR_TYPE_FRAME_END_INFO,
                                 .displayTime = m_currentFramePredictedDisplayTime,
                                 .environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE};
  if (m_swapchainImageState == SwapchainImageState::RELEASED) {
    frameEndInfo.layerCount = static_cast<uint32_t>(layers.size());
    frameEndInfo.layers = layers.data();
  }
  {
    XRMG_SCOPED_INSTRUMENT("xrEndFrame");
    XRMG_ASSERT_XR(xrEndFrame(m_session, &frameEndInfo));
  }
}

void XrUserInterface::handleEvents() {
  XrResult pollEventResult;
  do {
    XrEventDataBuffer evt = {.type = XR_TYPE_EVENT_DATA_BUFFER};
    XRMG_ASSERT_XR(pollEventResult = xrPollEvent(m_instance, &evt));
    if (pollEventResult == XR_SUCCESS) {
      char buffer[XR_MAX_STRUCTURE_NAME_SIZE] = {0};
      switch (evt.type) {
      case XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED:
        this->handle(reinterpret_cast<XrEventDataSessionStateChanged &>(evt));
        break;
      default:
        XRMG_ASSERT_XR(xrStructureTypeToString(m_instance, evt.type, buffer));
        XRMG_WARN("Unhandled XR event: {}", buffer);
        break;
      }
    }
  } while (pollEventResult != XR_EVENT_UNAVAILABLE);
}

void XrUserInterface::handle(XrEventDataSessionStateChanged &p_evt) {
  if (m_sessionState == p_evt.state) {
    return;
  }
  std::string state;
  switch (m_sessionState = p_evt.state) {
  case XR_SESSION_STATE_UNKNOWN: state = "XR_SESSION_STATE_UNKNOWN"; break;
  case XR_SESSION_STATE_IDLE: state = "XR_SESSION_STATE_IDLE"; break;
  case XR_SESSION_STATE_READY: state = "XR_SESSION_STATE_READY"; break;
  case XR_SESSION_STATE_SYNCHRONIZED: state = "XR_SESSION_STATE_SYNCHRONIZED"; break;
  case XR_SESSION_STATE_VISIBLE: state = "XR_SESSION_STATE_VISIBLE"; break;
  case XR_SESSION_STATE_FOCUSED: state = "XR_SESSION_STATE_FOCUSED"; break;
  case XR_SESSION_STATE_STOPPING: state = "XR_SESSION_STATE_STOPPING"; break;
  case XR_SESSION_STATE_LOSS_PENDING: state = "XR_SESSION_STATE_LOSS_PENDING"; break;
  case XR_SESSION_STATE_EXITING: state = "XR_SESSION_STATE_EXITING"; break;
  default: state = "<unknown state>"; break;
  }
  XRMG_INFO("New session state: {}", state);
  if (m_sessionState == XR_SESSION_STATE_STOPPING || m_sessionState == XR_SESSION_STATE_LOSS_PENDING) {
    XRMG_ASSERT_XR(xrEndSession(m_session));
    m_session = 0;
  }
}
} // namespace xrmg
