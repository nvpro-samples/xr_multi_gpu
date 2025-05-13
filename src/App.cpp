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
#include "App.hpp"

#include "WindowUserInterface.hpp"
#include "XrUserInterface.hpp"

namespace xrmg {
AppPtr g_app;

App::App(const std::vector<std::string> &p_args)
    : m_options(p_args), m_baseTorusTesselationCount(m_options.initialBaseTorusTesselation),
      m_baseTorusCount(m_options.initialBaseTorusCount), m_torusLayerCount(m_options.initialTorusLayerCount) {
  XRMG_ASSERT(!g_app.m_instance, "Only a single application instance is allowed.");
  g_app.m_instance = this;
  std::unique_ptr<UserInterface> userInterface;
  if (m_options.monitorIndex) {
    m_window = std::make_unique<Window>(m_options.monitorIndex.value());
    userInterface = std::make_unique<WindowUserInterface>(*m_window);
  } else if (m_options.windowClientAreaSize) {
    m_window = std::make_unique<Window>(m_options.windowClientAreaSize.value());
    userInterface = std::make_unique<WindowUserInterface>(*m_window);
  } else {
    m_window = std::make_unique<Window>(vk::Extent2D(1280, 720));
    userInterface = std::make_unique<XrUserInterface>(m_options.oxrCoreValidation);
  }
  m_window->pushUserInputSink(*this);
  m_renderer = std::make_unique<Renderer>(std::move(userInterface));
  this->createProfiler();

  m_scene = std::make_unique<Scene>(*m_renderer);
  m_scene->buildCage(m_baseTorusTesselationCount, m_baseTorusCount, m_torusLayerCount);
}

App::~App() {
  m_renderer->waitIdle();
  m_window->removeUserInputSink(*this);
  m_window.reset();
}

int32_t App::run() {
  float ftSum = 0.0f;
  uint32_t ftCount = 0;
  auto frameBegin = std::chrono::high_resolution_clock::now();
  while (!m_discontinue) {
    XRMG_SCOPED_INSTRUMENT("main loop iteration");

    this->checkProfiler();
    m_window->processMessages();
    m_renderer->nextFrame(*m_scene);

    auto frameEnd = std::chrono::high_resolution_clock::now();
    ftSum += std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(frameEnd - frameBegin).count();
    ++ftCount;
    if (static_cast<float>(m_options.frameTimeLogInterval.value_or(1000)) < ftSum) {
      float avg = ftSum / static_cast<float>(ftCount);
      XRMG_INFO_IF(m_options.frameTimeLogInterval, "Avg. frame time: {:.2f} ms.", avg);
      m_window->setText(std::format("{} | {:.2f} ms", SAMPLE_NAME, avg));
      ftSum = 0.0f;
      ftCount = 0;
    }
    frameBegin = frameEnd;
  }
  return 0;
}

void App::createProfiler() {
  m_profiler = std::make_unique<vap::VulkanAppProfiler>(
      [](const std::string &p_filename, int32_t p_line, vap::LogLevel p_level, const std::string &p_msg) {
        xrmg::LogLevel lvl = LogLevel::Info;
        switch (p_level) {
        case vap::LogLevel::Warn: lvl = xrmg::LogLevel::Warn;
        case vap::LogLevel::Error: lvl = xrmg::LogLevel::Error;
        case vap::LogLevel::Fatal: lvl = xrmg::LogLevel::Fatal;
        }
        xrmg::log(p_filename, p_line, lvl, p_msg);
      },
      m_renderer->vkInstance(), m_renderer->vkDevice(), m_renderer->getPhysicalDevice(0), 1000);
  if (!g_app->getOptions().simulatedPhysicalDeviceCount) {
    m_profiler->calibrateWAR(m_renderer->vkDevice(), m_renderer->getGraphicsQueueFamilyIndex(),
                             m_renderer->getPhysicalDeviceCount());
  }
}

void App::checkProfiler() {
  if (m_options.traceRange) {
    if (m_renderer->getCurrentFrameIndex() == m_options.traceRange.value().first) {
      XRMG_INFO("Tracing started.");
      m_profiler->setEnabled(true);
    } else if (m_renderer->getCurrentFrameIndex() == m_options.traceRange.value().second) {
      m_profiler->setEnabled(false);
      m_profiler->flush(m_renderer->vkDevice());
      m_profiler->writeTraceEventJson(m_options.traceFilePath);
    }
  }
}

bool App::onKeyDown(int32_t p_key) {
  switch (p_key) {
  case VirtualKey::NUMPAD7: m_baseTorusTesselationCount = std::min(m_baseTorusTesselationCount * 2, 128u); break;
  case VirtualKey::NUMPAD4: m_baseTorusTesselationCount = m_options.initialBaseTorusTesselation; break;
  case VirtualKey::NUMPAD1: m_baseTorusTesselationCount = std::max(m_baseTorusTesselationCount / 2, 8u); break;
  case VirtualKey::NUMPAD8: m_baseTorusCount = std::min(m_baseTorusCount + 1, Scene::MAX_BASE_TORUS_COUNT); break;
  case VirtualKey::NUMPAD5: m_baseTorusCount = m_options.initialBaseTorusCount; break;
  case VirtualKey::NUMPAD2: m_baseTorusCount = std::max(m_baseTorusCount - 1, 2u); break;
  case VirtualKey::NUMPAD9: m_torusLayerCount = std::min(m_torusLayerCount + 1, Scene::MAX_TORUS_LAYER_COUNT); break;
  case VirtualKey::NUMPAD6: m_torusLayerCount = m_options.initialTorusLayerCount; break;
  case VirtualKey::NUMPAD3: m_torusLayerCount = std::max(m_torusLayerCount - 1, 1u); break;
  default: return false;
  }
  m_scene->buildCage(m_baseTorusTesselationCount, m_baseTorusCount, m_torusLayerCount);
  return true;
}
} // namespace xrmg