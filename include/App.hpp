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
#include "xrmg.hpp"

#include "Options.hpp"
#include "Renderer.hpp"
#include "Scene.hpp"
#include "VulkanAppProfiler.hpp"
#include "Window.hpp"

#include <optional>

#define XRMG_CONCAT_T(_a, _b) _a##_b
#define XRMG_CONCAT(_a, _b) XRMG_CONCAT_T(_a, _b)
#define XRMG_SCOPED_INSTRUMENT(_name)                                                                                  \
  VAP_SCOPED_INSTRUMENT(g_app->getProfiler(), _name, g_app->getCurrentFrameIndex());                                   \
  nvtx3::scoped_range XRMG_CONCAT(_nvtx3_scope, __LINE__)(_name)

namespace xrmg {
class App : public UserInputSink {
public:
  App(const std::vector<std::string> &p_args);
  ~App();

  const Options &getOptions() const { return m_options; }
  vap::VulkanAppProfiler &getProfiler() { return *m_profiler; }
  int32_t run();
  void discontinue() { m_discontinue = true; }
  void togglePaused() { m_paused = !m_paused; }
  bool isPaused() const { return m_paused; }
  Scene &getScene() const { return *m_scene; }
  uint64_t getCurrentFrameIndex() const { return m_renderer->getCurrentFrameIndex(); }

  bool onKeyDown(int32_t p_key) override;
  bool onKeyUp(int32_t p_key) override { return false; }
  bool onMouseMove(int32_t p_deltaX, int32_t p_deltaY) override { return false; }
  bool onWheelMove(int32_t p_delta) override { return false; }

private:
  Options m_options;
  std::unique_ptr<Window> m_window;
  std::unique_ptr<Renderer> m_renderer;
  std::unique_ptr<vap::VulkanAppProfiler> m_profiler;

  bool m_paused = false;
  bool m_discontinue = false;
  std::unique_ptr<Scene> m_scene;
  uint32_t m_baseTorusTesselationCount;
  uint32_t m_baseTorusCount;
  uint32_t m_torusLayerCount;

  void createProfiler();
  void checkProfiler();
};

class AppPtr {
  friend class App;

public:
  App *operator->() const { return m_instance; }
  App &operator*() const { return *m_instance; }

private:
  App *m_instance = nullptr;
};

extern AppPtr g_app;
} // namespace xrmg