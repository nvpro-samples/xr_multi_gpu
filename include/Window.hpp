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

#include "Options.hpp"
#include "UserInputSink.hpp"

#include <optional>

namespace xrmg {
class Window : public UserInputSink {
public:
  explicit Window(const vk::Extent2D &p_clientAreaSize);
  explicit Window(uint32_t p_monitorIndex);

  void createSurface(vk::Instance p_vkInstance);
  const vk::SurfaceKHR &getVulkanSurface() const { return m_surface.get(); }
  void createSwapchain(vk::Device p_vkDevice, vk::Format p_swapchainFormat, uint32_t p_swapchainImageCount,
                       vk::PresentModeKHR p_presentMode);
  const vk::SwapchainKHR &getSwapchain() const { return m_swapchain.get(); }
  const vk::Extent2D &getSwapchainImageSize() const { return m_swapchainImageSize; }
  uint32_t getSwapchainImageCount() const { return static_cast<uint32_t>(m_swapchainImages.size()); }
  const vk::Image &getSwapchainImage(uint32_t p_index) const { return m_swapchainImages[p_index]; }
  uint32_t acquireNextImageIndex(vk::Device p_vkDevice, vk::Semaphore p_signalSemaphore);
  void present(vk::Queue p_queue, uint32_t p_swapchainImageIndex, vk::Semaphore p_waitSemahore) const;
  void processMessages() const;
  void setText(const std::string &p_text) const;
  void pushUserInputSink(UserInputSink &p_sink) { m_userInputSinks.emplace_back(&p_sink); }
  void removeUserInputSink(UserInputSink &p_sink);
  bool onKeyDown(int32_t p_key) override;
  bool onKeyUp(int32_t p_key) override;
  bool onMouseMove(int32_t p_deltaX, int32_t p_deltaY) override;
  bool onWheelMove(int32_t p_delta) override;

private:
#ifdef _WIN32
  static LRESULT wndProcRelay(HWND p_hwnd, UINT p_msg, WPARAM p_wParam, LPARAM p_lParam);
#endif

#ifdef _WIN32
  HWND m_hwnd;
  POINT m_cursorPos;
  bool m_rawInput = false;
#endif
  vk::UniqueSurfaceKHR m_surface;
  vk::UniqueSwapchainKHR m_swapchain;
  vk::Extent2D m_swapchainImageSize;
  std::vector<vk::Image> m_swapchainImages;
  std::vector<UserInputSink *> m_userInputSinks;

#ifdef _WIN32
  void createAndOpenWin32Window(DWORD p_initialStyle, int32_t p_x, int32_t p_y, int32_t p_width, int32_t p_height);
  void setRawInput(bool p_enabled);
  std::optional<LRESULT> wndProc(HWND p_hwnd, UINT p_msg, WPARAM p_wParam, LPARAM p_lParam);
#endif
};
} // namespace xrmg
