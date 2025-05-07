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
#include "Window.hpp"

#include "App.hpp"
#include "Scene.hpp"

#ifdef _WIN32
#include <hidusage.h>
#include <windowsx.h>
#endif

namespace xrmg {
#ifdef _WIN32
LRESULT Window::wndProcRelay(HWND p_hwnd, UINT p_msg, WPARAM p_wParam, LPARAM p_lParam) {
  auto window = reinterpret_cast<Window *>(GetWindowLongPtr(p_hwnd, 0));
  XRMG_WIN32_CHECK_LAST_ERROR_UNLESS(window, xrmg::LogLevel::Fatal);
  std::optional<LRESULT> result = window ? window->wndProc(p_hwnd, p_msg, p_wParam, p_lParam) : std::nullopt;
  return result ? result.value() : DefWindowProc(p_hwnd, p_msg, p_wParam, p_lParam);
}

static const DWORD g_windowedStyle = WS_OVERLAPPED | WS_SYSMENU | WS_CAPTION;
static const DWORD g_fullscreenStyle = WS_POPUP;

Window::Window(const vk::Extent2D &p_clientAreaSize) {
  RECT rect = {.right = static_cast<LONG>(p_clientAreaSize.width),
               .bottom = static_cast<LONG>(p_clientAreaSize.height)};
  AdjustWindowRect(&rect, g_windowedStyle, false);
  this->createAndOpenWin32Window(g_windowedStyle, CW_USEDEFAULT, CW_USEDEFAULT, rect.right - rect.left,
                                 rect.bottom - rect.top);
}

Window::Window(uint32_t p_monitorIndex) {
  std::vector<HMONITOR> monitors;
  BOOL r = EnumDisplayMonitors(
      NULL, NULL,
      [](HMONITOR p_monitor, HDC p_hdc, LPRECT p_rect, LPARAM p_lParam) {
        reinterpret_cast<std::vector<HMONITOR> *>(p_lParam)->emplace_back(p_monitor);
        return TRUE;
      },
      reinterpret_cast<LPARAM>(&monitors));
  XRMG_WIN32_ASSERT(r);
  XRMG_INFO("Monitors:{}", monitors.empty() ? " none" : "");
  for (uint32_t i = 0; i < monitors.size(); ++i) {
    MONITORINFO mi = {.cbSize = sizeof(MONITORINFO)};
    XRMG_WIN32_ASSERT(GetMonitorInfo(monitors[i], &mi) == TRUE);
    XRMG_INFO("{}[{}] LT({},{}), RB({},{})  [{} x {}]", p_monitorIndex == i ? ">" : " ", i, mi.rcMonitor.left,
              mi.rcMonitor.top, mi.rcMonitor.right, mi.rcMonitor.bottom, mi.rcMonitor.right - mi.rcMonitor.left,
              mi.rcMonitor.bottom - mi.rcMonitor.top);
  }

  XRMG_ASSERT(p_monitorIndex < monitors.size(), "Monitor index ({}) must be less than monitor count ({}).",
              p_monitorIndex, monitors.size());
  MONITORINFO mi = {.cbSize = sizeof(MONITORINFO)};
  XRMG_WIN32_ASSERT(GetMonitorInfo(monitors[p_monitorIndex], &mi));
  this->createAndOpenWin32Window(g_fullscreenStyle, mi.rcMonitor.left, mi.rcMonitor.top,
                                 mi.rcMonitor.right - mi.rcMonitor.left, mi.rcMonitor.bottom - mi.rcMonitor.top);
}

void Window::createAndOpenWin32Window(DWORD p_initialStyle, int32_t p_x, int32_t p_y, int32_t p_width,
                                      int32_t p_height) {
  WNDCLASSA wndClass = {.style = CS_DBLCLKS,
                        .lpfnWndProc = &Window::wndProcRelay,
                        .cbWndExtra = sizeof(Window *),
                        .hInstance = GetModuleHandle(NULL),
                        .lpszClassName = SAMPLE_NAME "-window-class"};
  XRMG_WIN32_ASSERT(RegisterClassA(&wndClass) != 0);
  m_hwnd = CreateWindowA(wndClass.lpszClassName, SAMPLE_NAME, p_initialStyle, p_x, p_y, p_width, p_height, NULL, NULL,
                         GetModuleHandle(NULL), NULL);
  XRMG_WIN32_ASSERT(m_hwnd);
  SetLastError(0);
  LONG_PTR r = SetWindowLongPtr(m_hwnd, 0, reinterpret_cast<LONG_PTR>(this));
  XRMG_WIN32_CHECK_LAST_ERROR_UNLESS(r, xrmg::LogLevel::Fatal);
  SetCapture(m_hwnd);
  ShowWindow(m_hwnd, SW_SHOW);
  RECT clientRect;
  GetClientRect(m_hwnd, &clientRect);
  m_swapchainImageSize.setWidth(static_cast<uint32_t>(clientRect.right - clientRect.left));
  m_swapchainImageSize.setHeight(static_cast<uint32_t>(clientRect.bottom - clientRect.top));
}

void Window::setRawInput(bool p_enabled) {
  if (m_rawInput != p_enabled) {
    RAWINPUTDEVICE rid = {
        .usUsagePage = HID_USAGE_PAGE_GENERIC,
        .usUsage = HID_USAGE_GENERIC_MOUSE,
        .dwFlags = static_cast<DWORD>(p_enabled ? RIDEV_NOLEGACY | RIDEV_INPUTSINK | RIDEV_CAPTUREMOUSE : RIDEV_REMOVE),
        .hwndTarget = p_enabled ? m_hwnd : NULL,
    };
    XRMG_WIN32_ASSERT(RegisterRawInputDevices(&rid, 1, sizeof(RAWINPUTDEVICE)));
    XRMG_WIN32_WARN_UNLESS(!p_enabled || GetCursorPos(&m_cursorPos));
    ShowCursor(p_enabled ? FALSE : TRUE);
    m_rawInput = p_enabled;
  }
}

void Window::createSurface(vk::Instance p_vkInstance) {
  vk::Win32SurfaceCreateInfoKHR createInfo({}, GetModuleHandleA(NULL), m_hwnd);
  m_surface = p_vkInstance.createWin32SurfaceKHRUnique(createInfo);
}
#endif

void Window::createSwapchain(vk::Device p_vkDevice, vk::Format p_swapchainFormat, uint32_t p_swapchainImageCount,
                             vk::PresentModeKHR p_presentMode) {
  vk::SwapchainCreateInfoKHR swapchainCreateInfo(
      {}, m_surface.get(), p_swapchainImageCount, p_swapchainFormat, vk::ColorSpaceKHR::eSrgbNonlinear,
      m_swapchainImageSize, 1, vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst,
      vk::SharingMode::eExclusive, {});
  swapchainCreateInfo.setPresentMode(p_presentMode);
  m_swapchain = p_vkDevice.createSwapchainKHRUnique(swapchainCreateInfo);
  m_swapchainImages = p_vkDevice.getSwapchainImagesKHR(m_swapchain.get());
}

uint32_t Window::acquireNextImageIndex(vk::Device p_vkDevice, vk::Semaphore p_signalSemaphore) {
  auto [result, index] = p_vkDevice.acquireNextImage2KHR({m_swapchain.get(), UINT64_MAX, p_signalSemaphore, {}, 0b1});
  XRMG_ASSERT(result == vk::Result::eSuccess, "acquireNextImage2KHR failed.");
  return index;
}

void Window::present(vk::Queue p_queue, uint32_t p_swapchainImageIndex, vk::Semaphore p_waitSemahore) const {
  vk::Result result;
  vk::PresentInfoKHR presentInfo(p_waitSemahore, m_swapchain.get(), p_swapchainImageIndex, result);
  XRMG_ASSERT(p_queue.presentKHR(presentInfo) == vk::Result::eSuccess, "presentKHR failed.");
  XRMG_WARN_UNLESS(result == vk::Result::eSuccess, "presentKHR returned {}.", vk::to_string(result));
}

void Window::setText(const std::string &p_text) const {
#ifdef _WIN32
  XRMG_WIN32_CHECK_LAST_ERROR_UNLESS(SetWindowText(m_hwnd, p_text.c_str()), LogLevel::Warn);
#else
#error todo
#endif
}

void Window::removeUserInputSink(UserInputSink &p_sink) {
  if (auto it = std::find(m_userInputSinks.begin(), m_userInputSinks.end(), &p_sink); it != m_userInputSinks.end()) {
    m_userInputSinks.erase(it);
  }
}

bool Window::onKeyDown(int32_t p_key) {
  for (auto it = m_userInputSinks.rbegin(); it != m_userInputSinks.rend(); ++it) {
    if ((*it)->onKeyDown(p_key)) {
      return true;
    }
  }
  return false;
}

bool Window::onKeyUp(int32_t p_key) {
  for (auto it = m_userInputSinks.rbegin(); it != m_userInputSinks.rend(); ++it) {
    if ((*it)->onKeyUp(p_key)) {
      return true;
    }
  }
  return false;
}

bool Window::onMouseMove(int32_t p_deltaX, int32_t p_deltaY) {
  for (auto it = m_userInputSinks.rbegin(); it != m_userInputSinks.rend(); ++it) {
    if ((*it)->onMouseMove(p_deltaX, p_deltaY)) {
      return true;
    }
  }
  return false;
}

bool Window::onWheelMove(int32_t p_delta) {
  for (auto it = m_userInputSinks.rbegin(); it != m_userInputSinks.rend(); ++it) {
    if ((*it)->onWheelMove(p_delta)) {
      return true;
    }
  }
  return false;
}

void Window::processMessages() const {
#ifdef _WIN32
  MSG msg;
  while (PeekMessageA(&msg, m_hwnd, 0, 0, PM_REMOVE)) {
    TranslateMessage(&msg);
    DispatchMessageA(&msg);
  }
#else
#warning TODO
#endif
}

std::optional<LRESULT> Window::wndProc(HWND p_hwnd, UINT p_msg, WPARAM p_wParam, LPARAM p_lParam) {
  if (p_msg == WM_DESTROY) {
    g_app->discontinue();
    PostQuitMessage(0);
    return 0;
  } else if ((p_msg == WM_KEYDOWN && this->onKeyDown(p_wParam)) || (p_msg == WM_KEYUP && this->onKeyUp(p_wParam))) {
    return 0;
  } else if (p_msg == WM_LBUTTONDOWN) {
    this->setRawInput(true);
    return 0;
  } else if (p_msg == WM_KILLFOCUS) {
    this->setRawInput(false);
    return 0;
  } else if (p_msg == WM_MOUSEWHEEL && this->onWheelMove(GET_WHEEL_DELTA_WPARAM(p_wParam))) {
    return 0;
  } else if (p_msg == WM_INPUT) {
    RAWINPUT rawInput;
    UINT cbSize = static_cast<UINT>(sizeof(RAWINPUT));
    XRMG_WIN32_ASSERT(GetRawInputData(reinterpret_cast<HRAWINPUT>(p_lParam), RID_INPUT,
                                      reinterpret_cast<void *>(&rawInput), &cbSize,
                                      sizeof(RAWINPUTHEADER)) != static_cast<UINT>(-1));
    if (rawInput.header.dwType == RIM_TYPEMOUSE) {
      XRMG_WIN32_WARN_UNLESS(SetCursorPos(m_cursorPos.x, m_cursorPos.y));
      this->onMouseMove(static_cast<int32_t>(rawInput.data.mouse.lLastX),
                        static_cast<int32_t>(rawInput.data.mouse.lLastY));
      if (rawInput.data.mouse.usButtonFlags & RI_MOUSE_BUTTON_1_UP) {
        this->setRawInput(false);
      }
      if ((rawInput.data.mouse.usButtonFlags & RI_MOUSE_WHEEL) && rawInput.data.mouse.usButtonData) {
        auto delta = *reinterpret_cast<int16_t *>(&rawInput.data.mouse.usButtonData);
        this->onWheelMove(delta);
      }
      return 0;
    } else {
      XRMG_WARN("Got unexptected non-mouse input.");
    }
  }
  return std::nullopt;
}
} // namespace xrmg
