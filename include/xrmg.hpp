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
#define _USE_MATH_DEFINES

#ifdef _WIN32
#include <stdlib.h>
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#include <nvtx3/nvtx3.hpp>
#include <vulkan/vulkan.hpp>

#define XR_USE_GRAPHICS_API_VULKAN
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>
#include <openxr/openxr_reflection.h>

#include <chrono>
#include <format>
#include <fstream>
#include <iostream>
#include <regex>
#include <string>

#ifndef SAMPLE_NAME
#define SAMPLE_NAME ""
#endif

#define XRMG_LOG_IF(_cond, _level, _format, ...)                                                                       \
  do {                                                                                                                 \
    if (_cond) {                                                                                                       \
      xrmg::log(__FILE__, __LINE__, _level, std::format(_format, __VA_ARGS__));                                        \
    }                                                                                                                  \
  } while (0)
#define XRMG_LOG_UNLESS(_cond, _level, _format, ...) XRMG_LOG_IF(!(_cond), _level, _format, __VA_ARGS__)
#define XRMG_LOG(_level, _format, ...) XRMG_LOG_IF(true, _level, _format, __VA_ARGS__)

#define XRMG_INFO_IF(_cond, _format, ...) XRMG_LOG_IF(_cond, xrmg::LogLevel::Info, _format, __VA_ARGS__)
#define XRMG_INFO_UNLESS(_cond, _format, ...) XRMG_LOG_UNLESS(_cond, LogLevel::Info, _format, __VA_ARGS__)
#define XRMG_INFO(_format, ...) XRMG_LOG(xrmg::LogLevel::Info, _format, __VA_ARGS__)
#define XRMG_INFO_ONCE(_format, ...)                                                                                   \
  do {                                                                                                                 \
    static bool _shown = false;                                                                                        \
    XRMG_INFO_UNLESS(_shown, _format, __VA_ARGS__);                                                                    \
    _shown = true;                                                                                                     \
  } while (0)

#define XRMG_WARN_IF(_cond, _format, ...) XRMG_LOG_IF(_cond, xrmg::LogLevel::Warn, _format, __VA_ARGS__)
#define XRMG_WARN_UNLESS(_cond, _format, ...) XRMG_LOG_UNLESS(_cond, xrmg::LogLevel::Warn, _format, __VA_ARGS__)
#define XRMG_WARN(_format, ...) XRMG_LOG(xrmg::LogLevel::Warn, _format, __VA_ARGS__)

#define XRMG_ERROR_IF(_cond, _format, ...) XRMG_LOG_IF(_cond, xrmg::LogLevel::Error, _format, __VA_ARGS__)
#define XRMG_ERROR_UNLESS(_cond, _format, ...) XRMG_LOG_UNLESS(_cond, xrmg::LogLevel::Error, _format, __VA_ARGS__)
#define XRMG_ERROR(_format, ...) XRMG_LOG(xrmg::LogLevel::Error, _format, __VA_ARGS__)

#define XRMG_FATAL_IF(_cond, _format, ...)                                                                             \
  do {                                                                                                                 \
    if (_cond) {                                                                                                       \
      XRMG_LOG(xrmg::LogLevel::Fatal, _format, __VA_ARGS__);                                                           \
      xrmg::breakableExit(1);                                                                                          \
    }                                                                                                                  \
  } while (0)
#define XRMG_FATAL_UNLESS(_cond, _format, ...) XRMG_FATAL_IF(!(_cond), _format, __VA_ARGS__)
#define XRMG_FATAL(_format, ...) XRMG_FATAL_IF(true, _format, __VA_ARGS__)
#define XRMG_ASSERT(_cond, _format, ...) XRMG_FATAL_UNLESS(_cond, _format, __VA_ARGS__)

#define XRMG_ASSERT_VK(_cmd)                                                                                           \
  do {                                                                                                                 \
    VkResult _res = _cmd;                                                                                              \
    XRMG_ASSERT(_res == VkResult::VK_SUCCESS, "[{}] caused by " #_cmd, vk::to_string(static_cast<vk::Result>(_res)));  \
  } while (0)

#define XRMG_ASSERT_XR(_cmd)                                                                                           \
  do {                                                                                                                 \
    XrResult _res = (_cmd);                                                                                            \
    if (!XR_SUCCEEDED(_res)) {                                                                                         \
      char buffer[XR_MAX_RESULT_STRING_SIZE] = {0};                                                                    \
      XRMG_FATAL("[{}] caused by " #_cmd,                                                                              \
                 SUCCEEDED(xrResultToString(m_instance, _res, buffer)) ? buffer : "Unknown OpenXR error");             \
    }                                                                                                                  \
  } while (0)

#define XRMG_BOOL_TO_STRING(_v) ((_v) ? "\033[32m✔\033[0m" : "\033[31m✘\033[0m")

#ifdef _WIN32
namespace xrmg {
inline std::string formatLastWin32Error(std::optional<DWORD> p_lastError = {}) {
  DWORD lastError = p_lastError.value_or(GetLastError());
  char msg[256];
  DWORD count = FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM, NULL, lastError, LANG_USER_DEFAULT, msg, sizeof(msg), NULL);
  return count == 0 ? std::format("FormatMessageA failed: 0x{:x}", lastError) : std::string(msg, msg + count);
}
} // namespace xrmg

#define _XRMG_WIN32_CHECK_LAST_ERROR_IF_(_cond, _level, _forceLog)                                                     \
  do {                                                                                                                 \
    if (_cond) {                                                                                                       \
      DWORD _lastError = GetLastError();                                                                               \
      if (_forceLog || !SUCCEEDED(_lastError)) {                                                                       \
        XRMG_FATAL_IF(_level == xrmg::LogLevel::Fatal, "{}", xrmg::formatLastWin32Error(_lastError));                  \
        XRMG_ERROR_IF(_level == xrmg::LogLevel::Error, "{}", xrmg::formatLastWin32Error(_lastError));                  \
        XRMG_WARN_IF(_level == xrmg::LogLevel::Warn, "{}", xrmg::formatLastWin32Error(_lastError));                    \
        XRMG_INFO_IF(_level == xrmg::LogLevel::Info, "{}", xrmg::formatLastWin32Error(_lastError));                    \
      }                                                                                                                \
    }                                                                                                                  \
  } while (0)

#define XRMG_WIN32_CHECK_LAST_ERROR_IF(_cond, _level) _XRMG_WIN32_CHECK_LAST_ERROR_IF_(_cond, _level, false)
#define XRMG_WIN32_CHECK_LAST_ERROR_UNLESS(_cond, _level) _XRMG_WIN32_CHECK_LAST_ERROR_IF_(!(_cond), _level, false)
#define XRMG_WIN32_CHECK_LAST_ERROR(_level) _XRMG_WIN32_CHECK_LAST_ERROR_IF_(true, _level, false)

#define XRMG_WIN32_WARN_UNLESS(_cond) _XRMG_WIN32_CHECK_LAST_ERROR_IF_(!(_cond), xrmg::LogLevel::Warn, true)
#define XRMG_WIN32_ERROR_UNLESS(_cond) _XRMG_WIN32_CHECK_LAST_ERROR_IF_(!(_cond), xrmg::LogLevel::Error, true)
#define XRMG_WIN32_ASSERT(_cond) _XRMG_WIN32_CHECK_LAST_ERROR_IF_(!(_cond), xrmg::LogLevel::Fatal, true)
#endif

namespace xrmg {
inline const uint32_t MAX_QUEUED_FRAMES = 3;
inline const vk::Format g_renderFormat = vk::Format::eR8G8B8A8Srgb;
inline const vk::Format g_depthFormat = vk::Format::eD32Sfloat;
inline const vk::ClearValue g_clearValues(vk::ClearColorValue(0.529f, 0.807f, 0.921f, 0.0f));

class App;

inline std::string formatByteSize(size_t p_byteSize) {
  if (p_byteSize < 1ull << 10) {
    return std::format("{} B", p_byteSize);
  } else if (p_byteSize < 1ull << 20) {
    return std::format("{:.3f} KiB", static_cast<float>(p_byteSize) / static_cast<float>(1ull << 10));
  } else if (p_byteSize < 1ull << 30) {
    return std::format("{:.3f} MiB", static_cast<float>(p_byteSize) / static_cast<float>(1ull << 20));
  } else if (p_byteSize < 1ull << 40) {
    return std::format("{:.3f} GiB", static_cast<float>(p_byteSize) / static_cast<float>(1ull << 30));
  } else {
    return std::format("{:.3f} TiB", static_cast<float>(p_byteSize) / static_cast<float>(1ull << 40));
  }
}

inline std::ofstream g_logFile(SAMPLE_NAME ".log", std::ios::binary);

enum class LogLevel { Info, Warn, Error, Fatal };

inline void log(const std::string &p_file, int32_t p_line, LogLevel p_logLevel, const std::string &p_message) {
  size_t length = p_message.length();
  while (length != 0 && (p_message[length - 1] == '\n' || p_message[length - 1] == '\r')) {
    --length;
  }
  std::string msg = p_message.substr(0, length);
  switch (p_logLevel) {
  case LogLevel::Info: std::cout << msg << std::endl; break;
  case LogLevel::Warn:
    std::cout << std::format("{}({}): \033[33m[WARN] {}\033[0m", p_file, p_line, msg) << std::endl;
    break;
  case LogLevel::Error:
  case LogLevel::Fatal:
    std::cerr << std::format("{}({}): \033[31m[ERROR] {}\033[0m", p_file, p_line, msg) << std::endl;
    break;
  }
  g_logFile << std::regex_replace(msg, std::regex("\x1b\\[\\d+m"), "") << std::endl;
}

inline void breakableExit(int32_t p_exitCode) {
#ifdef _WIN32
  _CrtDbgBreak();
#endif
  exit(p_exitCode);
}
} // namespace xrmg
