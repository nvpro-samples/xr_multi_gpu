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
#include <stdint.h>

#ifdef _WIN32

#else
#error todo
#endif

namespace xrmg {
class UserInputSink {
public:
  virtual bool onKeyDown(int32_t p_key) = 0;
  virtual bool onKeyUp(int32_t p_key) = 0;
  virtual bool onMouseMove(int32_t p_deltaX, int32_t p_deltaY) = 0;
  virtual bool onWheelMove(int32_t p_delta) = 0;
};

class VirtualKey {
public:
#ifdef _WIN32
  static const int32_t PAUSE = VK_PAUSE;
  static const int32_t ESCAPE = VK_ESCAPE;
  static const int32_t SPACE = VK_SPACE;
  static const int32_t SHIFT = VK_SHIFT;
  static const int32_t ADD = VK_ADD;
  static const int32_t SUBTRACT = VK_SUBTRACT;
  static const int32_t NUMPAD0 = VK_NUMPAD0;
  static const int32_t NUMPAD1 = VK_NUMPAD1;
  static const int32_t NUMPAD2 = VK_NUMPAD2;
  static const int32_t NUMPAD3 = VK_NUMPAD3;
  static const int32_t NUMPAD4 = VK_NUMPAD4;
  static const int32_t NUMPAD5 = VK_NUMPAD5;
  static const int32_t NUMPAD6 = VK_NUMPAD6;
  static const int32_t NUMPAD7 = VK_NUMPAD7;
  static const int32_t NUMPAD8 = VK_NUMPAD8;
  static const int32_t NUMPAD9 = VK_NUMPAD9;
#else
#error todo
#endif
};
} // namespace xrmg
