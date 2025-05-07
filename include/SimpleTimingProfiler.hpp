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
#include <xrmg.hpp>

#include <chrono>
#include <optional>
#include <string>
#include <vector>

namespace xrmg {
class SimpleTimingProfiler {
public:
  SimpleTimingProfiler(const std::string &p_name, size_t p_logInterval);

  const std::string &getName() const { return m_name; }
  void tic();
  void toc();

private:
  std::string m_name;
  size_t m_logInterval;
  std::optional<std::chrono::high_resolution_clock::time_point> m_prevTic;
  std::vector<std::chrono::nanoseconds> m_durations;

  void logDuration(std::chrono::nanoseconds p_duration);
};
} // namespace xrmg
