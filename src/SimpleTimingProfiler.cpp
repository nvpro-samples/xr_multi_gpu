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
#include "SimpleTimingProfiler.hpp"

namespace xrmg {
SimpleTimingProfiler::SimpleTimingProfiler(const std::string &p_name, size_t p_logInterval)
    : m_name(p_name), m_logInterval(p_logInterval) {}

void SimpleTimingProfiler::tic() {
  if (m_logInterval != 0) {
    auto now = std::chrono::high_resolution_clock::now();
    if (m_prevTic) {
      this->logDuration(now - m_prevTic.value());
    }
    m_prevTic = now;
  }
}

void SimpleTimingProfiler::toc() {
  if (m_logInterval != 0) {
    if (m_prevTic) {
      this->logDuration(std::chrono::high_resolution_clock::now() - m_prevTic.value());
      m_prevTic.reset();
    } else {
      XRMG_WARN("toc() called without preceding call to tic()");
    }
  }
}

void SimpleTimingProfiler::logDuration(std::chrono::nanoseconds p_duration) {
  m_durations.emplace_back(p_duration);
  if (m_durations.size() == m_logInterval) {
    std::chrono::nanoseconds sum(0);
    for (auto d : m_durations) {
      sum += d;
    }
    XRMG_INFO("avg {}: {:.2f} ms.", m_name,
              std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(sum).count() /
                  static_cast<float>(m_logInterval));
    m_durations.clear();
  }
}
} // namespace xrmg