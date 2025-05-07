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
#include <vulkan/vulkan.hpp>

#include <filesystem>
#include <functional>
#include <stdint.h>
#include <unordered_map>

#ifndef VAP_NAMESPACE
#define VAP_NAMESPACE vap
#endif

#define VAP_CONCAT_T(_a, _b) _a##_b
#define VAP_CONCAT(_a, _b) XRMG_CONCAT_T(_a, _b)
#define VAP_SCOPED_INSTRUMENT(_profiler, _name, _frameIndex)                                                           \
  vap::VulkanAppProfiler::Scope XRMG_CONCAT(_xrmg_scope_, __LINE__)(_profiler, _name, _frameIndex)

namespace VAP_NAMESPACE {
enum class LogLevel { Info, Warn, Error, Fatal };

class VulkanAppProfiler {
public:
  class Scope {
  public:
    Scope(VulkanAppProfiler &p_profiler, const std::string &p_name, uint64_t p_frameIndex);
    ~Scope();

    Scope(Scope &) = delete;
    Scope(Scope &&) = delete;
    Scope &operator=(Scope &) = delete;
    Scope &operator=(Scope &&) = delete;

  private:
    VulkanAppProfiler &m_profiler;
    bool m_enabled;
  };

  struct EventData {
    uint64_t beginNanos;
    std::optional<uint64_t> endNanos;
  };

  struct Event {
    std::string name;
    uint64_t frameIndex;
    EventData cpu;
    std::optional<EventData> gpu;
    std::optional<uint32_t> physicalDeviceIndex;
  };

  typedef std::vector<Event> Events;
  typedef std::function<void(const std::string &p_filename, int32_t p_line, LogLevel p_level, const std::string &p_msg)>
      LogFn;

  VulkanAppProfiler() = default;
  VulkanAppProfiler(const LogFn &p_logFn, vk::Instance p_vkInstance, vk::Device p_vkDevice,
                    vk::PhysicalDevice p_mainPhysicalDevice, uint32_t p_maxTimestamps);

  void calibrateWAR(vk::Device p_vkDevice, uint32_t p_graphicsQueueFamilyIndex, uint32_t p_physicalDeviceCount);
  void setEnabled(bool p_enabled) { m_enabled = p_enabled; }
  bool isEnabled() const { return m_enabled; }
  void resetQueryPool(vk::CommandBuffer cmdBuffer);
  const vk::QueryPool &getQueryPool() const { return m_queryPool.get(); }
  void pushInstant(const std::string &p_name, uint64_t p_frameIndex, uint32_t p_physicalDeviceIndex,
                   vk::CommandBuffer p_cmdBuffer,
                   vk::PipelineStageFlags2 p_stage = vk::PipelineStageFlagBits2::eAllCommands);
  void pushDurationBegin(const std::string &p_name, uint64_t p_frameIndex, uint32_t p_physicalDeviceIndex,
                         vk::CommandBuffer p_cmdBuffer,
                         vk::PipelineStageFlags2 p_stage = vk::PipelineStageFlagBits2::eAllCommands);
  void pushDurationEnd(vk::CommandBuffer p_cmdBuffer,
                       vk::PipelineStageFlags2 p_stage = vk::PipelineStageFlagBits2::eAllCommands);
  void pushCpuInstant(const std::string &p_name, uint64_t p_frameIndex);
  void pushCpuDurationBegin(const std::string &p_name, uint64_t p_frameIndex);
  void pushCpuDurationEnd();
  Events flush(vk::Device p_vkDevice);
  void writeTraceEventJson(const std::filesystem::path &p_filePath);

private:
  typedef uint32_t TimestampIndex;

  struct CpuEvent {
    std::chrono::high_resolution_clock::time_point primary;
    std::optional<std::chrono::high_resolution_clock::time_point> secondary;
  };

  struct GpuEvent {
    TimestampIndex primary;
    std::optional<TimestampIndex> secondary;
    uint32_t physicalDeviceIndex;
  };

  struct EventInProgress {
    std::string name;
    uint64_t frameIndex;
    std::optional<CpuEvent> cpu;
    std::optional<GpuEvent> gpu;
  };

  typedef std::pair<std::string, uint64_t> EventId;

  struct EventIdHash {
    size_t operator()(const EventId &p_evtId) const;
  };

  LogFn m_logFn;
  uint32_t m_maxTimestamps;
  vk::UniqueQueryPool m_queryPool;
  TimestampIndex m_nextTimestampIndex;
  uint64_t m_timestampPeriod;
  uint64_t m_cpuCalibratedTimestamp;
  uint64_t m_gpuCalibratedTimestamp;
  bool m_enabled;
  std::unordered_map<EventId, EventInProgress, EventIdHash> m_eventsInProgress;
  std::vector<EventId> m_openDurationEventStack;
  Events m_finishedEvents;
  std::vector<int64_t> m_physicalDeviceOffsetsWAR;

  TimestampIndex writeNext(vk::CommandBuffer p_cmdBuffer,
                           vk::PipelineStageFlags2 p_stage = vk::PipelineStageFlagBits2::eAllCommands);
  std::vector<uint64_t> getResults(vk::Device p_vkDevice) const;
};
} // namespace VAP_NAMESPACE
