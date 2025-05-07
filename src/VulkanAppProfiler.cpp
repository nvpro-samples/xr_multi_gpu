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
#include "VulkanAppProfiler.hpp"

#ifdef _WIN32
#define NOMINMAX
#include <Windows.h>
#endif

#include <format>
#include <fstream>
#include <iostream>

#define VAP_LOG(_level, _format, ...) m_logFn(__FILE__, __LINE__, _level, std::format(_format, __VA_ARGS__))

#define VAP_INFO_IF(_cond, _format, ...)                                                                               \
  do {                                                                                                                 \
    if (_cond) {                                                                                                       \
      VAP_LOG(VAP_NAMESPACE::LogLevel::Info, _format, __VA_ARGS__);                                                    \
    }                                                                                                                  \
  } while (0)
#define VAP_INFO_UNLESS(_cond, _format, ...) VAP_INFO_IF(!(_cond), _format, __VA_ARGS__)
#define VAP_INFO(_format, ...) VAP_INFO_IF(true, _format, __VA_ARGS__)

#define VAP_WARN_IF(_cond, _format, ...)                                                                               \
  do {                                                                                                                 \
    if (_cond) {                                                                                                       \
      VAP_LOG(VAP_NAMESPACE::LogLevel::Warn, _format, __VA_ARGS__);                                                    \
    }                                                                                                                  \
  } while (0)
#define VAP_WARN_UNLESS(_cond, _format, ...) VAP_WARN_IF(!(_cond), _format, __VA_ARGS__)
#define VAP_WARN(_format, ...) VAP_WARN_IF(true, _format, __VA_ARGS__)

#define VAP_ERROR(_format, ...) VAP_LOG(VAP_NAMESPACE::LogLevel::Error, _format, __VA_ARGS__)
#define VAP_ASSERT(_cond, _format, ...)                                                                                \
  do {                                                                                                                 \
    if (!(_cond)) {                                                                                                    \
      VAP_LOG(VAP_NAMESPACE::LogLevel::Fatal, _format, __VA_ARGS__);                                                   \
      assert(_cond);                                                                                                   \
      exit(1);                                                                                                         \
    }                                                                                                                  \
  } while (0)

#ifdef _WIN32
#define VAP_ASSERT_WIN32(_cond)                                                                                        \
  do {                                                                                                                 \
    if (!(_cond)) {                                                                                                    \
      char msg[256];                                                                                                   \
      if (FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM, nullptr, GetLastError(), LANG_USER_DEFAULT, msg, sizeof(msg),     \
                         nullptr) == 0) {                                                                              \
        VAP_ERROR("FormatMessageA failed: 0x{:x}", GetLastError());                                                    \
      } else {                                                                                                         \
        VAP_ERROR("{}", msg);                                                                                          \
      }                                                                                                                \
    }                                                                                                                  \
  } while (0)
#endif

PFN_vkGetCalibratedTimestampsKHR g_vkGetCalibratedTimestampsKHR;

VkResult vkGetCalibratedTimestampsKHR(VkDevice device, uint32_t timestampCount,
                                      const VkCalibratedTimestampInfoKHR *pTimestampInfos, uint64_t *pTimestamps,
                                      uint64_t *pMaxDeviation) {
  return g_vkGetCalibratedTimestampsKHR(device, timestampCount, pTimestampInfos, pTimestamps, pMaxDeviation);
}

namespace VAP_NAMESPACE {
VulkanAppProfiler::VulkanAppProfiler(const LogFn &p_logFn, vk::Instance p_vkInstance, vk::Device p_vkDevice,
                                     vk::PhysicalDevice p_mainPhysicalDevice, uint32_t p_maxTimestamps)
    : m_logFn(p_logFn), m_maxTimestamps(std::max(16u, p_maxTimestamps)),
      m_queryPool(p_vkDevice.createQueryPoolUnique({{}, vk::QueryType::eTimestamp, m_maxTimestamps})),
      m_nextTimestampIndex(m_maxTimestamps - 1),
      m_timestampPeriod(p_mainPhysicalDevice.getProperties().limits.timestampPeriod), m_enabled(false) {
  if (!g_vkGetCalibratedTimestampsKHR) {
    g_vkGetCalibratedTimestampsKHR =
        reinterpret_cast<PFN_vkGetCalibratedTimestampsKHR>(p_vkInstance.getProcAddr("vkGetCalibratedTimestampsKHR"));
  }
  auto [timestamps, maxDeviation] = p_vkDevice.getCalibratedTimestampsKHR({
      {vk::TimeDomainKHR::eDevice},
#ifdef _WIN32
      {vk::TimeDomainKHR::eQueryPerformanceCounter},
#elif defined(__unix__)
      {vk::TimeDomainKHR::eClockMonotonicRaw},
#endif
  });
  m_gpuCalibratedTimestamp = timestamps[0];

#ifdef _WIN32
  LARGE_INTEGER freq;
  VAP_ASSERT_WIN32(QueryPerformanceFrequency(&freq));
  if (1000000000ull % freq.QuadPart == 0) {
    m_cpuCalibratedTimestamp = timestamps[1] * (1000000000ull / freq.QuadPart);
  } else {
    m_cpuCalibratedTimestamp =
        static_cast<uint64_t>(static_cast<double>(timestamps[1]) * (1e9 / static_cast<double>(freq.QuadPart)));
  }
#elif defined(__unix__)
#error TODO
#endif
}

void VulkanAppProfiler::calibrateWAR(vk::Device p_vkDevice, uint32_t p_graphicsQueueFamilyIndex,
                                     uint32_t p_physicalDeviceCount) {
  vk::Queue queue = p_vkDevice.getQueue(p_graphicsQueueFamilyIndex, 0);
  vk::UniqueCommandPool cmdPool = p_vkDevice.createCommandPoolUnique({{}, p_graphicsQueueFamilyIndex});
  std::vector<vk::UniqueCommandBuffer> cmdBuffers = p_vkDevice.allocateCommandBuffersUnique(
      vk::CommandBufferAllocateInfo(cmdPool.get(), vk::CommandBufferLevel::ePrimary, p_physicalDeviceCount + 1));
  cmdBuffers.back()->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
  cmdBuffers.back()->resetQueryPool(m_queryPool.get(), 0, p_physicalDeviceCount);
  cmdBuffers.back()->end();
  vk::CommandBufferSubmitInfo resetCmdBufferSubmit(cmdBuffers.back().get(), 0b1);
  queue.submit2(vk::SubmitInfo2({}, {}, resetCmdBufferSubmit));
  queue.waitIdle();

  std::vector<vk::CommandBufferSubmitInfo> cmdBufferSubmits(p_physicalDeviceCount);
  for (uint32_t devIdx = 0; devIdx < p_physicalDeviceCount; ++devIdx) {
    cmdBuffers[devIdx]->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    cmdBuffers[devIdx]->writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, m_queryPool.get(), devIdx);
    cmdBuffers[devIdx]->end();
    cmdBufferSubmits[devIdx] = {cmdBuffers[devIdx].get(), 1u << devIdx};
  }
  queue.submit2(vk::SubmitInfo2({}, {}, cmdBufferSubmits));
  queue.waitIdle();

  auto [result, timestamps] = p_vkDevice.getQueryPoolResults<uint64_t>(m_queryPool.get(), 0, p_physicalDeviceCount,
                                                                       p_physicalDeviceCount * sizeof(uint64_t),
                                                                       sizeof(uint64_t), vk::QueryResultFlagBits::e64);
  VAP_WARN_UNLESS(result == vk::Result::eSuccess, "getQueryPoolResults() returned {}", vk::to_string(result));
  for (uint32_t devIdx = 0; devIdx < p_physicalDeviceCount; ++devIdx) {
    m_physicalDeviceOffsetsWAR.emplace_back(
        (static_cast<int64_t>(timestamps[devIdx]) - static_cast<int64_t>(timestamps[0])) * m_timestampPeriod);
    VAP_INFO_IF(0 < devIdx, "WAR: physical device {} timestamp offset: {}", devIdx, m_physicalDeviceOffsetsWAR[devIdx]);
  }
}

void VulkanAppProfiler::resetQueryPool(vk::CommandBuffer p_cmdBuffer) {
  p_cmdBuffer.resetQueryPool(m_queryPool.get(), 0, m_nextTimestampIndex);
  m_nextTimestampIndex = 0;
}

VulkanAppProfiler::TimestampIndex VulkanAppProfiler::writeNext(vk::CommandBuffer p_cmdBuffer,
                                                               vk::PipelineStageFlags2 p_stage) {
  VAP_ASSERT(m_nextTimestampIndex < m_maxTimestamps, "No more timestamps left.");
  p_cmdBuffer.writeTimestamp2(p_stage, m_queryPool.get(), m_nextTimestampIndex);
  return m_nextTimestampIndex++;
}

std::vector<uint64_t> VulkanAppProfiler::getResults(vk::Device p_vkDevice) const {
  if (m_nextTimestampIndex == 0) {
    return {};
  }
  auto [result, value] = p_vkDevice.getQueryPoolResults<uint64_t>(m_queryPool.get(), 0, m_nextTimestampIndex,
                                                                  m_nextTimestampIndex * sizeof(uint64_t),
                                                                  sizeof(uint64_t), vk::QueryResultFlagBits::e64);
  VAP_WARN_UNLESS(result == vk::Result::eSuccess, "Getting query pool results failed: {}", vk::to_string(result));
  return value;
}

void VulkanAppProfiler::pushInstant(const std::string &p_name, uint64_t p_frameIndex, uint32_t p_physicalDeviceIndex,
                                    vk::CommandBuffer p_cmdBuffer, vk::PipelineStageFlags2 p_stage) {
  if (m_enabled) {
    m_eventsInProgress.emplace(
        EventId{p_name, p_frameIndex},
        EventInProgress{
            .name = p_name,
            .frameIndex = p_frameIndex,
            .cpu = {{.primary = std::chrono::high_resolution_clock::now()}},
            .gpu = {{.primary = this->writeNext(p_cmdBuffer, p_stage), .physicalDeviceIndex = p_physicalDeviceIndex}},
        });
  }
}

void VulkanAppProfiler::pushDurationBegin(const std::string &p_name, uint64_t p_frameIndex,
                                          uint32_t p_physicalDeviceIndex, vk::CommandBuffer p_cmdBuffer,
                                          vk::PipelineStageFlags2 p_stage) {
  if (m_enabled) {
    this->pushInstant(p_name, p_frameIndex, p_physicalDeviceIndex, p_cmdBuffer, p_stage);
    m_openDurationEventStack.emplace_back(p_name, p_frameIndex);
  }
}

void VulkanAppProfiler::pushDurationEnd(vk::CommandBuffer p_cmdBuffer, vk::PipelineStageFlags2 p_stage) {
  if (m_enabled) {
    VAP_ASSERT(!m_openDurationEventStack.empty(), "No pending durations.");
    EventInProgress &evt = m_eventsInProgress.find(m_openDurationEventStack.back())->second;
    evt.gpu.value().secondary = this->writeNext(p_cmdBuffer, p_stage);
    evt.cpu.value().secondary = std::chrono::high_resolution_clock::now();
    m_openDurationEventStack.pop_back();
  }
}

void VulkanAppProfiler::pushCpuInstant(const std::string &p_name, uint64_t p_frameIndex) {
  if (m_enabled) {
    m_eventsInProgress.emplace(EventId{p_name, p_frameIndex},
                               EventInProgress{
                                   .name = p_name,
                                   .frameIndex = p_frameIndex,
                                   .cpu = {{.primary = std::chrono::high_resolution_clock::now()}},
                               });
  }
}

void VulkanAppProfiler::pushCpuDurationBegin(const std::string &p_name, uint64_t p_frameIndex) {
  if (m_enabled) {
    VAP_WARN_IF(std::find_if(m_openDurationEventStack.begin(), m_openDurationEventStack.end(),
                             [&](const EventId &p_evtId) { return p_evtId.first == p_name; }) !=
                    m_openDurationEventStack.end(),
                "Pushed two events of the same name: {} (during frame {})", p_name, p_frameIndex);
    this->pushCpuInstant(p_name, p_frameIndex);
    m_openDurationEventStack.emplace_back(p_name, p_frameIndex);
  }
}

void VulkanAppProfiler::pushCpuDurationEnd() {
  if (m_enabled) {
    VAP_ASSERT(!m_openDurationEventStack.empty(), "No pending durations.");
    EventInProgress &evt = m_eventsInProgress.find(m_openDurationEventStack.back())->second;
    evt.cpu.value().secondary = std::chrono::high_resolution_clock::now();
    m_openDurationEventStack.pop_back();
  }
}

VulkanAppProfiler::Events VulkanAppProfiler::flush(vk::Device p_vkDevice) {
  p_vkDevice.waitIdle();
  std::vector<uint64_t> timestamps = this->getResults(p_vkDevice);
  std::vector<std::string> traceEventLogEntries;
  for (auto &[evtId, evt] : m_eventsInProgress) {
    m_finishedEvents.emplace_back(Event{.name = evt.name, .frameIndex = evt.frameIndex});
    std::string evtName = std::format("{} (frame {})", evtId.first, evtId.second);
    if (evt.cpu && evt.cpu.value().secondary) {
      uint64_t tsBegNanos = evt.cpu.value().primary.time_since_epoch().count() - m_cpuCalibratedTimestamp;
      uint64_t tsEndNanos = evt.cpu.value().secondary.value().time_since_epoch().count() - m_cpuCalibratedTimestamp;
      m_finishedEvents.back().cpu = {.beginNanos = tsBegNanos, .endNanos = tsEndNanos};
    } else if (evt.cpu) {
      uint64_t tsNanos = evt.cpu.value().primary.time_since_epoch().count() - m_cpuCalibratedTimestamp;
      m_finishedEvents.back().cpu = {.beginNanos = tsNanos};
    }
    if (evt.gpu && evt.gpu.value().secondary) {
      uint64_t tsBegNanos = (timestamps[evt.gpu.value().primary] - m_gpuCalibratedTimestamp) * m_timestampPeriod;
      uint64_t tsEndNanos =
          (timestamps[evt.gpu.value().secondary.value()] - m_gpuCalibratedTimestamp) * m_timestampPeriod;
      if (!m_physicalDeviceOffsetsWAR.empty()) {
        tsBegNanos -= m_physicalDeviceOffsetsWAR[evt.gpu.value().physicalDeviceIndex];
        tsEndNanos -= m_physicalDeviceOffsetsWAR[evt.gpu.value().physicalDeviceIndex];
      }
      m_finishedEvents.back().gpu = {.beginNanos = tsBegNanos, .endNanos = tsEndNanos};
      m_finishedEvents.back().physicalDeviceIndex = evt.gpu.value().physicalDeviceIndex;
    } else if (evt.gpu) {
      uint64_t tsNanos = (timestamps[evt.gpu.value().primary] - m_gpuCalibratedTimestamp) * m_timestampPeriod;
      if (!m_physicalDeviceOffsetsWAR.empty()) {
        tsNanos -= m_physicalDeviceOffsetsWAR[evt.gpu.value().physicalDeviceIndex];
      }
      m_finishedEvents.back().gpu = {.beginNanos = tsNanos};
      m_finishedEvents.back().physicalDeviceIndex = evt.gpu.value().physicalDeviceIndex;
    }
  }
  m_eventsInProgress.clear();
  return m_finishedEvents;
}

#define DURATION_FMT                                                                                                   \
  "{{\"name\": \"{} ({})\", \"cat\": \"{}\", \"pid\": 0, \"tid\": {}, \"ph\": \"X\", \"ts\": {}, \"dur\": {}, "        \
  "\"args\": {{\"frame\": {}}}}}"

#define INSTANT_FMT                                                                                                    \
  "{{\"name\": \"{} ({})\", \"cat\": \"{}\", \"pid\": 0, \"tid\": {}, \"ph\": \"i\", \"ts\": {}, \"args\": "           \
  "{{\"frame\": {}}}}}"

void VulkanAppProfiler::writeTraceEventJson(const std::filesystem::path &p_filePath) {
  std::filesystem::create_directories(p_filePath.parent_path());
  if (std::ofstream traceEventLogFile(p_filePath); traceEventLogFile) {
    traceEventLogFile << "{" << std::endl;
    // traceEventLogFile << "  \"displayTimeUnit\": \"ns\"," << std::endl;
    traceEventLogFile << "  \"traceEvents\": [" << std::endl;
    for (size_t i = 0; i < m_finishedEvents.size(); ++i) {
      Event &evt = m_finishedEvents[i];
      std::string cpuEntry;
      std::optional<std::string> gpuEntry;
      if (evt.cpu.endNanos) {
        cpuEntry = std::format(DURATION_FMT, evt.name, evt.frameIndex, "cpu", 0, evt.cpu.beginNanos / 1000ull,
                               (evt.cpu.endNanos.value() - evt.cpu.beginNanos) / 1000ull, evt.frameIndex);
      } else {
        cpuEntry =
            std::format(INSTANT_FMT, evt.name, evt.frameIndex, "cpu", 0, evt.cpu.beginNanos / 1000ull, evt.frameIndex);
      }
      if (evt.gpu && evt.gpu.value().endNanos) {
        gpuEntry =
            std::format(DURATION_FMT, evt.name, evt.frameIndex, "gpu", 1 + evt.physicalDeviceIndex.value(),
                        evt.gpu.value().beginNanos / 1000ull,
                        (evt.gpu.value().endNanos.value() - evt.gpu.value().beginNanos) / 1000ull, evt.frameIndex);
      } else if (evt.gpu) {
        gpuEntry = std::format(INSTANT_FMT, evt.name, evt.frameIndex, "gpu", 1 + evt.physicalDeviceIndex.value(),
                               evt.gpu.value().beginNanos / 1000ull, evt.frameIndex);
      }
      traceEventLogFile << "    " << cpuEntry;
      if (gpuEntry) {
        traceEventLogFile << "," << std::endl << "    " << gpuEntry.value();
      }
      traceEventLogFile << (i == m_finishedEvents.size() - 1 ? "" : ",") << std::endl;
    }
    traceEventLogFile << "  ]";
    traceEventLogFile << "}";
    traceEventLogFile.close();
    VAP_INFO("Trace written to {}.", p_filePath.string());
  } else {
    VAP_WARN("Cannot open {} for writing.", p_filePath.string());
  }
  m_finishedEvents.clear();
}

VulkanAppProfiler::Scope::Scope(VulkanAppProfiler &p_profiler, const std::string &p_name, uint64_t p_frameIndex)
    : m_profiler(p_profiler), m_enabled(p_profiler.isEnabled()) {
  if (m_enabled) {
    m_profiler.pushCpuDurationBegin(p_name, p_frameIndex);
  }
}

VulkanAppProfiler::Scope::~Scope() {
  if (m_enabled && m_profiler.isEnabled()) {
    m_profiler.pushCpuDurationEnd();
  }
}

size_t VulkanAppProfiler::EventIdHash::operator()(const EventId &p_evtId) const {
  static std::hash<std::string> hash;
  return hash(p_evtId.first) + p_evtId.second;
}
} // namespace VAP_NAMESPACE
