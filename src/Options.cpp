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
#include "Options.hpp"

namespace xrmg {
std::optional<uint32_t> parseUintOption(const std::vector<std::string> &p_args, uint32_t &p_index, bool p_required) {
  XRMG_ASSERT(!p_required || p_index + 1 < p_args.size(), "Missing argument for {}.", p_args[p_index]);
  int32_t v;
  try {
    v = std::stoi(p_args[++p_index]);
    XRMG_ASSERT(!p_required || 0 <= v, "Value for {} must be positive.", p_args[p_index - 1]);
    return v < 0 ? std::nullopt : std::make_optional(static_cast<uint32_t>(v));
  } catch (std::invalid_argument &ex) {
    XRMG_ASSERT(!p_required, "Invalid value for {}.", p_args[p_index - 1]);
    --p_index;
    return std::nullopt;
  }
}

std::optional<std::pair<uint32_t, uint32_t>> parseUint2Option(const std::vector<std::string> &p_args, uint32_t &p_index,
                                                              bool p_required) {
  XRMG_ASSERT(!p_required || p_index + 2 < p_args.size(), "Missing arguments for {}.", p_args[p_index]);
  int32_t x;
  int32_t y;
  try {
    x = std::stoi(p_args[p_index + 1]);
    y = std::stoi(p_args[p_index + 2]);
    XRMG_ASSERT(!p_required || (0 <= x && 0 <= y), "Values for {} must be positive.", p_args[p_index]);
    if (0 <= x || 0 <= y) {
      p_index += 2;
      return std::make_optional(std::make_pair(static_cast<uint32_t>(x), static_cast<uint32_t>(y)));
    }
  } catch (std::invalid_argument &ex) {
    XRMG_ASSERT(!p_required, "Invalid value for {}.", p_args[p_index - 1]);
  }
  return std::nullopt;
}

Options::Options(const std::vector<std::string> &p_args) {
  if (std::find(p_args.begin(), p_args.end(), "--help") != p_args.end() ||
      std::find(p_args.begin(), p_args.end(), "-h") != p_args.end()) {
    this->printUsageAndExit(0);
  }

  for (uint32_t index = 1; index < p_args.size(); ++index) {
    if (p_args[index] == "-w" || p_args[index] == "--windowed") {
      std::optional<std::pair<std::uint32_t, std::uint32_t>> v = parseUint2Option(p_args, index, false);
      windowClientAreaSize = {v ? v.value().first : 1280, v ? v.value().second : 720};
      XRMG_INFO("Windowed rendering with client area size of {} x {}", windowClientAreaSize.value().width,
                windowClientAreaSize.value().height);
    } else if (p_args[index] == "-m" || p_args[index] == "--monitor") {
      monitorIndex = parseUintOption(p_args, index, true);
      XRMG_INFO("Fullscreen rendering on monitor {}", monitorIndex.value());
    } else if (p_args[index] == "--device-group") {
      devGroupIndex = parseUintOption(p_args, index, true);
      XRMG_INFO("Using device group {}.", devGroupIndex.value());
    } else if (p_args[index] == "--simulate") {
      simulatedPhysicalDeviceCount = parseUintOption(p_args, index, true);
      XRMG_ASSERT(simulatedPhysicalDeviceCount.value() == 2 || simulatedPhysicalDeviceCount.value() == 4,
                  "Simulated mode only available for 2 or 4 physical devices.");
      XRMG_WARN("Simulating {} physical devices on a single one.", simulatedPhysicalDeviceCount.value());
    } else if (p_args[index] == "--present-mode") {
      XRMG_ASSERT(++index < p_args.size(), "Missing value for --present-mode.");
      if (p_args[index] == "fifo") {
        presentMode = vk::PresentModeKHR::eFifo;
      } else if (p_args[index] == "fifoRelaxed") {
        presentMode = vk::PresentModeKHR::eFifoRelaxed;
      } else if (p_args[index] == "immediate") {
        presentMode = vk::PresentModeKHR::eImmediate;
      } else if (p_args[index] == "mailbox") {
        presentMode = vk::PresentModeKHR::eMailbox;
      } else {
        XRMG_FATAL("Unknown present mode: {}", p_args[index - 1]);
      }
      XRMG_INFO("Selected present mode: {}", vk::to_string(presentMode));
    } else if (p_args[index] == "--frame-time-log-interval") {
      frameTimeLogInterval = parseUintOption(p_args, index, true);
      XRMG_INFO("Frame time log interval: {} ms.", frameTimeLogInterval.value());
    } else if (p_args[index] == "--swap-eyes") {
      swapEyes = true;
    } else if (p_args[index] == "--render-projection-plane") {
      renderProjectionPlane = true;
    } else if (p_args[index] == "--trace-range") {
      traceRange = parseUint2Option(p_args, index, true);
    } else if (p_args[index] == "--trace-file") {
      XRMG_ASSERT(++index < p_args.size(), "Missing argument for --trace-file.");
      traceFilePath = p_args[index];
      XRMG_INFO("Trace file path set to {}.", traceFilePath.string());
      XRMG_ASSERT(!std::filesystem::is_directory(traceFilePath),
                  "Trace file path must not point to an existing directory.");
    } else if (p_args[index] == "--base-torus-tesselation") {
      initialBaseTorusTesselation = parseUintOption(p_args, index, true).value();
    } else if (p_args[index] == "--base-torus-count") {
      initialBaseTorusCount = parseUintOption(p_args, index, true).value();
    } else if (p_args[index] == "--torus-layer-count") {
      initialTorusLayerCount = parseUintOption(p_args, index, true).value();
    } else {
      XRMG_WARN("Unexpected argument {}", p_args[index]);
    }
  }

  XRMG_INFO("Initial base torus tesselation: {}", initialBaseTorusTesselation);
  XRMG_INFO("Initial base torus count: {}", initialBaseTorusCount);
  XRMG_INFO("Initial torus layer count: {}", initialTorusLayerCount);
  XRMG_INFO_IF(traceRange, "Tracing of frames {} to {} to file {}", traceRange.value().first, traceRange.value().second,
               traceFilePath.string());
  XRMG_ASSERT(!monitorIndex || !windowClientAreaSize,
              "Monitor index and window client area size must not be set simultaneously.");
  XRMG_INFO_UNLESS(windowClientAreaSize || monitorIndex, "Using OpenXR for rendering");
}

void Options::printUsageAndExit(int32_t p_exitCode) const {
  std::string usage = std::format(
      "Usage:\n"
      "  " SAMPLE_NAME " --help | -h\n"
      "  " SAMPLE_NAME " [--device-group <index>] [--simulate <count>] [--windowed [<width> <height>] | --monitor "
      "<index>] [--present-mode <string>] [--frame-time-log-interval <count>] [--trace-range <begin, end> "
      "[--trace-file <path>]] [--base-torus-tesselation <count>] [--base-torus-count <count>] [--torus-layer-count "
      "<count>]\n\n"
      "Options:\n"
      "  --help -h                            Show this text.\n"
      "  --device-group <index>               Select the device group to use explicitly by its index. Only device "
      "groups of size 2 and 4 are allowed when not in simulated mode. If absent, the first compatible device group "
      "will be used.\n"
      "  --simulate <count>                   Simulate multi-GPU rendering with <count> physical devices on a single "
      "one. All commands and resources will be executed and allocated on the first physical device of the selected "
      "device group; <count> must be 2 or 4.\n"
      "  --windowed [<width> <height>]        Open a window of size <width> x <height> instead of using OpenXR; "
      "default: 1280 x 720\n"
      "  --monitor <index>                    Open a fullscreen window on monitor <monitor index> instead of using "
      "OpenXR.\n"
      "  --present-mode <string>              Set present mode for windowed and fullscreen rendering. Must be one of "
      "{{fifo, fifoRelaxed, immediate, mailbox}}; default: mailbox.\n"
      "  --frame-time-log-interval <count>    Log the avg. frame time every <count> milliseconds to stdout.\n"
      "  --trace-range <begin, end>           Enable CPU and GPU tracing of frames <begin> to <end>.\n"
      "  --trace-file <path>                  Output file of tracing; default: {}\n"
      "  --base-torus-tesselation <count>     The initial parametric surface subdivision of each torus will be 2 x "
      "<count> x <count>; default: {}\n"
      "  --base-torus-count <count>           The number of tori per compass direction will be 2 x <count> x <count>; "
      "default: {}\n"
      "  --torus-layer-count <count>          The number of layers per torus to sculpt its spikes; default: {}",
      traceFilePath.string(), initialBaseTorusTesselation, initialBaseTorusCount, initialTorusLayerCount);
  XRMG_INFO("{}", usage);
  exit(p_exitCode);
}
} // namespace xrmg