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

#include <filesystem>

namespace xrmg {
struct Options {
  std::optional<uint32_t> devGroupIndex;
  std::optional<uint32_t> simulatedPhysicalDeviceCount;
  std::optional<vk::Extent2D> windowClientAreaSize;
  std::optional<uint32_t> monitorIndex;
  vk::PresentModeKHR presentMode = vk::PresentModeKHR::eMailbox;
  std::optional<uint32_t> frameTimeLogInterval;
  std::optional<std::pair<uint32_t, uint32_t>> traceRange;
  std::filesystem::path traceFilePath = "./trace.json";
  uint32_t initialBaseTorusTesselation = 16;
  uint32_t initialBaseTorusCount = 5;
  uint32_t initialTorusLayerCount = 8;

  bool renderProjectionPlane = false;
  bool oxrCoreValidation = false;
  std::optional<vk::Extent2D> xrResolutionPerEye;
  vk::Format swapchainFormat = vk::Format::eR8G8B8A8Srgb;
  uint32_t swapchainImageCount = 3;
  bool swapEyes = false;

  Options(const std::vector<std::string> &p_args);

  void printUsageAndExit(int32_t p_exitCode) const;
};
} // namespace xrmg