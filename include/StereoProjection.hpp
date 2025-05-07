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

#include "Matrix.hpp"

#include <utility>

namespace xrmg {
struct StereoProjection {
  enum class Eye : int32_t { LEFT = 0, RIGHT = 1 };

  static Mat4x4f createStereoEyeTranslation(Eye p_eye, float p_ipd);
  static StereoProjection create(Angle p_left, Angle p_right, Angle p_up, Angle p_down, float p_zNear, float p_zFar);
  static StereoProjection create(Eye p_eye, float p_ipd, float p_projectionPlaneDistance, Angle p_verticalFov,
                                 float p_aspectRatio, float p_zNear, float p_zFar);

  Mat4x4f projectionMatrix;
  Rect2Df relativeViewport;
};
} // namespace xrmg
