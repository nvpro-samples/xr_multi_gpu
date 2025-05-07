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
#include "StereoProjection.hpp"

namespace xrmg {
StereoProjection StereoProjection::create(Angle p_left, Angle p_right, Angle p_up, Angle p_down, float p_zNear,
                                          float p_zFar) {
  XRMG_ASSERT(p_left.rad() <= 0.0f && 0.0f <= p_right.rad() && p_down.rad() <= 0.0f && 0.0f <= p_up.rad(),
              "Left and up angle must be negative and right and down angle must be positive.");
  Angle hFov2 = Angle::rad(std::fmaxf(std::fabsf(p_left.rad()), std::fabsf(p_right.rad())));
  float w1 = std::tanf(std::fabsf(p_left.rad()));
  float w2 = std::tanf(std::fabsf(p_right.rad()));
  Angle vFov2 = Angle::rad(std::max(std::abs(p_up.rad()), std::abs(p_down.rad())));
  float h1 = std::tanf(std::fabsf(p_up.rad()));
  float h2 = std::tanf(std::fabsf(p_down.rad()));
  return {.projectionMatrix = Mat4x4f::createPerspectiveProjection(2.0f * hFov2, 2.0f * vFov2, p_zNear, p_zFar),
          .relativeViewport = {.x = w2 < w1 ? 0.0f : (w1 - w2) / (w1 + w2),
                               .y = h2 < h1 ? 0.0f : (h1 - h2) / (h1 + h2),
                               .width = 2.0f * std::fmaxf(w1, w2) / (w1 + w2),
                               .height = 2.0f * std::fmaxf(h1, h2) / (h1 + h2)}};
}

StereoProjection StereoProjection::create(Eye p_eye, float p_ipd, float p_projectionPlaneDistance, Angle p_verticalFov,
                                          float p_aspectRatio, float p_zNear, float p_zFar) {
  float tanAlpha = p_aspectRatio * (0.5f * p_verticalFov).tan();
  Angle left = -Angle::atan(tanAlpha + 0.5f * p_ipd / p_projectionPlaneDistance);
  Angle right = Angle::atan(tanAlpha - 0.5f * p_ipd / p_projectionPlaneDistance);
  if (p_eye == Eye::LEFT) {
    Angle oldLeft = left;
    left = -right;
    right = -oldLeft;
  }
  Angle up = 0.5f * p_verticalFov;
  Angle down = -up;
  return StereoProjection::create(left, right, up, down, p_zNear, p_zFar);
}

Mat4x4f StereoProjection::createStereoEyeTranslation(Eye p_eye, float p_ipd) {
  return Mat4x4f::createTranslation((static_cast<float>(p_eye) - 0.5f) * p_ipd, 0.0f, 0.0f);
}
} // namespace xrmg
