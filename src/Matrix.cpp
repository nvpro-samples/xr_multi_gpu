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
#include "Matrix.hpp"

#include <cmath>

namespace xrmg {
Mat4x4f createPerspectiveProjection(float p_tanAlpha, float p_tanBeta, float p_zNear, float p_zFar) {
  Mat4x4f m = {};
  m.v[0][0] = 1.0f / p_tanAlpha;
  m.v[1][1] = -1.0f / p_tanBeta;
  m.v[2][2] = p_zFar / (p_zNear - p_zFar);
  m.v[2][3] = p_zNear * p_zFar / (p_zNear - p_zFar);
  m.v[3][2] = -1.0f;
  return m;
}

Mat4x4f Mat4x4f::createPerspectiveProjection(Angle p_verticalFov, float p_aspectRatio, float p_zNear, float p_zFar) {
  float tanBeta = std::tanf(0.5f * p_verticalFov.rad());
  float tanAlpha = p_aspectRatio * tanBeta;
  return xrmg::createPerspectiveProjection(tanAlpha, tanBeta, p_zNear, p_zFar);
}

Mat4x4f Mat4x4f::createPerspectiveProjection(Angle p_horizontalFov, Angle p_verticalFov, float p_zNear, float p_zFar) {
  float tanAlpha = std::tanf(0.5f * p_horizontalFov.rad());
  float tanBeta = std::tanf(0.5f * p_verticalFov.rad());
  return xrmg::createPerspectiveProjection(tanAlpha, tanBeta, p_zNear, p_zFar);
}

Mat4x4f Mat4x4f::createScaling(float p_x, float p_y, float p_z) {
  return {p_x, 0.0f, 0.0f, 0.0f, 0.0f, p_y, 0.0f, 0.0f, 0.0f, 0.0f, p_z, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
}

Mat4x4f Mat4x4f::createRotation(Angle p_roll, Angle p_pitch, Angle p_yaw) {
  Mat4x4f roll = Mat4x4f::createRotationZ(p_roll);
  Mat4x4f pitch = Mat4x4f::createRotationX(p_pitch);
  Mat4x4f yaw = Mat4x4f::createRotationY(p_yaw);
  return yaw * pitch * roll;
}

Mat4x4f Mat4x4f::createTranslation(float p_x, float p_y, float p_z) {
  return {1.0f, 0.0f, 0.0f, p_x, 0.0f, 1.0f, 0.0f, p_y, 0.0f, 0.0f, 1.0f, p_z, 0.0f, 0.0f, 0.0f, 1.0f};
}

Mat4x4f Mat4x4f::createRotation(float p_quatX, float p_quatY, float p_quatZ, float p_quatW) {
  return {2.0f * (p_quatW * p_quatW + p_quatX * p_quatX) - 1.0f,
          2.0f * (p_quatX * p_quatY - p_quatW * p_quatZ),
          2.0f * (p_quatX * p_quatZ + p_quatW * p_quatY),
          0.0f,
          2.0f * (p_quatX * p_quatY + p_quatW * p_quatZ),
          2.0f * (p_quatW * p_quatW + p_quatY * p_quatY) - 1.0f,
          2.0f * (p_quatY * p_quatZ - p_quatW * p_quatX),
          0.0f,
          2.0f * (p_quatX * p_quatZ - p_quatW * p_quatY),
          2.0f * (p_quatY * p_quatZ + p_quatW * p_quatX),
          2.0f * (p_quatW * p_quatW + p_quatZ * p_quatZ) - 1.0f,
          0.0f,
          0.0f,
          0.0f,
          0.0f,
          1.0f};
}

Mat4x4f Mat4x4f::createRotationX(Angle p_angle) {
  return {1.0f, 0.0f,          0.0f,          0.0f, 0.0f, p_angle.cos(), -p_angle.sin(), 0.0f,
          0.0f, p_angle.sin(), p_angle.cos(), 0.0f, 0.0f, 0.0f,          0.0f,           1.0f};
}

Mat4x4f Mat4x4f::createRotationY(Angle p_angle) {
  return {p_angle.cos(),  0.0f, p_angle.sin(), 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
          -p_angle.sin(), 0.0f, p_angle.cos(), 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
}

Mat4x4f Mat4x4f::createRotationZ(Angle p_angle) {
  return {p_angle.cos(), -p_angle.sin(), 0.0f, 0.0f, p_angle.sin(), p_angle.cos(), 0.0f, 0.0f,
          0.0f,          0.0f,           1.0f, 0.0f, 0.0f,          0.0f,          0.0f, 1.0f};
}

Mat4x4f Mat4x4f::operator*(const Mat4x4f &p_right) const {
  const float (*w)[4] = p_right.v;
  return {v[0][0] * w[0][0] + v[0][1] * w[1][0] + v[0][2] * w[2][0] + v[0][3] * w[3][0],
          v[0][0] * w[0][1] + v[0][1] * w[1][1] + v[0][2] * w[2][1] + v[0][3] * w[3][1],
          v[0][0] * w[0][2] + v[0][1] * w[1][2] + v[0][2] * w[2][2] + v[0][3] * w[3][2],
          v[0][0] * w[0][3] + v[0][1] * w[1][3] + v[0][2] * w[2][3] + v[0][3] * w[3][3],
          v[1][0] * w[0][0] + v[1][1] * w[1][0] + v[1][2] * w[2][0] + v[1][3] * w[3][0],
          v[1][0] * w[0][1] + v[1][1] * w[1][1] + v[1][2] * w[2][1] + v[1][3] * w[3][1],
          v[1][0] * w[0][2] + v[1][1] * w[1][2] + v[1][2] * w[2][2] + v[1][3] * w[3][2],
          v[1][0] * w[0][3] + v[1][1] * w[1][3] + v[1][2] * w[2][3] + v[1][3] * w[3][3],
          v[2][0] * w[0][0] + v[2][1] * w[1][0] + v[2][2] * w[2][0] + v[2][3] * w[3][0],
          v[2][0] * w[0][1] + v[2][1] * w[1][1] + v[2][2] * w[2][1] + v[2][3] * w[3][1],
          v[2][0] * w[0][2] + v[2][1] * w[1][2] + v[2][2] * w[2][2] + v[2][3] * w[3][2],
          v[2][0] * w[0][3] + v[2][1] * w[1][3] + v[2][2] * w[2][3] + v[2][3] * w[3][3],
          v[3][0] * w[0][0] + v[3][1] * w[1][0] + v[3][2] * w[2][0] + v[3][3] * w[3][0],
          v[3][0] * w[0][1] + v[3][1] * w[1][1] + v[3][2] * w[2][1] + v[3][3] * w[3][1],
          v[3][0] * w[0][2] + v[3][1] * w[1][2] + v[3][2] * w[2][2] + v[3][3] * w[3][2],
          v[3][0] * w[0][3] + v[3][1] * w[1][3] + v[3][2] * w[2][3] + v[3][3] * w[3][3]};
}

float Mat4x4f::det() const {
  return v[0][0] * (v[1][1] * v[2][2] * v[3][3] + v[2][1] * v[3][2] * v[1][3] + v[3][1] * v[1][2] * v[2][3] -
                    v[1][3] * v[2][2] * v[3][1] - v[2][3] * v[3][2] * v[1][1] - v[3][3] * v[1][2] * v[2][1]) -
         v[0][1] * (v[1][0] * v[2][2] * v[3][3] + v[2][0] * v[3][2] * v[1][3] + v[3][0] * v[1][2] * v[2][3] -
                    v[1][3] * v[2][2] * v[3][0] - v[2][3] * v[3][2] * v[1][0] - v[3][3] * v[1][2] * v[2][0]) +
         v[0][2] * (v[1][0] * v[2][1] * v[3][3] + v[2][0] * v[3][1] * v[1][3] + v[3][0] * v[1][1] * v[2][3] -
                    v[1][3] * v[2][1] * v[3][0] - v[2][3] * v[3][1] * v[1][0] - v[3][3] * v[1][1] * v[2][0]) -
         v[0][3] * (v[1][0] * v[2][1] * v[3][2] + v[2][0] * v[3][1] * v[1][2] + v[3][0] * v[1][1] * v[2][2] -
                    v[1][2] * v[2][1] * v[3][0] - v[2][2] * v[3][1] * v[1][0] - v[3][2] * v[1][1] * v[2][0]);
}

Mat4x4f Mat4x4f::invert() const {
  float oneOverDet = 1.0f / this->det();
  Mat4x4f m;
  m.v[0][0] = +oneOverDet * (v[1][1] * v[2][2] * v[3][3] + v[2][1] * v[3][2] * v[1][3] + v[3][1] * v[1][2] * v[2][3] -
                             v[1][3] * v[2][2] * v[3][1] - v[2][3] * v[3][2] * v[1][1] - v[3][3] * v[1][2] * v[2][1]);
  m.v[1][0] = -oneOverDet * (v[1][0] * v[2][2] * v[3][3] + v[2][0] * v[3][2] * v[1][3] + v[3][0] * v[1][2] * v[2][3] -
                             v[1][3] * v[2][2] * v[3][0] - v[2][3] * v[3][2] * v[1][0] - v[3][3] * v[1][2] * v[2][0]);
  m.v[2][0] = +oneOverDet * (v[1][0] * v[2][1] * v[3][3] + v[2][0] * v[3][1] * v[1][3] + v[3][0] * v[1][1] * v[2][3] -
                             v[1][3] * v[2][1] * v[3][0] - v[2][3] * v[3][1] * v[1][0] - v[3][3] * v[1][1] * v[2][0]);
  m.v[3][0] = -oneOverDet * (v[1][0] * v[2][1] * v[3][2] + v[2][0] * v[3][1] * v[1][2] + v[3][0] * v[1][1] * v[2][2] -
                             v[1][2] * v[2][1] * v[3][0] - v[2][2] * v[3][1] * v[1][0] - v[3][2] * v[1][1] * v[2][0]);
  m.v[0][1] = -oneOverDet * (v[0][1] * v[2][2] * v[3][3] + v[2][1] * v[3][2] * v[0][3] + v[3][1] * v[0][2] * v[2][3] -
                             v[0][3] * v[2][2] * v[3][1] - v[2][3] * v[3][2] * v[0][1] - v[3][3] * v[0][2] * v[2][1]);
  m.v[1][1] = +oneOverDet * (v[0][0] * v[2][2] * v[3][3] + v[2][0] * v[3][2] * v[0][3] + v[3][0] * v[0][2] * v[2][3] -
                             v[0][3] * v[2][2] * v[3][0] - v[2][3] * v[3][2] * v[0][0] - v[3][3] * v[0][2] * v[2][0]);
  m.v[2][1] = -oneOverDet * (v[0][0] * v[2][1] * v[3][3] + v[2][0] * v[3][1] * v[0][3] + v[3][0] * v[0][1] * v[2][3] -
                             v[0][3] * v[2][1] * v[3][0] - v[2][3] * v[3][1] * v[0][0] - v[3][3] * v[0][1] * v[2][0]);
  m.v[3][1] = +oneOverDet * (v[0][0] * v[2][1] * v[3][2] + v[2][0] * v[3][1] * v[0][2] + v[3][0] * v[0][1] * v[2][2] -
                             v[0][2] * v[2][1] * v[3][0] - v[2][2] * v[3][1] * v[0][0] - v[3][2] * v[0][1] * v[2][0]);
  m.v[0][2] = +oneOverDet * (v[0][1] * v[1][2] * v[3][3] + v[1][1] * v[3][2] * v[0][3] + v[3][1] * v[0][2] * v[1][3] -
                             v[0][3] * v[1][2] * v[3][1] - v[1][3] * v[3][2] * v[0][1] - v[3][3] * v[0][2] * v[1][1]);
  m.v[1][2] = -oneOverDet * (v[0][0] * v[1][2] * v[3][3] + v[1][0] * v[3][2] * v[0][3] + v[3][0] * v[0][2] * v[1][3] -
                             v[0][3] * v[1][2] * v[3][0] - v[1][3] * v[3][2] * v[0][0] - v[3][3] * v[0][2] * v[1][0]);
  m.v[2][2] = +oneOverDet * (v[0][0] * v[1][1] * v[3][3] + v[1][0] * v[3][1] * v[0][3] + v[3][0] * v[0][1] * v[1][3] -
                             v[0][3] * v[1][1] * v[3][0] - v[1][3] * v[3][1] * v[0][0] - v[3][3] * v[0][1] * v[1][0]);
  m.v[3][2] = -oneOverDet * (v[0][0] * v[1][1] * v[3][2] + v[1][0] * v[3][1] * v[0][2] + v[3][0] * v[0][1] * v[1][2] -
                             v[0][2] * v[1][1] * v[3][0] - v[1][2] * v[3][1] * v[0][0] - v[3][2] * v[0][1] * v[1][0]);
  m.v[0][3] = -oneOverDet * (v[0][1] * v[1][2] * v[2][3] + v[1][1] * v[2][2] * v[0][3] + v[2][1] * v[0][2] * v[1][3] -
                             v[0][3] * v[1][2] * v[2][1] - v[1][3] * v[2][2] * v[0][1] - v[2][3] * v[0][2] * v[1][1]);
  m.v[1][3] = +oneOverDet * (v[0][0] * v[1][2] * v[2][3] + v[1][0] * v[2][2] * v[0][3] + v[2][0] * v[0][2] * v[1][3] -
                             v[0][3] * v[1][2] * v[2][0] - v[1][3] * v[2][2] * v[0][0] - v[2][3] * v[0][2] * v[1][0]);
  m.v[2][3] = -oneOverDet * (v[0][0] * v[1][1] * v[2][3] + v[1][0] * v[2][1] * v[0][3] + v[2][0] * v[0][1] * v[1][3] -
                             v[0][3] * v[1][1] * v[2][0] - v[1][3] * v[2][1] * v[0][0] - v[2][3] * v[0][1] * v[1][0]);
  m.v[3][3] = +oneOverDet * (v[0][0] * v[1][1] * v[2][2] + v[1][0] * v[2][1] * v[0][2] + v[2][0] * v[0][1] * v[1][2] -
                             v[0][2] * v[1][1] * v[2][0] - v[1][2] * v[2][1] * v[0][0] - v[2][2] * v[0][1] * v[1][0]);
  return m;
}

Mat4x4f Mat4x4f::transpose() const {
  return {v[0][0], v[1][0], v[2][0], v[3][0], v[0][1], v[1][1], v[2][1], v[3][1],
          v[0][2], v[1][2], v[2][2], v[3][2], v[0][3], v[1][3], v[2][3], v[3][3]};
}

Vec3f Mat4x4f::transformCoord(const Vec3f &p_coord) const {
  return {
      .x = v[0][0] * p_coord.x + v[0][1] * p_coord.y + v[0][2] * p_coord.z + v[0][3],
      .y = v[1][0] * p_coord.x + v[1][1] * p_coord.y + v[1][2] * p_coord.z + v[1][3],
      .z = v[2][0] * p_coord.x + v[2][1] * p_coord.y + v[2][2] * p_coord.z + v[2][3],
  };
}

Vec3f Mat4x4f::transformDir(const Vec3f &p_dir) const {
  return {
      .x = v[0][0] * p_dir.x + v[0][1] * p_dir.y + v[0][2] * p_dir.z,
      .y = v[1][0] * p_dir.x + v[1][1] * p_dir.y + v[1][2] * p_dir.z,
      .z = v[2][0] * p_dir.x + v[2][1] * p_dir.y + v[2][2] * p_dir.z,
  };
}

} // namespace xrmg
