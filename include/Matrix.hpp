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
#define _USE_MATH_DEFINES
#include <cmath>
#include <stdint.h>

namespace xrmg {
class Angle {
public:
  static const Angle ZERO;

  static Angle rad(float p_rad) { return {p_rad}; }
  static Angle deg(float p_deg) { return {static_cast<float>(M_PI) * p_deg / 180.0f}; }
  static Angle atan(float p_v) { return {std::atanf(p_v)}; }
  static Angle atan2(float p_num, float p_den) { return {std::atan2f(p_num, p_den)}; }

  Angle() : m_rad(0) {}

  float rad() const { return m_rad; }
  float deg() const { return 180.0f * m_rad / static_cast<float>(M_PI); }
  float sin() const { return std::sinf(m_rad); }
  float cos() const { return std::cosf(m_rad); }
  float tan() const { return std::tanf(m_rad); }
  Angle operator*(float p_factor) const { return {p_factor * m_rad}; }
  Angle operator/(float p_divisor) const { return {m_rad / p_divisor}; }
  Angle operator+(Angle p_right) const { return {m_rad + p_right.m_rad}; }
  Angle &operator+=(Angle p_right) { return *this = {m_rad + p_right.m_rad}; }
  bool operator<(Angle p_right) const { return m_rad < p_right.m_rad; }
  Angle operator-() const { return {-m_rad}; }

private:
  Angle(float p_rad) : m_rad(p_rad) {}

  float m_rad;
};

inline const Angle Angle::ZERO;

inline Angle operator*(float p_factor, const Angle &p_angle) { return Angle::rad(p_factor * p_angle.rad()); }

struct Vec2f {
  float x, y;
};

struct Rect2Df {
  float x;
  float y;
  float width;
  float height;
};

struct Vec3f {
  float x, y, z;

  Vec3f &operator+=(const Vec3f &p_value) { return *this = {x + p_value.x, y + p_value.y, z + p_value.z}; }
  Vec3f operator/(float p_divisor) const { return {x / p_divisor, y / p_divisor, z / p_divisor}; }

  Vec3f normalized() const { return *this / std::sqrtf(x * x + y * y + z * z); }
};

inline Vec3f operator*(float p_left, const Vec3f &p_right) {
  return {p_left * p_right.x, p_left * p_right.y, p_left * p_right.z};
}

struct Vec4f {
  union {
    float values[4];
    struct {
      float x, y, z, w;
    };
  };
};

struct Mat4x4f {
  static const Mat4x4f IDENTITY;
  static Mat4x4f createPerspectiveProjection(Angle p_verticalFov, float p_aspectRatio, float p_zNear, float p_zFar);
  static Mat4x4f createPerspectiveProjection(Angle p_horizontalFov, Angle p_verticalFov, float p_zNear, float p_zFar);
  static Mat4x4f createScaling(float p_x, float p_y, float p_z);
  static Mat4x4f createScaling(float p_factor) { return Mat4x4f::createScaling(p_factor, p_factor, p_factor); }
  static Mat4x4f createRotation(Angle p_roll, Angle p_pitch, Angle p_yaw);
  static Mat4x4f createRotation(float p_quatX, float p_quatY, float p_quatZ, float p_quatW);
  static Mat4x4f createRotationX(Angle p_angle);
  static Mat4x4f createRotationY(Angle p_angle);
  static Mat4x4f createRotationZ(Angle p_angle);
  static Mat4x4f createTranslation(float p_x, float p_y, float p_z);
  static Mat4x4f createTranslation(const Vec3f &p_xyz) { return Mat4x4f::createTranslation(p_xyz.x, p_xyz.y, p_xyz.z); }

  union {
    float v[4][4]; // row-major
    Vec4f rows[4];
  };

  Mat4x4f operator*(const Mat4x4f &p_right) const;
  Mat4x4f &operator*=(const Mat4x4f &p_right) { return *this = *this * p_right; }

  float det() const;
  [[nodiscard]] Mat4x4f invert() const;
  [[nodiscard]] Mat4x4f transpose() const;

  Vec3f transformCoord(const Vec3f &p_coord) const;
  Vec3f transformDir(const Vec3f &p_dir) const;
};

inline const Mat4x4f Mat4x4f::IDENTITY = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                                          0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
} // namespace xrmg
