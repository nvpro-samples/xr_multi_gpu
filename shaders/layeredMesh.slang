#include "perlin.h"

struct Camera {
  float4x4 view;
  float4x4 projection;
};

[[vk::push_constant]]
const Camera g_camera;

struct Vertex {
  float3 pos;
  float3 normal;
  float2 tex;
};

struct Instance {
  float4x4 localToGlobal;
  float4x4 localToGlobalIT;
  uint32_t colorHint;
  float relativeExtrusion;
  float absoluteExtrusion;
};

struct Fragment {
  float4 pos : SV_Position;
  float3 normal;
  float2 tex;
  uint32_t colorHint;
  float relativeExtrusion;
};

[shader("vertex")]
Fragment vs(Vertex p_vertex, Instance p_instance) {
  Fragment fragment = {};
  float3 extruded = p_vertex.pos + p_instance.absoluteExtrusion * normalize(p_vertex.normal);
  fragment.pos = mul(g_camera.projection, mul(g_camera.view, mul(p_instance.localToGlobal, float4(extruded, 1.0f))));
  fragment.normal = mul(p_instance.localToGlobalIT, float4(p_vertex.normal, 0.0f)).xyz;
  fragment.tex = p_vertex.tex;
  fragment.colorHint = p_instance.colorHint;
  fragment.relativeExtrusion = p_instance.relativeExtrusion;
  return fragment;
}

float2 rot2D(float2 src, float angleInRadians) {
  float sina = sin(angleInRadians);
  float cosa = cos(angleInRadians);
  return float2(cosa * src.x - sina * src.y, sina * src.x + cosa * src.y);
}

float perlinLayered(float2 tex) {
  float v = 0.5f * perlin(18.0f * rot2D(tex, -0.2f));
  v += 0.25f * perlin(31.0f * rot2D(tex, 0.3f));
  v += 0.125f * perlin(65.0f * rot2D(tex, 0.7f));
  v += 0.0625f * perlin(123.0f * rot2D(tex, -0.1f));
  return 0.5f + 8.0f / 15.0f * v;
}

float3 toFloat3Color(uint value) {
  return float3(float((value >> 16) & 0xff), float((value) >> 8 & 0xff), float(value & 0xff)) / 255.0f;
}

static const uint g_numGradients = 6;
static const uint g_gradientSize = 3;
static const float3 g_gradients[g_numGradients][g_gradientSize] = {
  { toFloat3Color(0xfff1bf), toFloat3Color(0xec458d), toFloat3Color(0x474ed7) },
  { toFloat3Color(0x2c6cbc), toFloat3Color(0x71c3f7), toFloat3Color(0xf6f6f6) },
  { toFloat3Color(0x1a2766), toFloat3Color(0xae1b1e), toFloat3Color(0xfc9f32) },
  { toFloat3Color(0x074170), toFloat3Color(0x7e9012), toFloat3Color(0xfff708) },
  { toFloat3Color(0x40E0D0), toFloat3Color(0xFF8C00), toFloat3Color(0xFF0080) },
  { toFloat3Color(0xff0000), toFloat3Color(0x00ff00), toFloat3Color(0x0000ff) }
};

float3 sampleColorGradient(uint colorGradientIdx, float t) {
  t = clamp(t, 0.0f, 1.0f);
  float dt = 1.0f / (g_gradientSize - 1);
  float t0 = 0.0f;
  uint i0 = 0;
  while (t0 + dt < t) {
    t0 += dt;
    ++i0;
  }
  return lerp(g_gradients[colorGradientIdx][i0], g_gradients[colorGradientIdx][i0 + 1], (t - t0) / dt);
}

static const float3 g_upColor = float3(0.529f, 0.807f, 0.921f);
static const float3 g_downColor = float3(0.02f, 0.02f, 0.02f);
static const float g_minAmbientOcclusion = 0.2f;

[shader("fragment")]
float4 fs(Fragment p_fragment) {
  float2 tex =
      rot2D(p_fragment.tex, lerp(p_fragment.tex.x, p_fragment.tex.y, sin(p_fragment.tex.x + p_fragment.tex.y)));
  float ao = lerp(g_minAmbientOcclusion, 1.0f, p_fragment.relativeExtrusion);
  if (0.0f < p_fragment.relativeExtrusion) {
    float2 baseTex = 128.0f * float2(tex.x, 2.0f * tex.y);
    float offset = distance(floor(baseTex) + 0.5f, baseTex);
    float height = 1.0f; // 0.5f + 0.5f * perlin(32.0f * float2(tex.x, 2.0f * tex.y));
    float cone = (height - p_fragment.relativeExtrusion) / (2.0f * height);
    if (cone < offset) {
      discard;
    }
  }

  float noise = perlinLayered(float2(tex.x, 2.0f * tex.y));
  noise = smoothstep(0.0f, 1.0f, noise);
  float3 color = sampleColorGradient(p_fragment.colorHint % g_numGradients, noise);
  // color = float3(noise);

  float3 n = normalize(p_fragment.normal);

  color *= lerp(g_downColor, g_upColor, 0.5f + 0.5f * n.y) * ao;
  return float4(color, 1.0f);
}
