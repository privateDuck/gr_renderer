float random(float x) {
  float f = 43758.5453123;
  return frac(sin(x) * f);
}

float random(float2 x) {
  float2 r = float2(12.9898, 78.233);
  return random(dot(x, r));
}

float mod(float x, float y) { return x - y * floor(x / y); }

float2 mod(float2 x, float2 y) { return x - y * floor(x / y); }
float3 mod(float3 x, float3 y) { return x - y * floor(x / y); }
float4 mod(float4 x, float4 y) { return x - y * floor(x / y); }

float noise(float x) {
  float i = floor(x);
  float fr = frac(x);
  return lerp(random(i), random(i + 1.0), fr);
}

float noise(float2 x) {
  float2 i = floor(x);
  float2 fr = frac(x);
  fr = fr * fr * (3.0 - 2.0 * fr);
  float res = lerp(
      lerp(random(i), random(i + float2(1.0, 0.0)), fr.x),
      lerp(random(i + float2(0.0, 1.0)), random(i + float2(1.0, 1.0)), fr.x),
      fr.y);
  return res * res;
}

float4 fl4(float x) { return float4(x, x, x, x); }

//	Simplex 3D Noise by Ian McEwan, Ashima Arts
float4 permute(float4 x) { return mod(((x * 34.0) + 1.0) * x, fl4(289.0)); }
float4 taylorInvSqrt(float4 r) {
  return 1.79284291400159 - 0.85373472095314 * r;
}

float noise(float3 v) {
  const float2 C = float2(1.0 / 6.0, 1.0 / 3.0);
  const float4 D = float4(0.0, 0.5, 1.0, 2.0);

  // First corner
  float3 i = floor(v + dot(v, C.yyy));
  float3 x0 = v - i + dot(i, C.xxx);

  // Other corners
  float3 g = step(x0.yzx, x0.xyz);
  float3 l = 1.0 - g;
  float3 i1 = min(g.xyz, l.zxy);
  float3 i2 = max(g.xyz, l.zxy);

  //  x0 = x0 - 0. + 0.0 * C
  float3 x1 = x0 - i1 + 1.0 * C.xxx;
  float3 x2 = x0 - i2 + 2.0 * C.xxx;
  float3 x3 = x0 - 1. + 3.0 * C.xxx;

  // Permutations
  i = mod(i, 289.0);
  float4 p = permute(permute(permute(i.z + float4(0.0, i1.z, i2.z, 1.0)) + i.y +
                             float4(0.0, i1.y, i2.y, 1.0)) +
                     i.x + float4(0.0, i1.x, i2.x, 1.0));

  // Gradients
  // ( N*N points uniformly over a square, mapped onto an octahedron.)
  float n_ = 1.0 / 7.0; // N=7
  float3 ns = n_ * D.wyz - D.xzx;

  float4 j = p - 49.0 * floor(p * ns.z * ns.z); //  mod(p,N*N)

  float4 x_ = floor(j * ns.z);
  float4 y_ = floor(j - 7.0 * x_); // mod(j,N)

  float4 x = x_ * ns.x + ns.yyyy;
  float4 y = y_ * ns.x + ns.yyyy;
  float4 h = 1.0 - abs(x) - abs(y);

  float4 b0 = float4(x.xy, y.xy);
  float4 b1 = float4(x.zw, y.zw);

  float4 s0 = floor(b0) * 2.0 + 1.0;
  float4 s1 = floor(b1) * 2.0 + 1.0;
  float4 sh = -step(h, fl4(0.0));

  float4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
  float4 a1 = b1.xzyw + s1.xzyw * sh.zzww;

  float3 p0 = float3(a0.xy, h.x);
  float3 p1 = float3(a0.zw, h.y);
  float3 p2 = float3(a1.xy, h.z);
  float3 p3 = float3(a1.zw, h.w);

  // Normalise gradients
  float4 norm =
      taylorInvSqrt(float4(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
  p0 *= norm.x;
  p1 *= norm.y;
  p2 *= norm.z;
  p3 *= norm.w;

  // Mix final noise value
  float4 m = max(
      0.6 - float4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), 0.0);
  m = m * m;
  return 42.0 *
         dot(m * m, float4(dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3)));
}

//
// FBM functions.
//

float fbm(float x, int octaves, float amplitude, float frequency,
          float lacunarity, float gain) {
  float value = 0.0;
  for (int i = 0; i < octaves; ++i) {
    value += amplitude * noise(x * frequency);
    frequency *= lacunarity;
    amplitude *= gain;
  }
  return value;
}

float fbm(float2 x, int octaves, float amplitude, float frequency,
          float lacunarity, float gain) {
  float value = 0.0;
  // Rotate to reduce axial bias
  float2x2 rot = float2x2(cos(0.5), sin(0.5), -sin(0.5), cos(0.50));
  for (int i = 0; i < octaves; ++i) {
    value += amplitude * noise(x * frequency);
    x = mul(rot, x);
    frequency *= lacunarity;
    amplitude *= gain;
  }
  return value;
}

float fbm_tileable_x(float2 p, float period, int octaves, float amplitude,
                     float frequency, float lacunarity, float gain) {
  float value = 0.0;
  float pi2 = 6.28318530718;

  // We map the X coordinate to an angle (0 to 2*PI over the given period)
  float angle = (p.x / period) * pi2;
  float2 dir = float2(cos(angle), sin(angle));

  for (int i = 0; i < octaves; ++i) {
    // To maintain the exact same spatial scale as 2D noise,
    // the circumference of the cylinder must be (period * frequency)
    float circ = period * frequency;
    float radius = circ / pi2;

    // Map 2D coordinate onto the 3D cylinder
    float3 p3 = float3(dir.x * radius, dir.y * radius, p.y * frequency);

    // Sample your 3D Simplex noise
    value += amplitude * noise(p3);

    // Advance the octave parameters
    frequency *= lacunarity;
    amplitude *= gain;

    // We drop the 2D rotation to preserve the X-tiling.
    // Instead, shift the Y-axis significantly to prevent octaves from
    // overlapping identical noise patterns.
    p.y += 13.37;
  }

  return value;
}

float fbm_disk_gas(float2 pos, float r, float logR, float stretch_azimuth,
                   float stretch_radial, float rotation_angle, int octaves,
                   float amplitude, float frequency, float lacunarity,
                   float gain) {
  float value = 0.0;

  float c = cos(rotation_angle);
  float s = sin(rotation_angle);

  // Rotate the local Cartesian coordinates directly
  float2 loop_coords =
      float2(pos.x * c + pos.y * s, pos.y * c - pos.x * s) * stretch_azimuth;

  // Distance from the center.
  float z_coord = logR * stretch_radial;

  float3 p = float3(loop_coords.x, loop_coords.y, z_coord);

  for (int i = 0; i < octaves; ++i) {
    value += amplitude * noise(p * frequency);
    frequency *= lacunarity;
    amplitude *= gain;

    // Random offsets to prevent octaves from stacking identical artifact patterns
    p += float3(13.37, 42.0, 7.77);
  }

  return value;
}

float fbm_disk_gas(float2 pos, float r, float logR, float stretch_azimuth,
                   float stretch_radial, float rotation_angle) {
  return fbm_disk_gas(pos, r, logR, stretch_azimuth, stretch_radial,
                      rotation_angle, 5, 0.5, 1.0, 2.0, 0.5);
}

float fbm(float3 x, int octaves, float amplitude, float frequency, float3 shift,
          float lacunarity, float gain) {
  float value = 0.0;
  for (int i = 0; i < octaves; ++i) {
    float sn = noise(x * frequency);
    value += amplitude * sn;
    x += shift;
    frequency *= lacunarity;
    amplitude *= gain;
  }
  return value;
}

float fbm(float x, int octaves) { return fbm(x, octaves, 0.5, 1.0, 2.0, 0.5); }

float fbm(float2 x, int octaves) { return fbm(x, octaves, 0.5, 1.0, 2.0, 0.5); }
float fbm_tile(float2 x, int octaves, float period) {
  return fbm_tileable_x(x, period, octaves, 0.5, 1.0, 2.0, 0.5);
}

float fbm(float3 uv, int octaves) {
  return fbm(uv, octaves, 0.5, 1.0, float3(8, 8, 8), 2.0, 0.5);
}