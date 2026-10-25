struct VSInput {
    float3 inPosition;
    float4 inColor;
};

struct PrimitiveUBO {
    float3 position;
    uint8_t type;
    uint8_t operation;
    uint16_t _pad;
    float3 rotation;
    float smoothing;
    float3 scale;
    float __pad;
 };

struct UniformBuffer {
    float4x4 model;
    float4x4 inverseModel;
    float4x4 view;
    float4x4 inverseView;
    float4x4 proj;
    float4x4 inverseProj;
    float2 iResolution;
    float iTime;
    uint32_t primitiveCount;
    PrimitiveUBO primitives[16];
};
[[vk::binding(0, 0)]]
ConstantBuffer<UniformBuffer> ubo;

[[vk::push_constant]]
cbuffer PushConstants {
    float4 cameraPos;
    float4 lightPos;
    float4 lightColour;
};

struct VSOutput {
    float4 pos : SV_Position;
    float4 fragPos;
    nointerpolation float4 color;
    float4 normal;
    float depth;
    float hit;
    float4 nextNormal;
};

// --------------------- PRIMITIVES ----------------------------

float sdSphere(float3 p, float3 origin, float radius) { return length(p - origin) - radius; }
float sdPlane(float3 p, float3 origin, float3 normal) { return dot(p - origin, normal); }
float sdEllipsoid(float3 p, float3 origin, float3 r) {
    float3 pRel = p - origin;
    float k0 = length(pRel / r);
    float k1 = length(pRel / (r * r));
    return k0 * (k0 - 1.0) / k1;
}
float sdNaiveEllipsoid(float3 p, float3 origin, float3 radii) { return length(p - origin - radii); }
float sdLine(float3 p, float3 origin, float3 dir) { // dir must be normalized
    float3 pa = p - origin;
    float3 proj = dir * dot(pa, dir);
    return length(pa - proj);
}
float sdSegment(float3 p, float3 a, float3 b) {
    float3 pa = p - a;
    float3 ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h);
}
float sdCapsule(float3 p, float3 a, float3 b, float radius) {
    return sdSegment(p, a, b) - radius;
}

// ----------------- OPERATORS ----------------------

float opAdd(float d1, float d2) { return min(d1, d2); }
float opSmoothAdd(float d1, float d2, float k) { float h = clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0); return lerp(d2, d1, h) - k * h * (1.0 - h); }
float opSubtract(float d1, float d2) { return max(d1, -d2); }
float opSmoothSubtract(float d1, float d2, float k) { float h = clamp(0.5 - 0.5 * (d2 + d1) / k, 0.0, 1.0); return lerp(d1, -d2, h) + k * h * (1.0 - h); }
float opIntersect(float d1, float d2) { return max(d2, d1); }

// ------------------ MAPPING ---------------------

float map(float3 p) {
    float map = sdSphere(p, ubo.primitives[0].position, ubo.primitives[0].scale.x);

    for (int i = 1; i < 16; i++) {
        PrimitiveUBO prim = ubo.primitives[i];
        uint8_t op = prim.operation, type = prim.type;
        float3 pos = prim.position, scale = prim.scale;
        float smoothing = prim.smoothing;
        map = opSmoothAdd(map, sdSphere(p, pos, scale.x), smoothing);
    }
    return map;
}

// ------------------ MARCHING -------------------

struct MarchResult {
    float t;
    float hit;
    float minDist; // closest approach, even on a miss
    float minDistDepth;
    float lastMinDistDepth;
    float lastMinDist;
};

MarchResult vertMarch(float3 origin, float3 direction, float limit) {
    MarchResult result;
    float hit = -1.0;
    float minDist = 1e9; // big num
    float minDistDepth = 0.0;
    float t = 0.0;

    float lastMinDistDepth = 0.0;
    float lastMinDist = 1e9;

    for (int i = 0; i < 50; i++){
        float3 p = origin + (direction * t);
        float d = map(p);

        if (d < minDist) {
            minDistDepth = t;
            minDist = d;
        } else {
            lastMinDistDepth = minDistDepth;
            lastMinDist = minDist;
        }

        t += d;
        if (d < 0.001) { hit = 1.0; break; }
        if (t > limit) { break; };

    }

    result.minDist = minDist;
    result.minDistDepth = minDistDepth;
    result.lastMinDistDepth = lastMinDistDepth;
    result.lastMinDist = lastMinDist;
    result.t = t;
    result.hit = hit;
    return result;
}

float3 getNormal(float3 p, float delta) {
    float2 e = float2(delta, 0);
    return normalize(float3(
        map(p + e.xyy) - map(p - e.xyy),
        map(p + e.yxy) - map(p - e.yxy),
        map(p + e.yyx) - map(p - e.yyx)
    ));
}

[shader("vertex")]
VSOutput vertMain(VSInput input) {

    VSOutput output;

    float marchLimit = 50.0; // needs to be about half of frustum to prevent clipping for some reason

    // much like "mul(proj, mul(view, worldVert))"" converts world to clip, undoing this requires "mul(invView, mul(invProj, clip))"
    // I should probably precompute "mul(invProj, clip)", and that will be my new input (maybe do this later)

    float2 uv = float2(input.inPosition.x, -input.inPosition.y);
    float4 worldPos = mul(ubo.inverseView, mul(ubo.inverseProj, float4(uv, 0.0, 1.0)));
    float persp = worldPos.w;
    worldPos /= persp; // perspective divide.

    float3 marchDir = normalize(worldPos.xyz - cameraPos.xyz); // march away from camera.
    float totalDist = 0.0;
    MarchResult marchResult = vertMarch(cameraPos.xyz, marchDir, marchLimit);

    float3 marchPos = cameraPos.xyz + (marchDir * marchResult.t);
    float3 marchPosNudged = marchPos;

    float3 minDistPos = cameraPos.xyz + (marchDir * marchResult.minDistDepth);
    float3 lastMinDistPos = cameraPos.xyz + (marchDir * marchResult.lastMinDistDepth);
    float3 normal = getNormal(lastMinDistPos, 0.001);
    float facing = dot(normal, marchDir);

    bool cinch = false;

    if (marchResult.lastMinDist < 0.024 * marchResult.lastMinDistDepth) {
        marchPosNudged = marchPos - (normal * (marchResult.lastMinDist / marchResult.lastMinDistDepth) * marchResult.t);
        cinch = true;
    }

    float4 clipPosAdj = mul(ubo.proj, mul(ubo.view, float4(marchPosNudged, 1.0)));
    float3 normalAdj = marchResult.hit > 0.0 ?  getNormal(marchPosNudged, 0.001) : 0;

    // output result
    output.depth = marchResult.minDistDepth;
    output.nextNormal = float4(getNormal(cameraPos.xyz + (marchDir * marchResult.t), 0.001), 1.0);
    output.pos = clipPosAdj;
    output.normal = float4(normalAdj, 1.0);
    output.fragPos = float4(marchPosNudged, 1.0);

    output.color = input.inColor;
    //output.hit = marchResult.hit;

    return output;
}

[shader("fragment")]
float4 fragMain(VSOutput input) : SV_TARGET {

    //Ambient
    const float ambientStrength = 0.1;
    float3 ambient = ambientStrength * lightColour.rgb;
    float3 fragPos = input.fragPos.xyz;

    // Diffuse
    float3 N = normalize(input.normal.xyz);
    float3 V = normalize(cameraPos.xyz - fragPos);
    float viewAngle = max(dot(N, V), 0.0);
    N = lerp(normalize(input.normal.xyz), normalize(input.nextNormal.xyz), viewAngle);
    viewAngle = max(dot(N, V), 0.0);
    float3 L = normalize(lightPos.xyz - fragPos);
    float3 diffuse = max(dot(N, L), 0.0) * lightColour.rgb;
    //float3 diffuse = 1.0;

    // float A = cos(input.depth / 70.0 * 3.14);
    // float A = viewAngle; // viewAngle * (1 - input.depth /50.0);
    // float A = (input.hit) * (viewAngle);
    float A = 1.0;//viewAngle;

    float3 glow = 0;

    float4 result = float4(ambient + diffuse + glow, A) * input.color;
    return result;
}
