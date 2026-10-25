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
    PrimitiveUBO primitives[128];
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
    PrimitiveUBO prim = ubo.primitives[0];
    float map = sdSphere(p, prim.position, prim.scale.x);

    for (int i = 1; i < 32; i++) {
        prim = ubo.primitives[i];
        //uint8_t op = prim.operation, type = prim.type;

        float3 pos = prim.position, scale = prim.scale;
        float smoothing = prim.smoothing;

        map = opSmoothAdd(map, sdSphere(p, pos, scale.x), smoothing);
    }

    //map = opSmoothSubtract(map, sdPlane(p, float3(0, 0, 0), float3(0, 1, 0)), 0.5);

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

MarchResult vertMarch(float3 origin, float3 direction, float limit, bool stopAtMin) {
    MarchResult result;
    float hit = -1.0;
    float minDist = 1e9; // big num
    float t = 0.0;
    float minDistDepth = 0.0;

    float lastMinDistDepth = 0.0;
    float lastMinDist = 1e9;

    for (int i = 0; i < 50; i++){
        float3 p = origin + (direction * t);
        float d = map(p);

        t += d;

        if (d < minDist) {
            minDistDepth = t;
            minDist = d;
        } else {
            lastMinDistDepth = minDistDepth;
            lastMinDist = minDist;
        }

        if (d < 0.01) { hit = 1.0; break; }
        if (t > limit || (d > minDist && stopAtMin)) { break; };
    }

    result.minDistDepth = minDistDepth;
    result.lastMinDistDepth = lastMinDistDepth;
    result.lastMinDist = lastMinDist;
    result.minDist = minDist;
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
    MarchResult marchResult = vertMarch(cameraPos.xyz, marchDir, marchLimit, false);
    float3 marchPos = cameraPos.xyz + marchDir * marchResult.t;
    float3 normal = getNormal(marchPos, 0.001);

    float3 lastMinDistPos = cameraPos.xyz + (marchDir * marchResult.lastMinDistDepth); // empty space next to pervious edge
    float3 normalOff = getNormal(lastMinDistPos, 0.001);                               // get normal in said empty space
    float facingOff = dot(normalOff, -marchDir);
    float3 flatNormalOff = normalize(normalOff - facingOff * normalOff);                      

    float2 uvScale = float2(1.0 / 64.0, 1.0 / 70.0);
    float4 uvWorldScale = mul(ubo.inverseView, mul(ubo.inverseProj, float4(uvScale, 0.0, 0.0)));
    uvWorldScale /= uvWorldScale.w;

    if (marchResult.hit > 0.0) {

        // // pinch the outside inwards
        // if (marchResult.lastMinDist < (1.0/10.0) * (1.0 + marchResult.lastMinDistDepth)) {
        // }

        // pinch the inside outwards
        float facing = dot(normal, -marchDir);
        float3 flatNormal = normal - (facing * -marchDir);
        float3 marchOrigin2 = marchPos + ((1.0 / 32.0) * flatNormal * (marchResult.t));
        MarchResult marchResult2 = vertMarch(marchOrigin2, marchDir, marchLimit, true);
        if (marchResult2.hit < 0.0) {
            float3 marchPos2 = marchOrigin2 + (marchDir * marchResult2.t);
            marchPos = marchPos2 - (flatNormal * marchResult2.minDist * 1.0);
        }


        //marchPos -= (normalOff * (marchResult.lastMinDist / marchResult.lastMinDistDepth) * marchResult.t);

        normal = getNormal(marchPos, 0.001);
        

    
    } else {

        //pinch the outside inwards
        // if (marchResult.lastMinDist < (1.0/10.0) * (1.0 + marchResult.lastMinDistDepth)) {
        //     float3 minDistPos = cameraPos.xyz + (marchDir * marchResult.lastMinDistDepth);

        // }

        //marchPos -= (flatNormalOff * (marchResult.lastMinDist / marchResult.lastMinDistDepth) * marchResult.t);

        //normal = float3(0.0, 0.0, 0.0);
    }


    // output result
    output.depth = marchResult.t;
    output.pos = mul(ubo.proj, mul(ubo.view, float4(marchPos, 1.0)));
    output.normal = float4(normal, 1.0);
    output.fragPos = float4(marchPos, 1.0);
    output.color = input.inColor;
    //output.color.r = persp / 10.0;

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
    float3 L = normalize(lightPos.xyz - fragPos);
    float3 diffuse = max(dot(N, L), 0.0) * lightColour.rgb;
    //float3 diffuse = 1.0;

    // Specular
    const float specularStrength = 0.8 * saturate(50 - input.depth);
    float3 R = reflect(-L, N);
    const float shininess = 64.0;
    float spec = pow(max(dot(R, V), 0.0), shininess);
    float specBack = pow(max(dot(R, -V), 0.0), shininess);
    float3 specular = specularStrength * spec * lightColour.rgb;
    float3 specularBack = specularStrength * specBack * lightColour.rgb;

    // float A = 50.0 - input.depth;
    //float A = saturate(50 - input.depth);
    float A = 1.0;

    float3 glow = 0.3 + 0.7 * saturate(viewAngle) * saturate(50 - input.depth);
    //float3 glow = 0;

    float4 result = float4((ambient + diffuse + specular + glow), A) * input.color;
    return result;
}
