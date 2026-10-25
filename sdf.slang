struct VSInput {
    float3 inPosition;
    float3 inColor;
};

struct UniformBuffer {
    float4x4 model;
    float4x4 inverseModel;
    float4x4 view;
    float4x4 projection;
    float2 iResolution;
    float iTime;
};
[[vk::binding(0, 0)]]
ConstantBuffer<UniformBuffer> ubo;

[[vk::push_constant]]
cbuffer PushConstants
{
    float4 cameraPos;
    float4 lightPos;
    float4 lightColour;
};

struct VSOutput
{
    float4 pos : SV_Position;
    float pointSize : SV_PointSize;
    float3 fragPos;
    float3 color;
    float3 normal;
    float3 viewPos;
};

struct FSInput
{
    float4 pos : SV_Position;
    float3 fragPos;
    float3 color;
    float3 normal;
    float3 viewPos;
    float2 pointCoord : SV_PointCoord;
};

// Signed distance function for a sphere
float sdSphere(float3 p, float r)
{
    return length(p) - r;
}

// Scene distance function — just one sphere for now
float map(float3 p)
{
    return sdSphere(p, 1.0);
}

// Estimate the surface normal via the gradient of the SDF
float3 getNormal(float3 p)
{
    float2 e = float2(0.001, 0.0);
    return normalize(float3(
        map(p + e.xyy) - map(p - e.xyy),
        map(p + e.yxy) - map(p - e.yxy),
        map(p + e.yyx) - map(p - e.yyx)
    ));
}

// Sphere-tracing ray march loop
float rayMarch(float3 ro, float3 rd)
{
    float t = 0.0;
    for (int i = 0; i < 50; i += 2)
    {
        float3 p = ro + rd * t;
        float d = map(p);
        t += d;
        if (d < 0.001) break;
        if (t > 100.0) break;
    }
    return t;
}

[shader("vertex")]
VSOutput vertMain(uint vertexID : SV_VertexID)
{
    VSOutput output;

    // Generate a triangle that covers the whole screen (and then some)
    // using just the vertex index — no vertex buffer needed.
    //   vertexID 0 -> (-1, -1)
    //   vertexID 1 -> ( 3, -1)
    //   vertexID 2 -> (-1,  3)
    float2 pos = float2(
        (vertexID == 2) ? 3.0 : -1.0,
        (vertexID == 1) ? 3.0 : -1.0
    );

    output.pos = float4(pos, 0.0, 1.0);
    return output;
}

[shader("fragment")]
float4 fragMain(FSInput input) : SV_Target
{
    float2 fragCoord = input.pos.xy;
    float2 uv = (fragCoord - 0.5 * ubo.iResolution) / ubo.iResolution.y;

    float3 ro = cameraPos.xyz;     // camera position
    float3 rd = normalize(float3(uv, 1.0)); // ray direction

    float t = rayMarch(ro, rd);

    float3 col = float3(0.0, 0.0, 0.0);

    if (t < 100.0)
    {
        float3 p = ro + rd * t;
        float3 n = getNormal(p);
        float3 lightDir = normalize(lightPos.xyz - input.fragPos);
        float diff = max(dot(n, lightDir), 0.0);
        col = float3(1.0, 1.0, 1.0) * diff;
    }

    return float4(col, 1.0);
}