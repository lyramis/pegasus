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

[shader("vertex")]
VSOutput vertMain(VSInput input) {
    VSOutput output;

    float4x4 modelView = mul(ubo.view, ubo.model);
    float4 pos4 = float4(input.inPosition, 1.0);

    output.pos = mul(ubo.projection, mul(modelView, pos4));
    output.fragPos = mul(ubo.model, pos4).xyz;
    output.normal = mul(float3x3(transpose(ubo.inverseModel)), normalize(input.inPosition)); // Sphere-specific shortcut using verts as normals;
    output.color = input.inColor;
    output.viewPos = cameraPos.xyz;

    output.pointSize = 20.0f;

    return output;
}

[shader("fragment")]
float4 fragMain(FSInput input) : SV_TARGET {

    // Ambient
    const float ambientStrength = 0.1;
    float3 ambient = ambientStrength * lightColour.rgb;

    // Diffuse
    float3 N = normalize(input.normal);
    float3 L = normalize(lightPos.xyz - input.fragPos);
    float diff = max(dot(N, L), 0.0);
    float3 diffuse = diff * lightColour.rgb;

    // Specular
    const float specularStrength = 0.5;
    float3 V = normalize(input.viewPos - input.fragPos);
    float3 R = reflect(-L, N);
    const float shininess = 64.0;
    float spec = pow(max(dot(R, V), 0.0), shininess);
    float3 specular = specularStrength * spec * lightColour.rgb;

    //float A = 1.0 - (pow(1.0 - dot(N, V), 3.0));
    //float A = 1.0 - pow(max(length(input.fragPos), 0.0), 2.0); 

    float3 result = (ambient + diffuse) * input.color;
    return float4(result, 1.0f);
}

