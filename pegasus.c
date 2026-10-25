#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>

#define FAST_OBJ_IMPLEMENTATION
#include "fast_obj.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <vulkan/vulkan.h>

#if defined(USE_SDL3)
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#else
#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>
#endif

#define APP_NAME "Pegasus"
#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 600
#define MAX_FRAMES_IN_FLIGHT 2
#define PI 3.14159265358979323846

#define VK_CHECK(x) \
    do { \
        VkResult err__ = (x); \
        if (err__ != VK_SUCCESS) { \
            fprintf(stderr, "Vulkan error %d at %s:%d\n", err__, __FILE__, __LINE__); \
            exit(1); \
        } \
    } while (0)

/*-------------------------------------- STRUCTS -----------------------------------------*/

typedef union {
    struct { float x, y; };
    struct { float r, g; };
    float v[2];
} vec2;

typedef union {
    struct { float x, y, z; };
    struct { float r, g, b; };
    float v[3];
} vec3;

typedef union {
    struct { float x, y, z, w; };
    struct { float r, g, b, a; };
    float v[4];
} vec4;

typedef float mat2[2][2];
typedef float mat3[3][3];
typedef float mat4[4][4];

typedef struct Transform {
    vec3 position;
    vec3 rotation;
    vec3 scale;
} Transform;

typedef struct Vertex3D {
    vec3 position;
    vec4 colour;
} Vertex3D;

typedef struct Mesh3D {
    Vertex3D *pVertices;
    uint32_t vertexCount;
    uint32_t *pIndices;
    uint32_t indexCount;
} Mesh3D;

typedef struct Model3D {
    Transform transform;
    Mesh3D mesh;
} Model3D;

typedef struct Vertex4D {
    vec4 position;
    vec4 colour;
} Vertex4D;

typedef struct Mesh4D {
    Vertex4D *pVertices;
    uint32_t vertexCount;
    uint32_t *pIndices;
    uint32_t indexCount;
} Mesh4D;

typedef struct Model4D {
    Transform transform;
    Mesh4D mesh;
} Model4D;

typedef struct Camera {
    Transform transform;
    mat4 projection;
    mat4 inverseProjection;
} Camera;

typedef struct Light {
    vec3 position;
    vec4 colour;
} Light;

// typedef struct Primitive {
//     uint8_t type;
//     uint8_t operation;
//     float smoothing;
//     Transform transform;
// } Primitive;

//this is passed directly into the ubo, which has vec4 padding. (eventually compress to bitflags)
typedef struct PrimitiveUBO {
    vec3 position;
    uint8_t type;
    uint8_t operation;
    uint16_t _pad;
    vec3 rotation;
    float smoothing;
    vec3 scale;
    float __pad;
} PrimitiveUBO;

typedef struct UniformBufferObject {
    mat4 model;
    mat4 inverseModel;
    mat4 view;
    mat4 inverseView;
    mat4 projection;
    mat4 inverseProjection;
    vec2 iResolution;
    float iTime;
    uint32_t primitiveCount;
    PrimitiveUBO primitives[128];
} UniformBufferObject;

typedef struct PushConstants {
    vec4 cameraPosition;
    vec4 lightPosition; // is vec4 instead of vec3 because needs to be mult of 4
    vec4 lightColour;
} PushConstants;  

typedef struct {
    SDL_Window* window;

    VkInstance instance;
    VkSurfaceKHR surface;

    VkPhysicalDevice physicalDevice;
    VkDevice device;
    uint32_t graphicsFamily;
    uint32_t presentFamily;
    VkQueue graphicsQueue;
    VkQueue presentQueue;

    VkSwapchainKHR swapchain;
    VkFormat swapchainFormat;
    VkExtent2D swapchainExtent;
    uint32_t swapchainImageCount;
    VkImage* swapchainImages;
    VkImageView* swapchainImageViews;

    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorSet descriptorSets[MAX_FRAMES_IN_FLIGHT];
    VkDescriptorPool descriptorPool;

    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;

    uint32_t vertexBufferMaxSize; // TODO: look into this.
    VkDeviceMemory deviceVertexMemory[MAX_FRAMES_IN_FLIGHT];
    VkBuffer vertexBuffers[MAX_FRAMES_IN_FLIGHT];
    void *pVertexBuffersMapped[MAX_FRAMES_IN_FLIGHT];

    VkDeviceMemory deviceIndexMemory;
    void *pIndexBufferMapped;
    VkBuffer indexBuffer;

    VkBuffer uniformBuffers[MAX_FRAMES_IN_FLIGHT];
    VkDeviceMemory uniformBuffersMemory[MAX_FRAMES_IN_FLIGHT];
    void *pUniformBuffersMapped[MAX_FRAMES_IN_FLIGHT];

    VkCommandPool commandPool;
    VkCommandBuffer* pCommandBuffers;

    VkSemaphore imageAvailableSemaphores[MAX_FRAMES_IN_FLIGHT];
    VkSemaphore *pRenderFinishedSemaphores;
    VkFence inFlightFences[MAX_FRAMES_IN_FLIGHT];

    VkImage depthImage;
    VkDeviceMemory depthImageMemory;
    VkImageView depthImageView;
    VkFormat depthFormat;

    uint32_t currentFrameIndex;
    bool framebufferResized;

    uint64_t lastFrameTime;
    float deltaTime;
    float fps;
    float timer;
    bool isFirstFrame;

    // Game stuff

    Camera camera;
    Model3D model;
    Light light;
    PrimitiveUBO *pPrimitives;
    uint32_t primitiveCount;
    bool pause;

} App;


/* ----------------------------------------------- HELPERS --------------------------------------------*/

static char* readFile(const char* filename, size_t* outSize) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buffer = malloc((size_t)size);
    if (fread(buffer, 1, (size_t)size, f) != (size_t)size) {
        fprintf(stderr, "Failed to read file: %s\n", filename);
        exit(1);
    }
    fclose(f);
    *outSize = (size_t)size;
    return buffer;
}

static VkVertexInputBindingDescription getVertex3DBindingDescription(void) {
    VkVertexInputBindingDescription bindingDescription = {0};
    bindingDescription.binding = 0;
    bindingDescription.stride = sizeof(Vertex3D);
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    return bindingDescription;
}

static void getVertex3DAttributeDescriptions(VkVertexInputAttributeDescription* out) {
    out[0] = (VkVertexInputAttributeDescription){
        .binding = 0, .location = 0,
        .format = VK_FORMAT_R32G32B32_SFLOAT,
        .offset = offsetof(Vertex3D, position)
    };
    out[1] = (VkVertexInputAttributeDescription){
        .binding = 0, .location = 1,
        .format = VK_FORMAT_R32G32B32_SFLOAT,
        .offset = offsetof(Vertex3D, colour)
    };
}

static VkVertexInputBindingDescription getVertex4DBindingDescription(void) {
    VkVertexInputBindingDescription bindingDescription = {0};
    bindingDescription.binding = 0;
    bindingDescription.stride = sizeof(Vertex4D);
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    return bindingDescription;
}

static void getVertex4DAttributeDescriptions(VkVertexInputAttributeDescription* out) {
    out[0] = (VkVertexInputAttributeDescription){
        .binding = 0, .location = 0,
        .format = VK_FORMAT_R32G32B32A32_SFLOAT,
        .offset = offsetof(Vertex4D, position)
    };
    out[1] = (VkVertexInputAttributeDescription){
        .binding = 0, .location = 1,
        .format = VK_FORMAT_R32G32B32A32_SFLOAT,
        .offset = offsetof(Vertex4D, colour)
    };
}

// VECTORS

//vec2
void vec2_set(vec2 *out, float x, float y){ out->x = x; out->y = y; }

//vec3
void vec3_from_vec2(vec3 *out, const vec2 in, float z){ out->x = in.x; out->y = in.y; out->z = z;}
void vec3_set(vec3 *out, float x, float y, float z){ out->x = x; out->y = y; out->z = z; }
void vec3_add(vec3 *out, const vec3 a, const vec3 b){ out->x = a.x + b.x; out->y = a.y + b.y; out->z = a.z + b.z; }
void vec3_sub(vec3 *out, const vec3 a, const vec3 b){ out->x = a.x - b.x; out->y = a.y - b.y; out->z = a.z - b.z; }
void vec3_scale(vec3 *out, const vec3 in, const float factor){ out->x = in.x * factor; out->y = in.y * factor; out->z = in.z * factor;}
void vec3_dot(float *out, const vec3 a, const vec3 b){ *out = a.x * b.x + a.y * b.y + a.z * b.z; }
void vec3_mag(float*out, const vec3 in){ *out = sqrtf(in.x * in.x + in.y * in.y + in.z * in.z); }
void vec3_cross(vec3 *out, const vec3 a, const vec3 b){ out->x = a.y * b.z - a.z * b.y; out->y = a.z * b.x - a.x * b.z; out->z = a.x * b.y - a.y * b.x; }
void vec3_from_vec4(vec3 *out, const vec4 in){ out->x = in.x; out->y = in.y; out->z = in.z;}

//vec4
void vec4_from_vec3(vec4 *out, const vec3 in, float w){ out->x = in.x; out->y = in.y; out->z = in.z; out->w = w;}
void vec4_set(vec4 *out, float x, float y, float z, float w){ out->x = x; out->y = y; out->z = z; out->w = w; }

// MATRICES

void mat4_mul_vec4(vec4 *out, const mat4 m, const vec4 v){
    out->x = m[0][0]*v.x + m[1][0]*v.y + m[2][0]*v.z + m[3][0]*v.w;
    out->y = m[0][1]*v.x + m[1][1]*v.y + m[2][1]*v.z + m[3][1]*v.w;
    out->z = m[0][2]*v.x + m[1][2]*v.y + m[2][2]*v.z + m[3][2]*v.w;
    out->w = m[0][3]*v.x + m[1][3]*v.y + m[2][3]*v.z + m[3][3]*v.w;
}

void mat4_row(vec4 *row, mat4 in, uint8_t index){
    row->x = in[0][index];
    row->y = in[1][index];
    row->z = in[2][index];
    row->w = in[3][index];
}

void mat4_col(vec4 *col, mat4 in, uint8_t index){
    col->x = in[index][0];
    col->y = in[index][1];
    col->z = in[index][2];
    col->w = in[index][3];
}

void mat4_identity(mat4 out) {
    out[0][0] = 1.0f; out[1][0] = 0.0f; out[2][0] = 0.0f; out[3][0] = 0.0f;
    out[0][1] = 0.0f; out[1][1] = 1.0f; out[2][1] = 0.0f; out[3][1] = 0.0f;
    out[0][2] = 0.0f; out[1][2] = 0.0f; out[2][2] = 1.0f; out[3][2] = 0.0f;
    out[0][3] = 0.0f; out[1][3] = 0.0f; out[2][3] = 0.0f; out[3][3] = 1.0f;
}

void mat4_translation(mat4 out, float x, float y, float z) {
    out[0][0] = 1.0f; out[1][0] = 0.0f; out[2][0] = 0.0f; out[3][0] = x;
    out[0][1] = 0.0f; out[1][1] = 1.0f; out[2][1] = 0.0f; out[3][1] = y;
    out[0][2] = 0.0f; out[1][2] = 0.0f; out[2][2] = 1.0f; out[3][2] = z;
    out[0][3] = 0.0f; out[1][3] = 0.0f; out[2][3] = 0.0f; out[3][3] = 1.0f;
}

void mat4_scale(mat4 out, float x, float y, float z) {
    out[0][0] = x;    out[1][0] = 0.0f; out[2][0] = 0.0f; out[3][0] = 0.0f;
    out[0][1] = 0.0f; out[1][1] = y;    out[2][1] = 0.0f; out[3][1] = 0.0f;
    out[0][2] = 0.0f; out[1][2] = 0.0f; out[2][2] = z;    out[3][2] = 0.0f;
    out[0][3] = 0.0f; out[1][3] = 0.0f; out[2][3] = 0.0f; out[3][3] = 1.0f;
}

// R = Rz * Ry * Rx  (applied to vector: v' = Rz*(Ry*(Rx*v)))
// i.e. rotate around X first, then Y, then Z.
void mat4_rotation(mat4 out, float x, float y, float z) {
    float cx = cosf(x), sx = sinf(x);
    float cy = cosf(y), sy = sinf(y);
    float cz = cosf(z), sz = sinf(z);

    out[0][0] = cy * cz; out[1][0] = sx * sy * cz - cx * sz; out[2][0] = cx * sy * cz + sx * sz; out[3][0] = 0.0f;
    out[0][1] = cy * sz; out[1][1] = sx * sy * sz + cx * cz; out[2][1] = cx * sy * sz - sx * cz; out[3][1] = 0.0f;
    out[0][2] = -sy;     out[1][2] = sx * cy;                out[2][2] = cx * cy;                out[3][2] = 0.0f;
    out[0][3] = 0.0f;    out[1][3] = 0.0f;                   out[2][3] = 0.0f;                   out[3][3] = 1.0f;
}

void mat4_mul(mat4 out, mat4 a, mat4 b) {
    float result[4][4];

    for (int col = 0; col < 4; col++) {
        for (int row = 0; row < 4; row++) {
            float sum = 0.0f;
            for (int k = 0; k < 4; k++) {
                sum += a[k][row] * b[col][k];
            }
            result[col][row] = sum;
        }
    }

    for (int col = 0; col < 4; col++) {
        for (int row = 0; row < 4; row++) {
            out[col][row] = result[col][row];
        }
    }
}

// full mat4 inversion
int mat4_inverse(mat4 out, mat4 in) {
    float inv[4][4];

    inv[0][0] =  in[1][1] * in[2][2] * in[3][3] -
                 in[1][1] * in[2][3] * in[3][2] -
                 in[2][1] * in[1][2] * in[3][3] +
                 in[2][1] * in[1][3] * in[3][2] +
                 in[3][1] * in[1][2] * in[2][3] -
                 in[3][1] * in[1][3] * in[2][2];

    inv[1][0] = -in[1][0] * in[2][2] * in[3][3] +
                 in[1][0] * in[2][3] * in[3][2] +
                 in[2][0] * in[1][2] * in[3][3] -
                 in[2][0] * in[1][3] * in[3][2] -
                 in[3][0] * in[1][2] * in[2][3] +
                 in[3][0] * in[1][3] * in[2][2];

    inv[2][0] =  in[1][0] * in[2][1] * in[3][3] -
                 in[1][0] * in[2][3] * in[3][1] -
                 in[2][0] * in[1][1] * in[3][3] +
                 in[2][0] * in[1][3] * in[3][1] +
                 in[3][0] * in[1][1] * in[2][3] -
                 in[3][0] * in[1][3] * in[2][1];

    inv[3][0] = -in[1][0] * in[2][1] * in[3][2] +
                 in[1][0] * in[2][2] * in[3][1] +
                 in[2][0] * in[1][1] * in[3][2] -
                 in[2][0] * in[1][2] * in[3][1] -
                 in[3][0] * in[1][1] * in[2][2] +
                 in[3][0] * in[1][2] * in[2][1];

    inv[0][1] = -in[0][1] * in[2][2] * in[3][3] +
                 in[0][1] * in[2][3] * in[3][2] +
                 in[2][1] * in[0][2] * in[3][3] -
                 in[2][1] * in[0][3] * in[3][2] -
                 in[3][1] * in[0][2] * in[2][3] +
                 in[3][1] * in[0][3] * in[2][2];

    inv[1][1] =  in[0][0] * in[2][2] * in[3][3] -
                 in[0][0] * in[2][3] * in[3][2] -
                 in[2][0] * in[0][2] * in[3][3] +
                 in[2][0] * in[0][3] * in[3][2] +
                 in[3][0] * in[0][2] * in[2][3] -
                 in[3][0] * in[0][3] * in[2][2];

    inv[2][1] = -in[0][0] * in[2][1] * in[3][3] +
                 in[0][0] * in[2][3] * in[3][1] +
                 in[2][0] * in[0][1] * in[3][3] -
                 in[2][0] * in[0][3] * in[3][1] -
                 in[3][0] * in[0][1] * in[2][3] +
                 in[3][0] * in[0][3] * in[2][1];

    inv[3][1] =  in[0][0] * in[2][1] * in[3][2] -
                 in[0][0] * in[2][2] * in[3][1] -
                 in[2][0] * in[0][1] * in[3][2] +
                 in[2][0] * in[0][2] * in[3][1] +
                 in[3][0] * in[0][1] * in[2][2] -
                 in[3][0] * in[0][2] * in[2][1];

    inv[0][2] =  in[0][1] * in[1][2] * in[3][3] -
                 in[0][1] * in[1][3] * in[3][2] -
                 in[1][1] * in[0][2] * in[3][3] +
                 in[1][1] * in[0][3] * in[3][2] +
                 in[3][1] * in[0][2] * in[1][3] -
                 in[3][1] * in[0][3] * in[1][2];

    inv[1][2] = -in[0][0] * in[1][2] * in[3][3] +
                 in[0][0] * in[1][3] * in[3][2] +
                 in[1][0] * in[0][2] * in[3][3] -
                 in[1][0] * in[0][3] * in[3][2] -
                 in[3][0] * in[0][2] * in[1][3] +
                 in[3][0] * in[0][3] * in[1][2];

    inv[2][2] =  in[0][0] * in[1][1] * in[3][3] -
                 in[0][0] * in[1][3] * in[3][1] -
                 in[1][0] * in[0][1] * in[3][3] +
                 in[1][0] * in[0][3] * in[3][1] +
                 in[3][0] * in[0][1] * in[1][3] -
                 in[3][0] * in[0][3] * in[1][1];

    inv[3][2] = -in[0][0] * in[1][1] * in[3][2] +
                 in[0][0] * in[1][2] * in[3][1] +
                 in[1][0] * in[0][1] * in[3][2] -
                 in[1][0] * in[0][2] * in[3][1] -
                 in[3][0] * in[0][1] * in[1][2] +
                 in[3][0] * in[0][2] * in[1][1];

    inv[0][3] = -in[0][1] * in[1][2] * in[2][3] +
                 in[0][1] * in[1][3] * in[2][2] +
                 in[1][1] * in[0][2] * in[2][3] -
                 in[1][1] * in[0][3] * in[2][2] -
                 in[2][1] * in[0][2] * in[1][3] +
                 in[2][1] * in[0][3] * in[1][2];

    inv[1][3] =  in[0][0] * in[1][2] * in[2][3] -
                 in[0][0] * in[1][3] * in[2][2] -
                 in[1][0] * in[0][2] * in[2][3] +
                 in[1][0] * in[0][3] * in[2][2] +
                 in[2][0] * in[0][2] * in[1][3] -
                 in[2][0] * in[0][3] * in[1][2];

    inv[2][3] = -in[0][0] * in[1][1] * in[2][3] +
                 in[0][0] * in[1][3] * in[2][1] +
                 in[1][0] * in[0][1] * in[2][3] -
                 in[1][0] * in[0][3] * in[2][1] -
                 in[2][0] * in[0][1] * in[1][3] +
                 in[2][0] * in[0][3] * in[1][1];

    inv[3][3] =  in[0][0] * in[1][1] * in[2][2] -
                 in[0][0] * in[1][2] * in[2][1] -
                 in[1][0] * in[0][1] * in[2][2] +
                 in[1][0] * in[0][2] * in[2][1] +
                 in[2][0] * in[0][1] * in[1][2] -
                 in[2][0] * in[0][2] * in[1][1];

    float det = in[0][0] * inv[0][0] + in[0][1] * inv[1][0] +
                in[0][2] * inv[2][0] + in[0][3] * inv[3][0];

    if (fabsf(det) < 1e-8f)
        return 0; /* singular matrix, no inverse */

    float inv_det = 1.0f / det;

    for (int col = 0; col < 4; col++)
        for (int row = 0; row < 4; row++)
            out[col][row] = inv[col][row] * inv_det;

    return 1;
}

// fast inverse for affine matrices (no shear/skew)
int mat4_inverse_affine(mat4 out, mat4 in) {
    /* Upper-left 3x3 block (column-major) */
    float a00 = in[0][0], a01 = in[1][0], a02 = in[2][0];
    float a10 = in[0][1], a11 = in[1][1], a12 = in[2][1];
    float a20 = in[0][2], a21 = in[1][2], a22 = in[2][2];

    /* Translation column */
    float tx = in[3][0], ty = in[3][1], tz = in[3][2];

    /* Cofactors of the 3x3 block */
    float c00 =  (a11 * a22 - a12 * a21);
    float c01 = -(a10 * a22 - a12 * a20);
    float c02 =  (a10 * a21 - a11 * a20);

    float c10 = -(a01 * a22 - a02 * a21);
    float c11 =  (a00 * a22 - a02 * a20);
    float c12 = -(a00 * a21 - a01 * a20);

    float c20 =  (a01 * a12 - a02 * a11);
    float c21 = -(a00 * a12 - a02 * a10);
    float c22 =  (a00 * a11 - a01 * a10);

    /* det via cofactor expansion along the first row */
    float det = a00 * c00 + a01 * c01 + a02 * c02;

    if (fabsf(det) < 1e-8f) {
        return 0; /* singular, no inverse */
    }

    float inv_det = 1.0f / det;

    /* inv(R) = adj(R) / det = transpose(cofactor matrix) / det */
    float r00 = c00 * inv_det, r01 = c10 * inv_det, r02 = c20 * inv_det;
    float r10 = c01 * inv_det, r11 = c11 * inv_det, r12 = c21 * inv_det;
    float r20 = c02 * inv_det, r21 = c12 * inv_det, r22 = c22 * inv_det;

    /* new translation = -inv(R) * t */
    float ntx = -(r00 * tx + r01 * ty + r02 * tz);
    float nty = -(r10 * tx + r11 * ty + r12 * tz);
    float ntz = -(r20 * tx + r21 * ty + r22 * tz);

    out[0][0] = r00; out[1][0] = r01; out[2][0] = r02; out[3][0] = ntx;
    out[0][1] = r10; out[1][1] = r11; out[2][1] = r12; out[3][1] = nty;
    out[0][2] = r20; out[1][2] = r21; out[2][2] = r22; out[3][2] = ntz;
    out[0][3] = 0.0f; out[1][3] = 0.0f; out[2][3] = 0.0f; out[3][3] = 1.0f;

    return 1;
}

void mat4_TRS(mat4 out, const vec3 position, const vec3 rotation, const vec3 scale ){
    mat4 modelTrans, modelRot, modelScale, modelRS;
    mat4_translation(modelTrans, position.x, position.y, position.z);
    mat4_rotation(modelRot, rotation.x, rotation.y, rotation.z);
    mat4_scale(modelScale, scale.x, scale.y, scale.z);
    mat4_mul(modelRS, modelScale, modelRot);
    mat4_mul(out, modelTrans, modelRS);
}

void mat4_lookat(mat4 out, const vec3 eye, const vec3 center, const vec3 up) {
    float fx = center.x - eye.x;
    float fy = center.y - eye.y;
    float fz = center.z - eye.z;
    float flen = sqrtf(fx * fx + fy * fy + fz * fz);
    if (flen > 0.0f) { fx /= flen; fy /= flen; fz /= flen; }

    /* s = f x up */
    float sx = fy * up.z - fz * up.y;
    float sy = fz * up.x - fx * up.z;
    float sz = fx * up.y - fy * up.x;
    float slen = sqrtf(sx * sx + sy * sy + sz * sz);
    if (slen > 0.0f) { sx /= slen; sy /= slen; sz /= slen; }

    /* u = s x f */
    float ux = sy * fz - sz * fy;
    float uy = sz * fx - sx * fz;
    float uz = sx * fy - sy * fx;

    out[0][0] = sx;  out[1][0] = sy;  out[2][0] = sz;  out[3][0] = -(sx * eye.x + sy * eye.y + sz * eye.z);
    out[0][1] = ux;  out[1][1] = uy;  out[2][1] = uz;  out[3][1] = -(ux * eye.x + uy * eye.y + uz * eye.z);
    out[0][2] = -fx; out[1][2] = -fy; out[2][2] = -fz; out[3][2] =  (fx * eye.x + fy * eye.y + fz * eye.z);
    out[0][3] = 0.0f; out[1][3] = 0.0f; out[2][3] = 0.0f; out[3][3] = 1.0f;
}

// uses vulkan 0.0-1.0 depth, but openGL's y-convention, we'll flip the viewport as per the docs
void mat4_perspective(mat4 out, float fovy_rad, float aspect, float znear, float zfar) {
    float f = 1.0f / tanf(fovy_rad * 0.5f);

    out[0][0] = f / aspect; out[1][0] = 0.0f; out[2][0] = 0.0f;                    out[3][0] = 0.0f;
    out[0][1] = 0.0f;       out[1][1] = f;    out[2][1] = 0.0f;                    out[3][1] = 0.0f;
    out[0][2] = 0.0f;       out[1][2] = 0.0f; out[2][2] = (zfar) / (znear - zfar); out[3][2] = (zfar * znear) / (znear - zfar);
    out[0][3] = 0.0f;       out[1][3] = 0.0f; out[2][3] = -1.0f;                   out[3][3] = 0.0f;
}

// uses vulkan 0.0-1.0 depth, but openGL's y-convention, we'll flip the viewport as per the docs
void mat4_orthographic(mat4 out, float left, float right, float bottom, float top, float znear, float zfar) {
    float rl = right - left;
    float tb = top - bottom;
    float fn = znear - zfar;

    out[0][0] = 2.0f / rl; out[1][0] = 0.0f;      out[2][0] = 0.0f;         out[3][0] = -(right + left) / rl;
    out[0][1] = 0.0f;      out[1][1] = 2.0f / tb; out[2][1] = 0.0f;         out[3][1] = -(top + bottom) / tb;
    out[0][2] = 0.0f;      out[1][2] = 0.0f;      out[2][2] = 1.0f / fn;    out[3][2] = znear / fn;
    out[0][3] = 0.0f;      out[1][3] = 0.0f;      out[2][3] = 0.0f;         out[3][3] = 1.0f;
}

static void parseOBJ(Mesh3D *pMesh, const char* path, vec4 *pColor) {
    fastObjMesh* mesh = fast_obj_read(path);

    if (!mesh) {
        fprintf(stderr, "Failed to load OBJ: %s\n", path);
        exit(1);
    }

    // Worst case: every face index becomes a unique vertex
    size_t maxVerts = mesh->index_count;
    pMesh->pVertices = malloc(maxVerts * sizeof(Vertex3D));
    pMesh->pIndices = malloc(maxVerts * sizeof(uint32_t));
    pMesh->vertexCount = 0;
    pMesh->indexCount = 0;

    for (unsigned int i = 0; i < mesh->index_count; i++) {
        fastObjIndex idx = mesh->indices[i];

        Vertex3D v = {0};

        v.position.x = mesh->positions[3 * idx.p + 0];
        v.position.y = mesh->positions[3 * idx.p + 1];
        v.position.z = mesh->positions[3 * idx.p + 2];

        //just default to white;
        if (pColor != NULL){
            v.colour.r = pColor->r;
            v.colour.g = pColor->g;
            v.colour.b = pColor->b;
        }else{
            v.colour.r = 1.0f;
            v.colour.g = 1.0f;
            v.colour.b = 1.0f;
        }
        
        // Simple linear search for deduplication (use a hash map for large meshes)
        int found = -1;
        for (size_t j = 0; j < pMesh->vertexCount; j++) {
            if (memcmp(&pMesh->pVertices[j], &v, sizeof(Vertex3D)) == 0) {
                found = (int)j;
                break;
            }
        }

        if (found == -1) {
            pMesh->pVertices [pMesh->vertexCount] = v;
            found = (int)pMesh->vertexCount;
            pMesh->vertexCount++;
        }

        pMesh->pIndices[ pMesh->indexCount++ ] = (uint32_t)found;
    }

    printf("Parsed OBJ file \"%s\" with %d vertices and %d indices.\n", path, pMesh->vertexCount, pMesh->indexCount);

    fast_obj_destroy(mesh);
}

//this func subdivides each triangle of a mesh into 2, but does not preserve welding.
static void subdivideMesh(Mesh3D *pIn) {
    int tri_count = pIn->indexCount / 3;
 
    Mesh3D out;

    out.vertexCount = pIn->vertexCount + tri_count; /* +1 vert per tri */
    out.indexCount = pIn->indexCount * 2;        /* 2 tris per tri  */
    out.pVertices = (Vertex3D *)malloc(sizeof(Vertex3D) * out.vertexCount);
    out.pIndices = (uint32_t *)malloc(sizeof(uint32_t) * out.indexCount);
 
    /* copy existing verts, new verts get appended after them */
    memcpy(out.pVertices, pIn->pVertices, sizeof(Vertex3D) * pIn->vertexCount);
 
    int next_vert = pIn->vertexCount;
    int oi = 0; /* output index write cursor */
 
    for (int t = 0; t < tri_count; t++) {
        uint32_t i0 = pIn->pIndices[t * 3 + 0];
        uint32_t i1 = pIn->pIndices[t * 3 + 1];
        uint32_t i2 = pIn->pIndices[t * 3 + 2];
 
        vec3 a = pIn->pVertices[i0].position;
        vec3 b = pIn->pVertices[i1].position;
        vec3 c = pIn->pVertices[i2].position;
 
        /* squared edge lengths, avoids sqrt just to compare */
        float lab = (a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y) + (a.z-b.z)*(a.z-b.z);
        float lbc = (b.x-c.x)*(b.x-c.x) + (b.y-c.y)*(b.y-c.y) + (b.z-c.z)*(b.z-c.z);
        float lca = (c.x-a.x)*(c.x-a.x) + (c.y-a.y)*(c.y-a.y) + (c.z-a.z)*(c.z-a.z);
 
        uint32_t p0, p1, opp; /* split edge endpoints + opposite vertex */
        vec3 e0, e1;
 
        if (lab >= lbc && lab >= lca) {
            p0 = i0; p1 = i1; opp = i2; e0 = a; e1 = b;
        } else if (lbc >= lab && lbc >= lca) {
            p0 = i1; p1 = i2; opp = i0; e0 = b; e1 = c;
        } else {
            p0 = i2; p1 = i0; opp = i1; e0 = c; e1 = a;
        }
 
        vec3 mid = {(e0.x + e1.x) * 0.5f, (e0.y + e1.y) * 0.5f, (e0.z + e1.z) * 0.5f};
 
        uint32_t mi = next_vert++;
        out.pVertices[mi].position = mid;

        out.pVertices[mi].colour = pIn->pVertices[i0].colour;
 
        /* two new triangles sharing the midpoint and the opposite vertex */
        out.pIndices[oi++] = p0; out.pIndices[oi++] = mi;  out.pIndices[oi++] = opp;
        out.pIndices[oi++] = mi; out.pIndices[oi++] = p1;  out.pIndices[oi++] = opp;
    }
 
    *pIn = out;
}

/* -------------------------------------------- SURFACE ---------------------------------------- */

static bool initWindow(App* app) {
#if SDL_MAJOR_VERSION >= 3
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
        return false;
    }
    app->window = SDL_CreateWindow(APP_NAME, WINDOW_WIDTH, WINDOW_HEIGHT,
                                    SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
#else
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
        return false;
    }
    app->window = SDL_CreateWindow(APP_NAME,
                                    SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                    WINDOW_WIDTH, WINDOW_HEIGHT,
                                    SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
#endif
    if (!app->window) {
        fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
        return false;
    }
    return true;
}

static void createInstance(App* app) {
    uint32_t sdlExtensionCount = 0;
    const char** sdlExtensions;
    bool freeExtensions = false;

#if SDL_MAJOR_VERSION >= 3
    sdlExtensions = SDL_Vulkan_GetInstanceExtensions(&sdlExtCount);
    if (!sdlExtensions) {
        fprintf(stderr, "SDL_Vulkan_GetInstanceExtensions failed: %s\n", SDL_GetError());
        exit(1);
    }
#else
    if (!SDL_Vulkan_GetInstanceExtensions(app->window, &sdlExtensionCount, NULL)) {
        fprintf(stderr, "SDL_Vulkan_GetInstanceExtensions failed: %s\n", SDL_GetError());
        exit(1);
    }
    sdlExtensions = malloc(sizeof(const char*) * sdlExtensionCount);
    if (!SDL_Vulkan_GetInstanceExtensions(app->window, &sdlExtensionCount, sdlExtensions)) {
        fprintf(stderr, "SDL_Vulkan_GetInstanceExtensions failed: %s\n", SDL_GetError());
        exit(1);
    }
    freeExtensions = true;
#endif

    VkApplicationInfo appInfo = {0};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = APP_NAME;
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_4;

    VkInstanceCreateInfo createInfo = {0};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount = sdlExtensionCount;
    createInfo.ppEnabledExtensionNames = sdlExtensions;

    VK_CHECK(vkCreateInstance(&createInfo, NULL, &app->instance));

    if (freeExtensions) {
        free((void*)sdlExtensions);
    }
}

static void createSurface(App* app) {
#if SDL_MAJOR_VERSION >= 3
    if (!SDL_Vulkan_CreateSurface(app->window, app->instance, NULL, &app->surface)) {
        fprintf(stderr, "SDL_Vulkan_CreateSurface failed: %s\n", SDL_GetError());
        exit(1);
    }
#else
    if (!SDL_Vulkan_CreateSurface(app->window, app->instance, &app->surface)) {
        fprintf(stderr, "SDL_Vulkan_CreateSurface failed: %s\n", SDL_GetError());
        exit(1);
    }
#endif
}

/* ----------------------------------------------- DEVICES -------------------------------------------- */

static bool findQueueFamilies(App* app, VkPhysicalDevice device, uint32_t* graphicsFamily, uint32_t* presentFamily) {
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, NULL);
    VkQueueFamilyProperties* families = malloc(sizeof(VkQueueFamilyProperties) * count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, families);

    bool foundGraphics = false, foundPresent = false;
    for (uint32_t i = 0; i < count; i++) {
        if (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            *graphicsFamily = i;
            foundGraphics = true;
        }
        VkBool32 presentSupport = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, i, app->surface, &presentSupport);
        if (presentSupport) {
            *presentFamily = i;
            foundPresent = true;
        }
        if (foundGraphics && foundPresent) break;
    }
    free(families);
    return foundGraphics && foundPresent;
}

static bool checkSwapchainExtension(VkPhysicalDevice device) {
    uint32_t extCount = 0;
    vkEnumerateDeviceExtensionProperties(device, NULL, &extCount, NULL);
    VkExtensionProperties* exts = malloc(sizeof(VkExtensionProperties) * extCount);
    vkEnumerateDeviceExtensionProperties(device, NULL, &extCount, exts);

    bool found = false;
    for (uint32_t i = 0; i < extCount; i++) {
        if (strcmp(exts[i].extensionName, VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0) {
            found = true;
            break;
        }
    }
    free(exts);
    return found;
}

static void pickPhysicalDevice(App* app) {
    
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(app->instance, &deviceCount, NULL);

    if (deviceCount == 0) {
        fprintf(stderr, "No Vulkan-capable GPUs found\n");
        exit(1);
    }

    VkPhysicalDevice* devices = malloc(sizeof(VkPhysicalDevice) * deviceCount);
    vkEnumeratePhysicalDevices(app->instance, &deviceCount, devices);

    for (uint32_t i = 0; i < deviceCount; i++) {
        uint32_t g = 0, p = 0;
        if (findQueueFamilies(app, devices[i], &g, &p) &&
            checkSwapchainExtension(devices[i])) {
            uint32_t formatCount = 0, presentModeCount = 0;
            vkGetPhysicalDeviceSurfaceFormatsKHR(devices[i], app->surface, &formatCount, NULL);
            vkGetPhysicalDeviceSurfacePresentModesKHR(devices[i], app->surface, &presentModeCount, NULL);
            if (formatCount > 0 && presentModeCount > 0) {
                app->physicalDevice = devices[i];
                app->graphicsFamily = g;
                app->presentFamily = p;
                free(devices);
                return;
            }
        }
    }
    fprintf(stderr, "No suitable GPU found\n");
    exit(1);
}

static void createLogicalDevice(App* app) {
    float priority = 1.0f;
    VkDeviceQueueCreateInfo queueCreateInfos[2];
    uint32_t queueCreateCount = 0;

    VkDeviceQueueCreateInfo qci = {0};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = app->graphicsFamily;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;
    queueCreateInfos[queueCreateCount++] = qci;

    if (app->presentFamily != app->graphicsFamily) {
        VkDeviceQueueCreateInfo qci2 = {0};
        qci2.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qci2.queueFamilyIndex = app->presentFamily;
        qci2.queueCount = 1;
        qci2.pQueuePriorities = &priority;
        queueCreateInfos[queueCreateCount++] = qci2;
    }
 
    VkPhysicalDeviceVulkan13Features featuresVulkan13 = {0};
    featuresVulkan13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    featuresVulkan13.dynamicRendering = VK_TRUE;
    featuresVulkan13.synchronization2 = VK_TRUE;

    VkPhysicalDeviceFeatures2 features2 = {0};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &featuresVulkan13;

    const char* deviceExtensions[] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };

    VkDeviceCreateInfo createInfo = {0};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pQueueCreateInfos = queueCreateInfos;
    createInfo.queueCreateInfoCount = queueCreateCount;
    //createInfo.pEnabledFeatures = &featuresVulkan13;
    createInfo.pNext = &features2;
    createInfo.enabledExtensionCount = 1;
    createInfo.ppEnabledExtensionNames = deviceExtensions;

    VK_CHECK(vkCreateDevice(app->physicalDevice, &createInfo, NULL, &app->device));

    vkGetDeviceQueue(app->device, app->graphicsFamily, 0, &app->graphicsQueue);
    vkGetDeviceQueue(app->device, app->presentFamily, 0, &app->presentQueue);
}

/* ------------------------------------------------- SWAPCHAIN ---------------------------------------------- */

static VkSurfaceFormatKHR chooseSurfaceFormat(VkSurfaceFormatKHR* formats, uint32_t count) {
    for (uint32_t i = 0; i < count; i++) {
        if (formats[i].format == VK_FORMAT_B8G8R8A8_SRGB &&
            formats[i].colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return formats[i];
        }
    }
    return formats[0];
}

static VkPresentModeKHR choosePresentMode(VkPresentModeKHR* modes, uint32_t count) {
    for (uint32_t i = 0; i < count; i++) {
        if (modes[i] == VK_PRESENT_MODE_MAILBOX_KHR) return modes[i];
    }
    return VK_PRESENT_MODE_FIFO_KHR; /* always available */
}

static VkExtent2D chooseExtent(App* app, VkSurfaceCapabilitiesKHR* caps) {
    if (caps->currentExtent.width != UINT32_MAX) {
        return caps->currentExtent;
    }
    int w, h;
#if SDL_MAJOR_VERSION >= 3
    SDL_GetWindowSizeInPixels(app->window, &w, &h);
#else
    SDL_Vulkan_GetDrawableSize(app->window, &w, &h);
#endif
    VkExtent2D extent;
    extent.width = (uint32_t)w;
    extent.height = (uint32_t)h;
    if (extent.width < caps->minImageExtent.width) extent.width = caps->minImageExtent.width;
    if (extent.width > caps->maxImageExtent.width) extent.width = caps->maxImageExtent.width;
    if (extent.height < caps->minImageExtent.height) extent.height = caps->minImageExtent.height;
    if (extent.height > caps->maxImageExtent.height) extent.height = caps->maxImageExtent.height;
    return extent;
}

static void createImageView(App *app, VkImageView *pImageView, VkImage *pImage, VkFormat *pFormat, VkImageAspectFlags aspectFlags){
    VkImageViewCreateInfo createInfo = {0};
    createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    createInfo.image = *pImage;
    createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    createInfo.format = *pFormat;
    createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.subresourceRange.aspectMask = aspectFlags;
    createInfo.subresourceRange.baseMipLevel = 0;
    createInfo.subresourceRange.levelCount = 1;
    createInfo.subresourceRange.baseArrayLayer = 0;
    createInfo.subresourceRange.layerCount = 1;

    VK_CHECK(vkCreateImageView(app->device, &createInfo, NULL, pImageView));
}

static void createSwapchain(App* app) {
    VkSurfaceCapabilitiesKHR caps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(app->physicalDevice, app->surface, &caps);

    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(app->physicalDevice, app->surface, &formatCount, NULL);
    VkSurfaceFormatKHR* formats = malloc(sizeof(VkSurfaceFormatKHR) * formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(app->physicalDevice, app->surface, &formatCount, formats);

    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(app->physicalDevice, app->surface, &presentModeCount, NULL);
    VkPresentModeKHR* presentModes = malloc(sizeof(VkPresentModeKHR) * presentModeCount);
    vkGetPhysicalDeviceSurfacePresentModesKHR(app->physicalDevice, app->surface, &presentModeCount, presentModes);

    VkSurfaceFormatKHR surfaceFormat = chooseSurfaceFormat(formats, formatCount);
    VkPresentModeKHR presentMode = choosePresentMode(presentModes, presentModeCount);
    VkExtent2D extent = chooseExtent(app, &caps);

    uint32_t imageCount = caps.minImageCount + 1;
    if (caps.maxImageCount > 0 && imageCount > caps.maxImageCount) {
        imageCount = caps.maxImageCount;
    }

    VkSwapchainCreateInfoKHR createInfo = {0};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = app->surface;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    uint32_t queueFamilyIndices[] = { app->graphicsFamily, app->presentFamily };
    if (app->graphicsFamily != app->presentFamily) {
        createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        createInfo.queueFamilyIndexCount = 2;
        createInfo.pQueueFamilyIndices = queueFamilyIndices;
    } else {
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    createInfo.preTransform = caps.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE;
    createInfo.oldSwapchain = VK_NULL_HANDLE;

    VK_CHECK(vkCreateSwapchainKHR(app->device, &createInfo, NULL, &app->swapchain));

    vkGetSwapchainImagesKHR(app->device, app->swapchain, &app->swapchainImageCount, NULL);
    app->swapchainImages = malloc(sizeof(VkImage) * app->swapchainImageCount);
    vkGetSwapchainImagesKHR(app->device, app->swapchain, &app->swapchainImageCount, app->swapchainImages);

    app->swapchainFormat = surfaceFormat.format;
    app->swapchainExtent = extent;

    app->swapchainImageViews = malloc(sizeof(VkImageView) * app->swapchainImageCount);

    for (uint32_t i = 0; i < app->swapchainImageCount; i++) {
        createImageView(app, &app->swapchainImageViews[i], &app->swapchainImages[i], &app->swapchainFormat, VK_IMAGE_ASPECT_COLOR_BIT);
    }

    free(formats);
    free(presentModes);
}

/*--------------------------------------------- DESCRIPTORS ------------------------------------------------ */

static void createDescriptorSetLayout(App *app){

    VkDescriptorSetLayoutBinding descSetLayoutBinding = {0};
    descSetLayoutBinding.binding = 0;
    descSetLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descSetLayoutBinding.descriptorCount = 1;
    descSetLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutCreateInfo descSetLayoutInfo = {0};
    descSetLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descSetLayoutInfo.bindingCount = 1;
    descSetLayoutInfo.pBindings = &descSetLayoutBinding;

    vkCreateDescriptorSetLayout(app->device, &descSetLayoutInfo, NULL, &app->descriptorSetLayout);
}

static void createDescriptorPool(App *app){   
    VkDescriptorPoolSize poolSizes[1] = {0};

    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = (uint32_t)MAX_FRAMES_IN_FLIGHT;

    VkDescriptorPoolCreateInfo poolInfo = {0};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = (uint32_t)(sizeof(poolSizes) / sizeof(poolSizes[0]));
    poolInfo.pPoolSizes = poolSizes;
    poolInfo.maxSets = (uint32_t)MAX_FRAMES_IN_FLIGHT;
    // this flag allows individual descriptor sets to be freed via vkFreeDescriptorSets. We'll see if we need it later.
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

    if (vkCreateDescriptorPool(app->device, &poolInfo, NULL, &app->descriptorPool) != VK_SUCCESS) {
        fprintf(stderr, "Failed to create descriptor pool!\n");
        exit(1);
    }
}

static void createDescriptorSets(App *app){

    // We only have 1 descriptor layout (UBO), but we need to make one for each descriptor set
    VkDescriptorSetLayout duplicateDescLayouts[MAX_FRAMES_IN_FLIGHT];
    for (int i = 0; i< MAX_FRAMES_IN_FLIGHT; i++){
        duplicateDescLayouts[i] = app->descriptorSetLayout;
    }

    VkDescriptorSetAllocateInfo allocInfo = {0};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = app->descriptorPool;
    allocInfo.descriptorSetCount = (uint32_t)MAX_FRAMES_IN_FLIGHT;
    allocInfo.pSetLayouts = duplicateDescLayouts;

    VK_CHECK(vkAllocateDescriptorSets(app->device, &allocInfo, app->descriptorSets));

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        VkDescriptorBufferInfo bufferInfo = {0};
        bufferInfo.buffer = app->uniformBuffers[i];
        bufferInfo.offset = 0;
        bufferInfo.range  = sizeof(UniformBufferObject);

        VkWriteDescriptorSet descriptorWrites[1] = {0};

        descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[0].dstSet = app->descriptorSets[i];
        descriptorWrites[0].dstBinding = 0;
        descriptorWrites[0].dstArrayElement = 0;
        descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].pBufferInfo = &bufferInfo;

        vkUpdateDescriptorSets(app->device, 1, descriptorWrites, 0, NULL);
    }
}

/* -------------------------------------------- RENDER PIPELINE ------------------------------------------- */

static VkShaderModule createShaderModule(App* app, const char* code, size_t size) {
    VkShaderModuleCreateInfo createInfo = {0};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = size;
    createInfo.pCode = (const uint32_t*)code;

    VkShaderModule shaderModule;
    VK_CHECK(vkCreateShaderModule(app->device, &createInfo, NULL, &shaderModule));
    return shaderModule;
}

VkFormat findSupportedFormat(App *app, VkFormat *pCandidates, size_t candidateCount, VkImageTiling tiling, VkFormatFeatureFlags featureFlags){
    for (size_t i = 0; i < candidateCount; ++i){
        
        VkFormatProperties props;
        vkGetPhysicalDeviceFormatProperties(app->physicalDevice, pCandidates[i], &props);

        if ( tiling == VK_IMAGE_TILING_LINEAR && (featureFlags == (props.linearTilingFeatures & featureFlags))
        || tiling == VK_IMAGE_TILING_OPTIMAL && (featureFlags == (props.optimalTilingFeatures & featureFlags))){
            return pCandidates[i];
        }
    }

    fprintf(stderr, "Failed to find suitable image format type!\n");
    exit(1);
}

static void createGraphicsPipeline(App* app) {
    size_t shaderSize;
    char* shaderCode = readFile("shaders/vertMarchingPush.spv", &shaderSize);

    VkShaderModule shaderModule = createShaderModule(app, shaderCode, shaderSize);
    free(shaderCode);

    VkPipelineShaderStageCreateInfo vertStage = {0};
    vertStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertStage.module = shaderModule;
    vertStage.pName = "vertMain";

    VkPipelineShaderStageCreateInfo fragStage = {0};
    fragStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragStage.module = shaderModule;
    fragStage.pName = "fragMain";

    VkPipelineShaderStageCreateInfo stages[] = { vertStage, fragStage };

    VkVertexInputBindingDescription vertexBindingDesc = getVertex3DBindingDescription();
    VkVertexInputAttributeDescription vertexAttributeDescs[2];
    getVertex3DAttributeDescriptions(vertexAttributeDescs);

    VkPipelineVertexInputStateCreateInfo vertexInput = {0};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInput.pVertexBindingDescriptions = &vertexBindingDesc;
    vertexInput.vertexBindingDescriptionCount = 1;
    vertexInput.pVertexAttributeDescriptions = vertexAttributeDescs;
    vertexInput.vertexAttributeDescriptionCount = 2;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly = {0};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    //inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    //inputAssembly.primitiveRestartEnable = VK_FALSE;

    VkViewport viewport = {0};
    viewport.x = 0.0f;
    viewport.y = (float)app->swapchainExtent.height; //flipping vewport to negative Y to compensate for vulkan-gl y-flip
    viewport.width = (float)app->swapchainExtent.width;
    viewport.height = -(float)app->swapchainExtent.height; //negative now, because of above
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor = {0};
    scissor.offset.x = 0;
    scissor.offset.y = 0;
    scissor.extent = app->swapchainExtent;

    VkPipelineViewportStateCreateInfo viewportState = {0};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    VkPipelineRasterizationStateCreateInfo rasterizer = {0};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; //need to invert this to CCW because of the vulkan-gl y-flip

    VkPipelineMultisampleStateCreateInfo multisampling = {0};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisampling.minSampleShading = 1.0f;

    VkPipelineColorBlendAttachmentState colorBlendAttachment = {0};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_TRUE;
    colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
    colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

    VkPipelineColorBlendStateCreateInfo colorBlending = {0};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;

    VkPipelineDepthStencilStateCreateInfo depthStencil = {0};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.stencilTestEnable = VK_FALSE;

    VkPushConstantRange pushConstantRange = {0};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(PushConstants);

    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {0};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &app->descriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

    VK_CHECK(vkCreatePipelineLayout(app->device, &pipelineLayoutInfo, NULL, &app->pipelineLayout));

    VkFormat depthFormats[] = { VK_FORMAT_D24_UNORM_S8_UINT };
    app->depthFormat = findSupportedFormat(app, depthFormats, 1, VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
    VkPipelineRenderingCreateInfo pipelineRenderingCreateInfo = {0};
    pipelineRenderingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    pipelineRenderingCreateInfo.colorAttachmentCount = 1;
    pipelineRenderingCreateInfo.pColorAttachmentFormats = &app->swapchainFormat;
    pipelineRenderingCreateInfo.depthAttachmentFormat = app->depthFormat;

    VkGraphicsPipelineCreateInfo graphicsPipelineInfo = {0};
    graphicsPipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    graphicsPipelineInfo.pNext = &pipelineRenderingCreateInfo;
    graphicsPipelineInfo.stageCount = 2;
    graphicsPipelineInfo.pStages = stages;
    graphicsPipelineInfo.pVertexInputState = &vertexInput;
    graphicsPipelineInfo.pInputAssemblyState = &inputAssembly;
    graphicsPipelineInfo.pViewportState = &viewportState;
    graphicsPipelineInfo.pRasterizationState = &rasterizer;
    graphicsPipelineInfo.pMultisampleState = &multisampling;
    graphicsPipelineInfo.pColorBlendState = &colorBlending;
    graphicsPipelineInfo.pDepthStencilState = &depthStencil;
    graphicsPipelineInfo.layout = app->pipelineLayout;
    graphicsPipelineInfo.renderPass = NULL; //not using render pass anymore?
    graphicsPipelineInfo.subpass = 0;
    graphicsPipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

    VK_CHECK(vkCreateGraphicsPipelines(app->device, VK_NULL_HANDLE, 1, &graphicsPipelineInfo, NULL, &app->graphicsPipeline));

    vkDestroyShaderModule(app->device, shaderModule, NULL);
}

/* -------------------------------------------------- BUFFERS ------------------------------------------------ */


uint32_t findSupportedMemoryType(VkPhysicalDeviceMemoryProperties *pPhysicalProperies, uint32_t typeFilter, VkMemoryPropertyFlags propertyFlags) {

    // vulkan docs says there are two arrays, types and heaps. Ignoring heaps for now.
    for (uint32_t i = 0; i < pPhysicalProperies->memoryTypeCount; i++){
        if ((typeFilter & (1 << i)) && (pPhysicalProperies->memoryTypes[i].propertyFlags & propertyFlags) == propertyFlags){
            return i;
        }
    }

    fprintf(stderr, "Failed to find suitable memory type!\n");
    exit(1);
}

static void createBuffer(App *app, VkBufferCreateInfo *pCreateInfo, VkBuffer *pBuffer, VkDeviceMemory *pDeviceMemory, void **ppMappedBuffer){

        vkCreateBuffer(app->device, pCreateInfo, NULL, pBuffer);

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(app->device, *pBuffer, &memRequirements);

        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(app->physicalDevice, &memProperties);

        uint32_t memType = findSupportedMemoryType(&memProperties, memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VkMemoryAllocateInfo allocInfo = {0};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = memType;

        if (vkAllocateMemory(app->device, &allocInfo, NULL, pDeviceMemory) != VK_SUCCESS) {
            fprintf(stderr, "Failed to allocate buffer memory!\n");
            exit(1);
        }

        vkBindBufferMemory(app->device, *pBuffer, *pDeviceMemory, 0);
        vkMapMemory(app->device, *pDeviceMemory, 0, pCreateInfo->size, 0, ppMappedBuffer);
        //vkUnmapMemory(app->device, app->deviceMemory[i]); unmap here?
}

static void createVertexBuffers(App* app){
    VkBufferCreateInfo vertexBufferCreateInfo = {0};
    vertexBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    vertexBufferCreateInfo.size = app->vertexBufferMaxSize * sizeof(Vertex3D);
    vertexBufferCreateInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    vertexBufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        createBuffer(app, &vertexBufferCreateInfo, &app->vertexBuffers[i], &app->deviceVertexMemory[i], &app->pVertexBuffersMapped[i]);
    }
}

static void createIndexBuffer(App* app){
    VkBufferCreateInfo indexBufferCreateInfo = {0};
    indexBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    indexBufferCreateInfo.size = app->vertexBufferMaxSize * sizeof(uint32_t);
    indexBufferCreateInfo.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    indexBufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    createBuffer(app, &indexBufferCreateInfo, &app->indexBuffer, &app->deviceIndexMemory, &app->pIndexBufferMapped);
}

static void createUniformBuffers(App *app){
    VkBufferCreateInfo bufferCreateInfo = {0};
    bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size = sizeof(UniformBufferObject);
    bufferCreateInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++){
        createBuffer(app, &bufferCreateInfo, &app->uniformBuffers[i], &app->uniformBuffersMemory[i], &app->pUniformBuffersMapped[i]);
    }
}

static void updateUniformBuffers(App *app, Camera *pCamera, Model3D *pModel){
    UniformBufferObject ubo = {0};

    mat4_TRS(ubo.model, pModel->transform.position, pModel->transform.rotation, pModel->transform.scale);
    mat4_inverse_affine(ubo.inverseModel, ubo.model);
    mat4_TRS(ubo.inverseView, pCamera->transform.position, pCamera->transform.rotation, pCamera->transform.scale);
    mat4_inverse_affine(ubo.view, ubo.inverseView);
    memcpy(ubo.projection, pCamera->projection, sizeof(mat4));
    memcpy(ubo.inverseProjection, pCamera->inverseProjection, sizeof(mat4));
    vec2_set(&ubo.iResolution, app->swapchainExtent.width, app->swapchainExtent.height);

    memcpy(&ubo.primitives, app->pPrimitives, app->primitiveCount * sizeof(PrimitiveUBO));

    // copy to ubo buffer
    memcpy(app->pUniformBuffersMapped[app->currentFrameIndex], &ubo, sizeof(ubo));
}

static void updatePushConstants(App *app, Camera *pCamera, Light *pLight){
    PushConstants pc;
    vec4_from_vec3(&pc.cameraPosition, pCamera->transform.position, 1.0f);
    vec4_from_vec3(&pc.lightPosition, pLight->position, 1.0f);
    pc.lightColour = pLight->colour;

    VkShaderStageFlags shaderStageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    vkCmdPushConstants(app->pCommandBuffers[app->currentFrameIndex], app->pipelineLayout, shaderStageFlags, 0, sizeof(PushConstants), &pc);
}

/* ------------------------------------------------- TEXTURES ------------------------------------------------ */

static void createImage(App *app, VkImage *pImage, VkDeviceMemory *pImageMemory,
    uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags propertyFlags){
    
    VkImageCreateInfo imageInfo = {0};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType   = VK_IMAGE_TYPE_2D;
    imageInfo.format      = format;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels   = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples     = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling      = tiling;
    imageInfo.usage       = usage;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VK_CHECK(vkCreateImage(app->device, &imageInfo, NULL, pImage));

    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(app->device, *pImage, &memRequirements);

    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(app->physicalDevice, &memProperties);

    VkMemoryAllocateInfo allocInfo = {0};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findSupportedMemoryType(&memProperties, memRequirements.memoryTypeBits , propertyFlags);

    VK_CHECK(vkAllocateMemory(app->device, &allocInfo, NULL, pImageMemory));

    VK_CHECK(vkBindImageMemory(app->device, *pImage, *pImageMemory, 0));
}

static void createDepthImages(App *app){
    createImage(app, &app->depthImage, &app->depthImageMemory, app->swapchainExtent.width, app->swapchainExtent.height, 
        app->depthFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    createImageView(app, &app->depthImageView, &app->depthImage, &app->depthFormat, VK_IMAGE_ASPECT_STENCIL_BIT | VK_IMAGE_ASPECT_DEPTH_BIT );
}

/* ---------------------------------------------- COMMANDS ------------------------------------------- */

static void createCommandPool(App* app) {
    VkCommandPoolCreateInfo poolInfo = {0};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = app->graphicsFamily;

    VK_CHECK(vkCreateCommandPool(app->device, &poolInfo, NULL, &app->commandPool));
}

static void createCommandBuffers(App* app) {
    app->pCommandBuffers = malloc(MAX_FRAMES_IN_FLIGHT * sizeof(VkCommandBuffer));

    VkCommandBufferAllocateInfo allocInfo = {0};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = app->commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = MAX_FRAMES_IN_FLIGHT;

    VK_CHECK(vkAllocateCommandBuffers(app->device, &allocInfo, app->pCommandBuffers));
}

static void transitionImageLayout(App *app, VkImage *pImage, VkImageLayout oldLayout, VkImageLayout newLayout, VkImageAspectFlags aspectFlags, 
    VkAccessFlags srcAccessMask, VkAccessFlags dstAccessMask, VkPipelineStageFlags srcStageMask, VkPipelineStageFlags dstStageMask) {

    VkImageSubresourceRange subresourceRange = {0};
    subresourceRange.baseMipLevel = 0;
    subresourceRange.levelCount = 1;
    subresourceRange.baseArrayLayer = 0;
    subresourceRange.layerCount = 1;
    subresourceRange.aspectMask = aspectFlags;

    VkImageMemoryBarrier2 barrier = {0};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barrier.srcStageMask        = srcStageMask;
    barrier.srcAccessMask       = srcAccessMask;
    barrier.dstStageMask        = dstStageMask;
    barrier.dstAccessMask       = dstAccessMask;
    barrier.oldLayout           = oldLayout;
    barrier.newLayout           = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image               = *pImage;
    barrier.subresourceRange    = subresourceRange;

    VkDependencyInfo dependencyInfo = {0};
    dependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependencyInfo.dependencyFlags         = (VkDependencyFlags){0};
    dependencyInfo.imageMemoryBarrierCount = 1;
    dependencyInfo.pImageMemoryBarriers    = &barrier;

    vkCmdPipelineBarrier2(app->pCommandBuffers[app->currentFrameIndex], &dependencyInfo);
}

static void recordCommandBuffer(App* app, uint32_t imageIndex) {
    VkCommandBuffer cmd = app->pCommandBuffers[app->currentFrameIndex];
    VK_CHECK(vkResetCommandBuffer(cmd, 0));

    VkCommandBufferBeginInfo beginInfo = {0};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));

    // Transition the color image layout for rendering
    transitionImageLayout(
        app, 
        &app->swapchainImages[imageIndex], 
        VK_IMAGE_LAYOUT_UNDEFINED, 
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 
        VK_IMAGE_ASPECT_COLOR_BIT,
        (VkAccessFlagBits)0, 
        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, 
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
    );

    // Transition the depth image layout for rendering
    transitionImageLayout(
        app, 
        &app->depthImage, 
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 
        VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT,
        0,
        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, 
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, 
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT
    );

    //color buffer
    VkClearValue clearColor = {{{0.0f, 0.25f, 1.0f, 1.0f}}};
    VkRenderingAttachmentInfo colorAttachmentInfo = {0};
    colorAttachmentInfo.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachmentInfo.clearValue = clearColor;
    colorAttachmentInfo.imageView = app->swapchainImageViews[imageIndex];
    colorAttachmentInfo.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachmentInfo.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachmentInfo.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    // depth/stencil buffer
    VkClearDepthStencilValue clearDepthValues = {0};
    clearDepthValues.depth = 1.0f;
    clearDepthValues.stencil = 0;
    clearColor.depthStencil = clearDepthValues;
    VkRenderingAttachmentInfo depthAttchmentInfo = {0};
    depthAttchmentInfo.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depthAttchmentInfo.clearValue = (VkClearValue)clearColor;
    depthAttchmentInfo.imageView = app->depthImageView;
    depthAttchmentInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depthAttchmentInfo.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttchmentInfo.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

    VkRenderingInfo renderingInfo = {0};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderingInfo.renderArea.extent = app->swapchainExtent;
    renderingInfo.renderArea.offset = (VkOffset2D){0, 0};
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachmentInfo;
    renderingInfo.pDepthAttachment = &depthAttchmentInfo;

    vkCmdBeginRendering(cmd, &renderingInfo);
    
    //for each camera...
    //for each light...
    updatePushConstants(app, &app->camera, &app->light);

    //for each model...
    updateUniformBuffers(app, &app->camera, &app->model);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, app->graphicsPipeline);

    VkBuffer vertexBuffers[] = {app->vertexBuffers[app->currentFrameIndex]};
    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(cmd, 0, 1, vertexBuffers, offsets);
    vkCmdBindIndexBuffer(cmd, app->indexBuffer, offsets[0], VK_INDEX_TYPE_UINT32);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, app->pipelineLayout, 0, 1, &app->descriptorSets[app->currentFrameIndex], 0, NULL);

    vkCmdDrawIndexed(cmd, app->model.mesh.indexCount, 1, 0, 0, 0); //draw call here

    vkCmdEndRendering(cmd);

    // Transition the color image layout for presentation
    transitionImageLayout(
        app, 
        &app->swapchainImages[imageIndex], 
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 
        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        VK_IMAGE_ASPECT_COLOR_BIT,
        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        0, 
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT
    );

    VK_CHECK(vkEndCommandBuffer(cmd));
}

static void createSyncObjects(App* app) {
    VkSemaphoreCreateInfo semaphoreInfo = {0};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo = {0};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    app->pRenderFinishedSemaphores = malloc(sizeof(VkSemaphore) * app->swapchainImageCount);

    for (uint32_t i = 0; i < app->swapchainImageCount; i++) {
        VK_CHECK(vkCreateSemaphore(app->device, &semaphoreInfo, NULL, &app->pRenderFinishedSemaphores[i]));
    }

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        VK_CHECK(vkCreateSemaphore(app->device, &semaphoreInfo, NULL, &app->imageAvailableSemaphores[i]));
        VK_CHECK(vkCreateFence(app->device, &fenceInfo, NULL, &app->inFlightFences[i]));
    }
}

/* ----------------------------------- SWAPCHAIN CREATE AND DESTROY ------------------------------------ */

static void cleanupSwapchain(App* app) {

    vkFreeCommandBuffers(app->device, app->commandPool, MAX_FRAMES_IN_FLIGHT, app->pCommandBuffers);
    free(app->pCommandBuffers);

    vkDestroyPipeline(app->device, app->graphicsPipeline, NULL);
    vkDestroyPipelineLayout(app->device, app->pipelineLayout, NULL);

    for (uint32_t i = 0; i < app->swapchainImageCount; i++) {
        vkDestroyImageView(app->device, app->swapchainImageViews[i], NULL);
    }
    vkDestroyImageView(app->device, app->depthImageView, NULL);

    vkFreeMemory(app->device, app->depthImageMemory, NULL);
    free(app->swapchainImageViews);
    free(app->swapchainImages);

    vkDestroySwapchainKHR(app->device, app->swapchain, NULL);
}

static void recreateSwapchain(App* app) {
    int width = 0, height = 0;
#if SDL_MAJOR_VERSION >= 3
    SDL_GetWindowSizeInPixels(app->window, &width, &height);
    while (width == 0 || height == 0) {
        SDL_GetWindowSizeInPixels(app->window, &width, &height);
        SDL_WaitEvent(NULL);
    }
#else
    SDL_Vulkan_GetDrawableSize(app->window, &width, &height);
    while (width == 0 || height == 0) {
        SDL_Vulkan_GetDrawableSize(app->window, &width, &height);
        SDL_WaitEvent(NULL);
    }
#endif
    vkDeviceWaitIdle(app->device);

    cleanupSwapchain(app);

    createSwapchain(app);
    createGraphicsPipeline(app);
    createDepthImages(app);
    createCommandBuffers(app);
}

/* ----------------------------------------- DRAW LOOP ------------------------------------------- */

static void drawFrame(App* app) {
    vkWaitForFences(app->device, 1, &app->inFlightFences[app->currentFrameIndex], VK_TRUE, UINT64_MAX);
    
    vkResetFences(app->device, 1, &app->inFlightFences[app->currentFrameIndex]);

    uint32_t imageIndex;
    VkResult result = vkAcquireNextImageKHR(app->device, app->swapchain, UINT64_MAX, app->imageAvailableSemaphores[app->currentFrameIndex], VK_NULL_HANDLE, &imageIndex);
    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        recreateSwapchain(app);
        return;
    } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        fprintf(stderr, "Failed to acquire swapchain image\n");
        exit(1);
    }

    recordCommandBuffer(app, imageIndex);

    VkSemaphore waitSemaphores[] = { app->imageAvailableSemaphores[app->currentFrameIndex] };
    VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
    VkSemaphore signalSemaphores[] = { app->pRenderFinishedSemaphores[imageIndex] };

    VkSubmitInfo submitInfo = {0};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &app->pCommandBuffers[app->currentFrameIndex];
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    VK_CHECK(vkQueueSubmit(app->graphicsQueue, 1, &submitInfo, app->inFlightFences[app->currentFrameIndex]));

    VkPresentInfoKHR presentInfo = {0};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;
    VkSwapchainKHR swapchains[] = { app->swapchain };
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapchains;
    presentInfo.pImageIndices = &imageIndex;

    result = vkQueuePresentKHR(app->presentQueue, &presentInfo);

    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || app->framebufferResized) {
        app->framebufferResized = false;
        recreateSwapchain(app);
    } else if (result != VK_SUCCESS) {
        fprintf(stderr, "Failed to present swapchain image\n");
        exit(1);
    }

    app->currentFrameIndex = (app->currentFrameIndex + 1) % MAX_FRAMES_IN_FLIGHT;
}

/*--------------------------------------------- DELTA TIME -------------------------------------------------- */

static void startTimers(App *app){
    app->isFirstFrame = true;
    app->lastFrameTime = 0.0f;
    app->deltaTime = 0.0f;
    app->timer = 0.0f;
    app->fps = 0.0f;
}

static void updateTimers(App* app) {
    uint64_t now = SDL_GetPerformanceCounter();
    uint64_t freq = SDL_GetPerformanceFrequency();
    float dt = 0.0f;
    if(!app->isFirstFrame){
        dt = (float)((now - app->lastFrameTime) / (float)freq);
        app->fps = 1.0f/dt;
    }else{
        app->isFirstFrame = false;
    }
    app->deltaTime = dt;
    app->lastFrameTime = now;
    app->timer += app->deltaTime;
}


/* -------------------------------------------- PRE-SETUP (AWAKE) -------------------------------------------- */

static void awake(App *app){
    app->vertexBufferMaxSize = UINT16_MAX * 256;
}

static void createSDFLens(Mesh3D *pMesh, int fidelity, float aspectRatio) {
    
    int width = fidelity, height = (fidelity / aspectRatio);
    int vCols = width + 1, vRows = height + 1;
    int vertexCount = vCols * vRows;
    int indexCount = width * height * 2 * 3;

    pMesh->pVertices = (Vertex3D*)malloc(sizeof(Vertex3D) * vertexCount);
    pMesh->pIndices = (uint32_t*)malloc(sizeof(uint32_t) * indexCount);

    float size = 2.0f;
    float half = size * 0.5f; //2.0 is size of clip space

    for (int row = 0; row < vRows; row++) {
        for (int col = 0; col < vCols; col++) {

            float mag;
            float rx = (rand() % 100) / 100.f, ry = (rand() % 100) / 100.f, rz = (rand() % 100) / 100.f;
            vec3_mag(&mag, (vec3){rx, ry, rz});

            int idx = row * vCols + col;
            float u = (float)col / width;
            float v = (float)row / height;

            vec4 projSpace, viewSpace;
            vec4_set(&projSpace, (u * size - half), (v * size - half), 0.0f, 1.0f); // y-flip for vulkan. And z starts at 0.0.

            vec3_set(&pMesh->pVertices[idx].position, projSpace.x, projSpace.y, projSpace.z);
            //vec3_set(&pMesh->pVertices[idx].position, projSpace.x, projSpace.y, projSpace.z);
            vec4_set(&pMesh->pVertices[idx].colour, rx/mag, ry/mag, rz/mag, 1.0);
            //vec4_set(&pMesh->pVertices[idx].colour, 1.0f, 1.0f, 1.0f, 1.0);
            //vec4_set(&pMesh->pVertices[idx].colour, 0.0f, 0.0f, 0.0f, 1.0);

            pMesh->vertexCount++;
        }
    }

    int t = 0;
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {

            int topLeft     = row * vCols + col;
            int topRight    = topLeft + 1;
            int bottomLeft  = (row + 1) * vCols + col;
            int bottomRight = bottomLeft + 1;

            pMesh->pIndices[t] = topLeft;
            pMesh->pIndices[t + 1] = bottomLeft;
            pMesh->pIndices[t + 2] = topRight;

            pMesh->pIndices[t + 3] = topRight;
            pMesh->pIndices[t + 4] = bottomLeft;
            pMesh->pIndices[t + 5] = bottomRight;
            t += 6;
        }
    }

    pMesh->indexCount = t;

    printf("Created SDFLens of %dx%d quads, [%d] triangles.\n", width, height, indexCount/3);

}





static void startSim(App *app){

}

/* -------------------------------------------- POST-SETUP (START) -------------------------------------------- */

static void start(App *app){

    //vec4 col = {1.0f, 1.0f, 1.0f, 1.0f};
    //parseOBJ(&app->model.mesh, "./sphere.obj", &col);
    
    srand((int)SDL_GetPerformanceCounter());

    mat4_perspective(app->camera.projection, PI/3, (float)app->swapchainExtent.width/(float)app->swapchainExtent.height, 0.1f, 100.0f);

    mat4 cloudLensProjection;
    mat4_perspective(cloudLensProjection, PI/3, (float)app->swapchainExtent.width/(float)app->swapchainExtent.height, 0.1f, 100.0f);
    mat4_inverse(app->camera.inverseProjection, cloudLensProjection);

    createSDFLens(&app->model.mesh, 64, (float)app->swapchainExtent.width/(float)app->swapchainExtent.height);

    memcpy(app->pIndexBufferMapped, app->model.mesh.pIndices, sizeof(uint32_t) * app->model.mesh.indexCount);
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        memcpy(app->pVertexBuffersMapped[i], app->model.mesh.pVertices, sizeof(Vertex3D) * app->model.mesh.vertexCount);
    }

    vec3_set(&app->model.transform.position, 0.0f, 0.0f, 0.0f);
    vec3_set(&app->model.transform.rotation, 0.0f, 0.0f, 0.0f);
    vec3_set(&app->model.transform.scale, 1.0f, 1.0f, 1.0f);

    vec3_set(&app->camera.transform.position, 0.0f, 0.0f, -10.0f);
    vec3_set(&app->camera.transform.rotation, 0.0f, PI, 0.0f);
    vec3_set(&app->camera.transform.scale, 1.0f, 1.0f, 1.0f);

    vec3_set(&app->light.position, 0.0f, 5.0f, 5.0f);
    vec4_set(&app->light.colour, 1.0f, 0.8f, 0.8f, 1.0f);

    app->primitiveCount = 128;
    app->pPrimitives = (PrimitiveUBO*)malloc(sizeof(PrimitiveUBO) * app->primitiveCount);

    for(int i = 0; i < app->primitiveCount; i++){

        float rx = (rand() % 100) / 100.f, ry = (rand() % 100) / 100.f, rz = (rand() % 100) / 100.f;

        app->pPrimitives[i].type = 'S';
        app->pPrimitives[i].operation = 'a';
        app->pPrimitives[i].smoothing = 1.0f;
        vec3_set(&app->pPrimitives[i].position, rx * 10.0f - 5.0f, ry * 10.0f - 5.0f, rz * 10.0f - 5.0f);
        vec3_set(&app->pPrimitives[i].rotation, 0.0f, 0.0f, 0.0f);
        vec3_set(&app->pPrimitives[i].scale, 0.5 + rx, 0.5 + rx, 0.5 + rx);
    }
}

/* -------------------------------------------- UPDATE ------------------------------------------- */

static void update(App *app){

    const uint8_t *pKeyStates = SDL_GetKeyboardState(NULL);

    printf("RESOLUTION: %d x %d, FPS: %f\r", app->swapchainExtent.width, app->swapchainExtent.height, app->fps);

    mat4 cameraTRS;
    vec3 scaledCameraX, scaledCameraZ;
    float speed = 3.0f;
    mat4_TRS(cameraTRS, app->camera.transform.position, app->camera.transform.rotation, app->camera.transform.scale);
    vec3_scale(&scaledCameraX, *(vec3*)cameraTRS[0], app->deltaTime * speed);
    vec3_scale(&scaledCameraZ, *(vec3*)cameraTRS[2], app->deltaTime * speed);

    if(pKeyStates[SDL_SCANCODE_A]){
        vec3_sub(&app->camera.transform.position, app->camera.transform.position, scaledCameraX);
    }
    if(pKeyStates[SDL_SCANCODE_D]){
        vec3_add(&app->camera.transform.position, app->camera.transform.position, scaledCameraX);
    }
    if(pKeyStates[SDL_SCANCODE_W]){
        vec3_sub(&app->camera.transform.position, app->camera.transform.position, scaledCameraZ);
    }
    if(pKeyStates[SDL_SCANCODE_S]){
        vec3_add(&app->camera.transform.position, app->camera.transform.position, scaledCameraZ);
    }
    if(pKeyStates[SDL_SCANCODE_LEFT]){
        app->camera.transform.rotation.y += app->deltaTime;
    }
    if(pKeyStates[SDL_SCANCODE_RIGHT]){
        app->camera.transform.rotation.y -= app->deltaTime;
    }
    if(pKeyStates[SDL_SCANCODE_UP]){
        app->camera.transform.position.y += app->deltaTime;
    }
    if(pKeyStates[SDL_SCANCODE_DOWN]){
        app->camera.transform.position.y -= app->deltaTime;
    }

    if(app->pause){
        return;
    }

    for(int i = 0; i < app->primitiveCount; i++){
        app->pPrimitives[i].position.y = cos(app->timer * app->pPrimitives[i].scale.x);
        //app->pPrimitives[i].position.x = 5.0 * sin(app->timer * app->pPrimitives[i].scale.x);
        //app->pPrimitives[i].position.z = 5.0 * cos(app->timer * app->pPrimitives[i].scale.x);
    }

    //app->camera.transform.rotation.z = cos(app->timer);

}

/* --------------------------------------- INPUT HOOKS ------------------------------------------- */

static void onKeyDown(App *app, SDL_KeyCode key){
    if(key == SDLK_SPACE){
        app->pause = !app->pause;
    }
}

static void onKeyUp(App *app, SDL_KeyCode key){

}

/* ---------------------------------------------- MAIN ------------------------------------------- */

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    App app;
    memset(&app, 0, sizeof(app));

    if (!initWindow(&app)) return 1;

    // Create everything!

    awake(&app);
    createInstance(&app);
    createSurface(&app);
    pickPhysicalDevice(&app);
    createLogicalDevice(&app);
    createSwapchain(&app);
    createDescriptorSetLayout(&app);
    createGraphicsPipeline(&app);
    createCommandPool(&app);
    createDepthImages(&app);
    createVertexBuffers(&app);
    createIndexBuffer(&app);
    createUniformBuffers(&app);
    createDescriptorPool(&app);
    createDescriptorSets(&app);
    createCommandBuffers(&app);
    createSyncObjects(&app);
    startTimers(&app);
    start(&app);

    // Run loop!

    bool running = true;
    SDL_Event event;
    while (running) {

        while (SDL_PollEvent(&event)) {
#if SDL_MAJOR_VERSION >= 3
            if (event.type == SDL_EVENT_QUIT) running = false;
            if (event.type == SDL_EVENT_WINDOW_RESIZED) app.framebufferResized = true;
#else
            if (event.type == SDL_QUIT) running = false;
            if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_RESIZED) app.framebufferResized = true;
            if (event.type == SDL_KEYDOWN) onKeyDown(&app, event.key.keysym.sym);
            if (event.type == SDL_KEYUP) onKeyUp(&app, event.key.keysym.sym);

#endif
        }

        update(&app);
        drawFrame(&app);
        updateTimers(&app);
    }
    
    vkDeviceWaitIdle(app.device);

    // Clean everything up!
    // remember to free all the malloced stuff here

    cleanupSwapchain(&app);
    vkDestroyBuffer(app.device, app.indexBuffer, NULL);
    vkFreeMemory(app.device, app.deviceIndexMemory, NULL);
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vkDestroySemaphore(app.device, app.imageAvailableSemaphores[i], NULL);
        vkDestroyFence(app.device, app.inFlightFences[i], NULL);
        vkDestroyBuffer(app.device, app.vertexBuffers[i], NULL);
        vkDestroyBuffer(app.device, app.uniformBuffers[i], NULL);
        vkFreeMemory(app.device, app.deviceVertexMemory[i], NULL);
        vkFreeMemory(app.device, app.uniformBuffersMemory[i], NULL);
    }
    for (uint32_t i = 0; i < app.swapchainImageCount; i++) {
        vkDestroySemaphore(app.device, app.pRenderFinishedSemaphores[i], NULL);
    }
    free(app.pRenderFinishedSemaphores);
    vkDestroyCommandPool(app.device, app.commandPool, NULL);
    vkDestroyDevice(app.device, NULL);
    vkDestroySurfaceKHR(app.instance, app.surface, NULL);
    vkDestroyInstance(app.instance, NULL);
    SDL_DestroyWindow(app.window);
    SDL_Quit();

    return 0;
}
