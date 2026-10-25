SDL_VERSION ?= 2
BUILD_DIR   ?= build

ifeq ($(SDL_VERSION),3)
    SDL_PKG    = sdl3
    SDL_DEFINE = -DUSE_SDL3
else
    SDL_PKG    = sdl2
    SDL_DEFINE =
endif

CC      = gcc
CFLAGS  = -std=c11 -Wall -Wextra -MMD -MP -g $(SDL_DEFINE) $(shell pkg-config --cflags $(SDL_PKG))
LDFLAGS = -lvulkan $(shell pkg-config --libs $(SDL_PKG)) -lm
SLANGC  = $(VULKAN_SDK)/bin/slangc

SRCS = pegasus.c
OBJS = $(SRCS:%.c=$(BUILD_DIR)/%.o)
DEPS = $(OBJS:.o=.d)
TARGET = $(BUILD_DIR)/pegasus

all: shaders $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $@ $(LDFLAGS)

# Compile into BUILD_DIR; -MMD drops the matching .d file beside the .o
$(BUILD_DIR)/%.o: %.c
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

-include $(DEPS)



shaders: shaders/phong.spv shaders/sdf.spv shaders/vertMarching.spv shaders/vertMarchingPush.spv

shaders/phong.spv: phong.slang
	@mkdir -p shaders
	$(SLANGC) phong.slang -target spirv -profile spirv_1_4 \
	    -emit-spirv-directly -fvk-use-entrypoint-name \
	    -entry vertMain -entry fragMain -o shaders/phong.spv

shaders/sdf.spv: sdf.slang
	@mkdir -p shaders
	$(SLANGC) sdf.slang -target spirv -profile spirv_1_4 \
	    -emit-spirv-directly -fvk-use-entrypoint-name \
	    -entry vertMain -entry fragMain -o shaders/sdf.spv

shaders/vertMarching.spv: vertMarching.slang
	@mkdir -p shaders
	$(SLANGC) vertMarching.slang -target spirv -profile spirv_1_4 \
	    -emit-spirv-directly -fvk-use-entrypoint-name \
	    -entry vertMain -entry fragMain -o shaders/vertMarching.spv

shaders/vertMarchingPush.spv: vertMarchingPush.slang
	@mkdir -p shaders
	$(SLANGC) vertMarchingPush.slang -target spirv -profile spirv_1_4 \
		-emit-spirv-directly -fvk-use-entrypoint-name \
		-entry vertMain -entry fragMain -o shaders/vertMarchingPush.spv

clean:
	rm -rf $(BUILD_DIR) shaders

.PHONY: all clean shaders
