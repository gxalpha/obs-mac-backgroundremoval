/*
Plugin Name
Copyright (C) <Year> <Developer> <Email Address>

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program. If not, see <https://www.gnu.org/licenses/>
*/

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>

// ARM-specific optimizations
#ifdef __arm64__
#define OBS_ARM_OPTIMIZED 1
#include <arm_neon.h>
#else
#define OBS_ARM_OPTIMIZED 0
#endif

// Cache alignment for ARM processors
#define OBS_ARM_CACHE_LINE_SIZE 64
#define OBS_ARM_ALIGNED __attribute__((aligned(OBS_ARM_CACHE_LINE_SIZE)))

// Plugin constants
extern const char *PLUGIN_NAME;
extern const char *PLUGIN_VERSION;

// Standard logging levels
#ifndef LOG_ERROR
#define LOG_ERROR   3
#endif
#ifndef LOG_WARNING
#define LOG_WARNING 4
#endif
#ifndef LOG_INFO
#define LOG_INFO    6
#endif
#ifndef LOG_DEBUG
#define LOG_DEBUG   7
#endif

// Core logging functions
void obs_log(int log_level, const char *format, ...);
void obs_log_with_metrics(int log_level, const char *function_name, 
                         uint64_t execution_time_ns, const char *format, ...);

// External OBS logging function
extern void blogva(int log_level, const char *format, va_list args);

// Performance monitoring macros
#define OBS_PERF_START() uint64_t __start_time = obs_get_high_precision_time()
#define OBS_PERF_END_LOG(level, func_name, format, ...) \
    do { \
        uint64_t __end_time = obs_get_high_precision_time(); \
        obs_log_with_metrics(level, func_name, __end_time - __start_time, format, ##__VA_ARGS__); \
    } while(0)

// Memory management utilities
char *obs_strdup_optimized(const char *src);
int obs_strncat_safe(char *dst, size_t dst_size, const char *src);
void obs_memcpy_optimized(void *dst, const void *src, size_t size);
size_t obs_align_for_arm_cache(size_t size);

// High-precision timing
uint64_t obs_get_high_precision_time(void);

// ARM-optimized data structures
typedef struct {
    uint8_t *data OBS_ARM_ALIGNED;
    size_t size;
    size_t capacity;
    size_t aligned_size;
} obs_arm_buffer_t;

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t stride;
    uint32_t format;
    void *data OBS_ARM_ALIGNED;
    size_t data_size;
} obs_arm_frame_t;

// ARM-specific memory allocation
static inline void *obs_aligned_alloc(size_t alignment, size_t size) {
    void *ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return NULL;
    }
    return ptr;
}

static inline void obs_aligned_free(void *ptr) {
    free(ptr);
}

// ARM NEON optimized functions (when available)
#if OBS_ARM_OPTIMIZED
static inline void obs_memcpy_neon(void *dst, const void *src, size_t size) {
    // Use NEON for large transfers
    if (size >= 64) {
        uint8_t *d = (uint8_t *)dst;
        const uint8_t *s = (const uint8_t *)src;
        
        // Process 64-byte chunks with NEON
        while (size >= 64) {
            uint8x16x4_t data = vld1q_u8_x4(s);
            vst1q_u8_x4(d, data);
            s += 64;
            d += 64;
            size -= 64;
        }
        
        // Handle remaining bytes
        if (size > 0) {
            memcpy(d, s, size);
        }
    } else {
        memcpy(dst, src, size);
    }
}

static inline void obs_memset_neon(void *dst, int value, size_t size) {
    if (size >= 64) {
        uint8_t *d = (uint8_t *)dst;
        uint8x16_t val = vdupq_n_u8((uint8_t)value);
        
        while (size >= 64) {
            vst1q_u8(d, val);
            vst1q_u8(d + 16, val);
            vst1q_u8(d + 32, val);
            vst1q_u8(d + 48, val);
            d += 64;
            size -= 64;
        }
        
        if (size > 0) {
            memset(d, value, size);
        }
    } else {
        memset(dst, value, size);
    }
}

// ARM-optimized alpha blending
static inline void obs_blend_alpha_neon(uint8_t *dst, const uint8_t *src, 
                                       const uint8_t *alpha, size_t count) {
    size_t i = 0;
    
    // Process 16 pixels at a time with NEON
    for (; i + 16 <= count; i += 16) {
        uint8x16_t src_data = vld1q_u8(&src[i * 4]);
        uint8x16_t dst_data = vld1q_u8(&dst[i * 4]);
        uint8x16_t alpha_data = vld1q_u8(&alpha[i]);
        
        // Expand alpha to 4 channels (RGBA)
        uint8x16x4_t alpha_expanded = {
            vdupq_laneq_u8(alpha_data, 0),
            vdupq_laneq_u8(alpha_data, 1),
            vdupq_laneq_u8(alpha_data, 2),
            vdupq_laneq_u8(alpha_data, 3)
        };
        
        // Blend: dst = dst * (1 - alpha) + src * alpha
        uint16x8_t blend_lo = vmull_u8(vget_low_u8(src_data), vget_low_u8(alpha_expanded.val[0]));
        uint16x8_t blend_hi = vmull_u8(vget_high_u8(src_data), vget_high_u8(alpha_expanded.val[0]));
        
        uint8x16_t result = vcombine_u8(
            vqmovn_u16(blend_lo),
            vqmovn_u16(blend_hi)
        );
        
        vst1q_u8(&dst[i * 4], result);
    }
    
    // Handle remaining pixels
    for (; i < count; i++) {
        uint8_t a = alpha[i];
        dst[i * 4 + 0] = (dst[i * 4 + 0] * (255 - a) + src[i * 4 + 0] * a) / 255;
        dst[i * 4 + 1] = (dst[i * 4 + 1] * (255 - a) + src[i * 4 + 1] * a) / 255;
        dst[i * 4 + 2] = (dst[i * 4 + 2] * (255 - a) + src[i * 4 + 2] * a) / 255;
        dst[i * 4 + 3] = (dst[i * 4 + 3] * (255 - a) + src[i * 4 + 3] * a) / 255;
    }
}

#else
// Fallback implementations for non-ARM platforms
#define obs_memcpy_neon(dst, src, size) memcpy(dst, src, size)
#define obs_memset_neon(dst, value, size) memset(dst, value, size)
#define obs_blend_alpha_neon(dst, src, alpha, count) obs_blend_alpha_generic(dst, src, alpha, count)
#endif

// Generic alpha blending fallback
static inline void obs_blend_alpha_generic(uint8_t *dst, const uint8_t *src, 
                                          const uint8_t *alpha, size_t count) {
    for (size_t i = 0; i < count; i++) {
        uint8_t a = alpha[i];
        dst[i * 4 + 0] = (dst[i * 4 + 0] * (255 - a) + src[i * 4 + 0] * a) / 255;
        dst[i * 4 + 1] = (dst[i * 4 + 1] * (255 - a) + src[i * 4 + 1] * a) / 255;
        dst[i * 4 + 2] = (dst[i * 4 + 2] * (255 - a) + src[i * 4 + 2] * a) / 255;
        dst[i * 4 + 3] = (dst[i * 4 + 3] * (255 - a) + src[i * 4 + 3] * a) / 255;
    }
}

// Utility macros for performance
#define OBS_LIKELY(x)   __builtin_expect(!!(x), 1)
#define OBS_UNLIKELY(x) __builtin_expect(!!(x), 0)
#define OBS_PREFETCH(addr) __builtin_prefetch(addr, 0, 3)

// Thread-safe counter for statistics
typedef struct {
    _Atomic uint64_t value;
    _Atomic uint64_t updates;
} obs_atomic_counter_t;

static inline void obs_atomic_counter_init(obs_atomic_counter_t *counter) {
    atomic_store(&counter->value, 0);
    atomic_store(&counter->updates, 0);
}

static inline void obs_atomic_counter_increment(obs_atomic_counter_t *counter, uint64_t delta) {
    atomic_fetch_add(&counter->value, delta);
    atomic_fetch_add(&counter->updates, 1);
}

static inline uint64_t obs_atomic_counter_get(obs_atomic_counter_t *counter) {
    return atomic_load(&counter->value);
}

static inline uint64_t obs_atomic_counter_get_updates(obs_atomic_counter_t *counter) {
    return atomic_load(&counter->updates);
}

#ifdef __cplusplus
}
#endif
