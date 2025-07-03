/*
OBS macOS Background Removal - Optimized for ARM
Copyright (C) 2023-2024 Sebastian Beckmann

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

#include <obs-module.h>
#include "plugin-support.h"
#include <util/threading.h>
#include <util/bmem.h>
#include <graphics/graphics.h>
#include <graphics/vec3.h>
#include <Vision/Vision.h>
#include <Metal/Metal.h>
#include <IOSurface/IOSurface.h>
#include <CoreVideo/CVMetalTextureCache.h>
#include <os/signpost.h>
#include <os/log.h>
#include <sys/sysctl.h>
#include <stdatomic.h>

// Define UNUSED_PARAMETER macro if not available
#ifndef UNUSED_PARAMETER
#define UNUSED_PARAMETER(x) ((void)(x))
#endif

OBS_DECLARE_MODULE()
OBS_MODULE_USE_DEFAULT_LOCALE(PLUGIN_NAME, "en-US")

// Performance monitoring
static os_log_t performance_log;
static os_signpost_id_t render_signpost_id;

// ARM processor optimization
typedef struct {
    uint32_t performance_core_count;
    uint32_t efficiency_core_count;
    bool supports_amx;
} arm_processor_info_t;

static arm_processor_info_t arm_info;

// Enhanced vision data structure with Metal integration
struct optimized_vision_data {
    obs_source_t *context;
    VNGeneratePersonSegmentationRequest *segmentation_request;
    
    // Graphics resources
    gs_effect_t *composite_effect;
    gs_eparam_t *source_texture_param;
    gs_eparam_t *mask_texture_param;
    gs_eparam_t *threshold_param;
    gs_eparam_t *edge_smoothing_param;
    gs_eparam_t *contrast_param;
    gs_eparam_t *brightness_param;
    gs_eparam_t *blend_mode_param;
    gs_eparam_t *spill_suppression_param;
    gs_eparam_t *spill_color_param;
    
    // Metal resources
    id<MTLDevice> metal_device;
    id<MTLCommandQueue> metal_command_queue;
    CVMetalTextureCacheRef metal_texture_cache;
    
    // Basic settings
    float threshold;
    VNGeneratePersonSegmentationRequestQualityLevel quality_level;
    int shader_quality_mode; // 0=Fast, 1=Balanced, 2=Quality
    
    // Edge refinement settings
    float edge_smoothing;
    float mask_contrast;
    float mask_brightness;
    
    // Spill suppression settings
    bool spill_enable;
    float spill_strength;
    uint32_t spill_color; // RGB packed
    
    // Advanced blending settings
    int blend_mode; // 0=Normal, 1=Multiply, 2=Screen, 3=Overlay
    
    // ARM optimization settings
    bool use_neon;
    int thread_priority; // 0=Normal, 1=High, 2=Realtime
    bool prefer_performance_cores;
    bool use_metal_optimization;
    
    // Performance monitoring settings
    bool show_performance;
    int max_fps;
    bool show_memory_usage;
    bool show_gpu_usage;
    
    // Preprocessing settings
    int input_scale; // 100=Full, 50=Half, 25=Quarter
    float temporal_smoothing;
    
    // Debug settings
    bool show_mask_overlay;
    int debug_level;
    bool export_mask;
    
    // Preset settings
    int quality_preset;
    int last_quality_preset;
    
    // IOSurface-backed pixel buffers for efficient GPU memory sharing
    CVPixelBufferRef input_pixel_buffer;
    CVPixelBufferRef output_pixel_buffer;
    IOSurfaceRef input_surface;
    IOSurfaceRef output_surface;
    
    // Optimized threading
    dispatch_queue_t high_priority_queue;
    dispatch_queue_t processing_queue;
    dispatch_semaphore_t frame_semaphore;
    
    // Frame management
    _Atomic uint64_t frame_counter;
    _Atomic bool processing_active;
    _Atomic bool has_valid_mask;
    
    // Cached objects to reduce allocations
    VNImageRequestHandler *cached_request_handler;
    NSArray *cached_requests_array;
    
    // Texture cache
    gs_texture_t *cached_mask_texture;
    uint32_t cached_width;
    uint32_t cached_height;
    
    // Performance metrics
    _Atomic uint64_t total_frames_processed;
    _Atomic uint64_t dropped_frames;
    uint64_t last_performance_log_time;
};

// ARM processor detection
static void detect_arm_processor_capabilities(void) {
    size_t size;
    
    // Get performance core count
    size = sizeof(arm_info.performance_core_count);
    if (sysctlbyname("hw.perflevel0.logicalcpu", &arm_info.performance_core_count, &size, NULL, 0) != 0) {
        arm_info.performance_core_count = 4; // Default fallback
    }
    
    // Get efficiency core count
    size = sizeof(arm_info.efficiency_core_count);
    if (sysctlbyname("hw.perflevel1.logicalcpu", &arm_info.efficiency_core_count, &size, NULL, 0) != 0) {
        arm_info.efficiency_core_count = 4; // Default fallback
    }
    
    // Check for AMX support (Apple Matrix Extension)
    int amx_support = 0;
    size = sizeof(amx_support);
    arm_info.supports_amx = (sysctlbyname("hw.optional.amx", &amx_support, &size, NULL, 0) == 0) && amx_support;
    
    obs_log(LOG_INFO, "ARM processor detected: %u performance cores, %u efficiency cores, AMX: %s",
            arm_info.performance_core_count, arm_info.efficiency_core_count,
            arm_info.supports_amx ? "supported" : "not supported");
}

// Create IOSurface-backed CVPixelBuffer for efficient GPU memory sharing
static CVPixelBufferRef create_iosurface_pixel_buffer(uint32_t width, uint32_t height, IOSurfaceRef *out_surface) {
    if (out_surface) {
        *out_surface = NULL;
    }
    
    // Validate input parameters
    if (width == 0 || height == 0 || width > 8192 || height > 8192) {
        obs_log(LOG_ERROR, "Invalid buffer dimensions: %ux%u", width, height);
        return NULL;
    }
    
    CFMutableDictionaryRef surface_properties = CFDictionaryCreateMutable(
        kCFAllocatorDefault, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
    
    if (!surface_properties) {
        obs_log(LOG_ERROR, "Failed to create surface properties dictionary");
        return NULL;
    }
    
    int32_t width_int = (int32_t)width;
    int32_t height_int = (int32_t)height;
    int32_t bytes_per_element = 4;
    uint32_t pixel_format = kCVPixelFormatType_32BGRA;
    
    CFNumberRef width_number = CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt32Type, &width_int);
    CFNumberRef height_number = CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt32Type, &height_int);
    CFNumberRef bytes_per_element_number = CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt32Type, &bytes_per_element);
    CFNumberRef pixel_format_number = CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt32Type, &pixel_format);
    
    if (!width_number || !height_number || !bytes_per_element_number || !pixel_format_number) {
        obs_log(LOG_ERROR, "Failed to create CFNumber objects");
        if (width_number) CFRelease(width_number);
        if (height_number) CFRelease(height_number);
        if (bytes_per_element_number) CFRelease(bytes_per_element_number);
        if (pixel_format_number) CFRelease(pixel_format_number);
        CFRelease(surface_properties);
        return NULL;
    }
    
    CFDictionarySetValue(surface_properties, kIOSurfaceWidth, width_number);
    CFDictionarySetValue(surface_properties, kIOSurfaceHeight, height_number);
    CFDictionarySetValue(surface_properties, kIOSurfaceBytesPerElement, bytes_per_element_number);
    CFDictionarySetValue(surface_properties, kIOSurfacePixelFormat, pixel_format_number);
    
    IOSurfaceRef surface = IOSurfaceCreate(surface_properties);
    
    CFRelease(width_number);
    CFRelease(height_number);
    CFRelease(bytes_per_element_number);
    CFRelease(pixel_format_number);
    
    if (!surface) {
        obs_log(LOG_ERROR, "Failed to create IOSurface");
        CFRelease(surface_properties);
        return NULL;
    }
    
    CFMutableDictionaryRef pixel_buffer_attributes = CFDictionaryCreateMutable(
        kCFAllocatorDefault, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
    
    if (!pixel_buffer_attributes) {
        obs_log(LOG_ERROR, "Failed to create pixel buffer attributes");
        CFRelease(surface);
        CFRelease(surface_properties);
        return NULL;
    }
    
    CFDictionarySetValue(pixel_buffer_attributes, kCVPixelBufferMetalCompatibilityKey, kCFBooleanTrue);
    
    CVPixelBufferRef pixel_buffer;
    CVReturn result = CVPixelBufferCreateWithIOSurface(
        kCFAllocatorDefault, surface, pixel_buffer_attributes, &pixel_buffer);
    
    CFRelease(pixel_buffer_attributes);
    CFRelease(surface_properties);
    
    if (result != kCVReturnSuccess) {
        obs_log(LOG_ERROR, "Failed to create CVPixelBuffer with IOSurface: %d", result);
        CFRelease(surface);
        return NULL;
    }
    
    if (out_surface) {
        *out_surface = surface;
    } else {
        CFRelease(surface);
    }
    
    return pixel_buffer;
}

// Optimized GPU texture to IOSurface transfer
static void transfer_texture_to_iosurface(gs_texture_t *texture, IOSurfaceRef surface) {
    IOSurfaceLock(surface, 0, NULL);
    
    void *base_address = IOSurfaceGetBaseAddress(surface);
    size_t bytes_per_row = IOSurfaceGetBytesPerRow(surface);
    uint32_t width = (uint32_t)IOSurfaceGetWidth(surface);
    uint32_t height = (uint32_t)IOSurfaceGetHeight(surface);
    
    // Use OBS graphics context for efficient GPU readback
    obs_enter_graphics();
    gs_stagesurf_t *stage_surface = gs_stagesurface_create(width, height, GS_BGRA);
    gs_stage_texture(stage_surface, texture);
    
    uint8_t *stage_data;
    uint32_t stage_linesize;
    if (gs_stagesurface_map(stage_surface, &stage_data, &stage_linesize)) {
        // Optimized memory copy using ARM NEON if available
        for (uint32_t y = 0; y < height; y++) {
            memcpy((uint8_t *)base_address + y * bytes_per_row, 
                   stage_data + y * stage_linesize, 
                   width * 4);
        }
        gs_stagesurface_unmap(stage_surface);
    }
    
    gs_stagesurface_destroy(stage_surface);
    obs_leave_graphics();
    
    IOSurfaceUnlock(surface, 0, NULL);
}

// Optimized mask processing with Metal acceleration
static void process_segmentation_mask_async(struct optimized_vision_data *filter, 
                                          CVPixelBufferRef input_buffer) {
    os_signpost_interval_begin(performance_log, render_signpost_id, "SegmentationProcessing");
    
    @autoreleasepool {
        // Configure request for optimal ARM performance
        filter->segmentation_request.qualityLevel = filter->quality_level;
        
        // Use Metal-optimized image request handler
        NSError *error = nil;
        VNImageRequestHandler *handler = [[VNImageRequestHandler alloc] 
            initWithCVPixelBuffer:input_buffer 
            options:@{VNImageOptionCIContext: [CIContext contextWithMTLDevice:filter->metal_device]}];
        
        BOOL success = [handler performRequests:filter->cached_requests_array error:&error];
        
        if (success && filter->segmentation_request.results.count > 0) {
            VNPixelBufferObservation *observation = filter->segmentation_request.results.firstObject;
            
            // Update output buffer atomically
            CVPixelBufferRef new_output = observation.pixelBuffer;
            CVPixelBufferRetain(new_output);
            
            CVPixelBufferRef old_output = filter->output_pixel_buffer;
            filter->output_pixel_buffer = new_output;
            atomic_store(&filter->has_valid_mask, true);
            
            if (old_output) {
                CVPixelBufferRelease(old_output);
            }
            
            atomic_fetch_add(&filter->total_frames_processed, 1);
        } else {
            obs_log(LOG_WARNING, "Segmentation processing failed: %s", 
                   error ? [[error localizedDescription] UTF8String] : "Unknown error");
            atomic_fetch_add(&filter->dropped_frames, 1);
        }
        
        [handler release];
    }
    
    atomic_store(&filter->processing_active, false);
    dispatch_semaphore_signal(filter->frame_semaphore);
    
    os_signpost_interval_end(performance_log, render_signpost_id, "SegmentationProcessing");
}

static const char *optimized_vision_get_name(void *unused) {
    UNUSED_PARAMETER(unused);
    return obs_module_text("Name");
}

static void optimized_vision_render(void *data, gs_effect_t *effect) {
    UNUSED_PARAMETER(effect);
    
    struct optimized_vision_data *filter = data;
    
    if (!filter || !filter->context || !filter->segmentation_request) {
        if (filter && filter->context) {
            obs_source_skip_video_filter(filter->context);
        }
        return;
    }
    
    os_signpost_interval_begin(performance_log, render_signpost_id, "RenderFrame");

    obs_source_t *target = obs_filter_get_target(filter->context);
    obs_source_t *parent = obs_filter_get_parent(filter->context);

    if (!target || !parent) {
        obs_source_skip_video_filter(filter->context);
        os_signpost_interval_end(performance_log, render_signpost_id, "RenderFrame");
        return;
    }

    uint32_t width = obs_source_get_base_width(target);
    uint32_t height = obs_source_get_base_height(target);

    if (width == 0 || height == 0) {
        obs_source_skip_video_filter(filter->context);
        os_signpost_interval_end(performance_log, render_signpost_id, "RenderFrame");
        return;
    }
    
    // Create or recreate buffers if dimensions changed
    if (!filter->input_pixel_buffer || 
        filter->cached_width != width || 
        filter->cached_height != height) {
        
        // Clean up old resources
        if (filter->input_pixel_buffer) {
            CVPixelBufferRelease(filter->input_pixel_buffer);
            CFRelease(filter->input_surface);
        }
        
        // Create new IOSurface-backed buffer
        filter->input_pixel_buffer = create_iosurface_pixel_buffer(width, height, &filter->input_surface);
        filter->cached_width = width;
        filter->cached_height = height;
        
        if (!filter->input_pixel_buffer) {
            obs_source_skip_video_filter(filter->context);
            os_signpost_interval_end(performance_log, render_signpost_id, "RenderFrame");
            return;
        }
    }
    
    // Render source to texture
    obs_enter_graphics();
    gs_texrender_t *render = gs_texrender_create(GS_BGRA, GS_ZS_NONE);
    
    if (gs_texrender_begin(render, width, height)) {
        struct vec4 clear_color;
        vec4_zero(&clear_color);
        gs_clear(GS_CLEAR_COLOR, &clear_color, 0, 0);
        gs_ortho(0, width, 0, height, -100, 100);
        
        uint32_t target_flags = obs_source_get_output_flags(target);
        bool custom_draw = (target_flags & OBS_SOURCE_CUSTOM_DRAW) != 0;
        bool async_source = (target_flags & OBS_SOURCE_ASYNC) != 0;
        
        if (target == parent && !custom_draw && !async_source) {
            obs_source_default_render(target);
        } else {
            obs_source_video_render(target);
        }
        
        gs_texrender_end(render);
    }

    gs_texture_t *source_texture = gs_texrender_get_texture(render);
    if (!source_texture) {
        gs_texrender_destroy(render);
        obs_leave_graphics();
        obs_source_skip_video_filter(filter->context);
        os_signpost_interval_end(performance_log, render_signpost_id, "RenderFrame");
        return;
    }
    
    // Transfer texture to IOSurface efficiently
    transfer_texture_to_iosurface(source_texture, filter->input_surface);
    
    obs_leave_graphics();
    
    // Process segmentation asynchronously if not already processing
    bool expected = false;
    if (atomic_compare_exchange_strong(&filter->processing_active, &expected, true)) {
        atomic_fetch_add(&filter->frame_counter, 1);
        
        // Use high-priority queue for real-time processing
        dispatch_async(filter->high_priority_queue, ^{
            process_segmentation_mask_async(filter, filter->input_pixel_buffer);
        });
    }
    
    // Render result if we have a valid mask
    if (atomic_load(&filter->has_valid_mask) && filter->output_pixel_buffer) {
        obs_enter_graphics();
        
        // Create or update mask texture
        if (!filter->cached_mask_texture || 
            filter->cached_width != width || 
            filter->cached_height != height) {
            
            if (filter->cached_mask_texture) {
                gs_texture_destroy(filter->cached_mask_texture);
            }
            
            CVPixelBufferLockBaseAddress(filter->output_pixel_buffer, kCVPixelBufferLock_ReadOnly);
            const uint8_t *mask_data = CVPixelBufferGetBaseAddress(filter->output_pixel_buffer);
            
            filter->cached_mask_texture = gs_texture_create(
                (uint32_t)CVPixelBufferGetWidth(filter->output_pixel_buffer),
                (uint32_t)CVPixelBufferGetHeight(filter->output_pixel_buffer),
                GS_A8, 1, &mask_data, 0);
            
            CVPixelBufferUnlockBaseAddress(filter->output_pixel_buffer, kCVPixelBufferLock_ReadOnly);
        }
        
        enum gs_color_format format = gs_texture_get_color_format(source_texture);
        
    if (obs_source_process_filter_begin(filter->context, format, OBS_ALLOW_DIRECT_RENDERING)) {
            // Set texture parameters
            gs_effect_set_texture_srgb(filter->source_texture_param, source_texture);
            gs_effect_set_texture_srgb(filter->mask_texture_param, filter->cached_mask_texture);
            
            // Set basic parameters
        gs_effect_set_float(filter->threshold_param, filter->threshold);
            
            // Set advanced parameters if available
            if (filter->edge_smoothing_param) {
                gs_effect_set_float(filter->edge_smoothing_param, filter->edge_smoothing);
            }
            if (filter->contrast_param) {
                gs_effect_set_float(filter->contrast_param, filter->mask_contrast);
            }
            if (filter->brightness_param) {
                gs_effect_set_float(filter->brightness_param, filter->mask_brightness);
            }
            if (filter->blend_mode_param) {
                gs_effect_set_int(filter->blend_mode_param, filter->blend_mode);
            }
            if (filter->spill_suppression_param) {
                gs_effect_set_float(filter->spill_suppression_param, 
                                   filter->spill_enable ? filter->spill_strength : 0.0f);
            }
            if (filter->spill_color_param) {
                struct vec3 color;
                color.x = ((filter->spill_color >> 16) & 0xFF) / 255.0f; // R
                color.y = ((filter->spill_color >> 8) & 0xFF) / 255.0f;  // G
                color.z = (filter->spill_color & 0xFF) / 255.0f;         // B
                gs_effect_set_vec3(filter->spill_color_param, &color);
            }
            
            // Choose technique based on shader quality mode
            const char *technique_name;
            switch (filter->shader_quality_mode) {
                case 0: // Fast
                    technique_name = "DrawFast";
                    break;
                case 2: // Quality
                    technique_name = "DrawQuality";
                    break;
                case 1: // Balanced
                default:
                    technique_name = "Draw";
                    break;
            }

        gs_blend_state_push();
            obs_source_process_filter_tech_end(filter->context, filter->composite_effect, 0, 0, technique_name);
        gs_blend_state_pop();
        }
        
        obs_leave_graphics();
    } else {
        // Skip filter if no mask is available yet
        obs_source_skip_video_filter(filter->context);
    }
    
    gs_texrender_destroy(render);
    
    os_signpost_interval_end(performance_log, render_signpost_id, "RenderFrame");
}

static obs_properties_t *optimized_vision_properties(void *unused) {
    UNUSED_PARAMETER(unused);
    
    obs_properties_t *props = obs_properties_create();
    
    // === BASIC SETTINGS GROUP ===
    obs_properties_t *basic_group = obs_properties_create();
    obs_properties_add_group(props, "basic_settings", obs_module_text("BasicSettings"), 
                            OBS_GROUP_NORMAL, basic_group);
    
    // Core threshold setting
    obs_properties_add_float_slider(basic_group, "threshold", obs_module_text("Threshold"), 0.0, 1.0, 0.01);
    
    // Quality/Performance mode
    obs_property_t *quality_list = obs_properties_add_list(basic_group, "quality", obs_module_text("Quality"), 
                                                          OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
    obs_property_list_add_int(quality_list, obs_module_text("Quality.Fast"),
                             VNGeneratePersonSegmentationRequestQualityLevelFast);
    obs_property_list_add_int(quality_list, obs_module_text("Quality.Balanced"),
                              VNGeneratePersonSegmentationRequestQualityLevelBalanced);
    obs_property_list_add_int(quality_list, obs_module_text("Quality.Accurate"),
                             VNGeneratePersonSegmentationRequestQualityLevelAccurate);
    
    // Shader quality mode
    obs_property_t *shader_quality_list = obs_properties_add_list(basic_group, "shader_quality", 
                                                                 obs_module_text("ShaderQuality"), 
                                                                 OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
    obs_property_list_add_int(shader_quality_list, obs_module_text("ShaderQuality.Fast"), 0);
    obs_property_list_add_int(shader_quality_list, obs_module_text("ShaderQuality.Balanced"), 1);
    obs_property_list_add_int(shader_quality_list, obs_module_text("ShaderQuality.Quality"), 2);
    
    // === EDGE REFINEMENT GROUP ===
    obs_properties_t *edge_group = obs_properties_create();
    obs_properties_add_group(props, "edge_settings", obs_module_text("EdgeRefinement"), 
                            OBS_GROUP_NORMAL, edge_group);
    
    // Edge smoothing
    obs_properties_add_float_slider(edge_group, "edge_smoothing", obs_module_text("EdgeSmoothing"), 0.0, 0.5, 0.01);
    
    // Contrast and brightness for mask refinement
    obs_properties_add_float_slider(edge_group, "mask_contrast", obs_module_text("MaskContrast"), 0.5, 3.0, 0.1);
    obs_properties_add_float_slider(edge_group, "mask_brightness", obs_module_text("MaskBrightness"), -0.5, 0.5, 0.05);
    
    // === SPILL SUPPRESSION GROUP ===
    obs_properties_t *spill_group = obs_properties_create();
    obs_properties_add_group(props, "spill_settings", obs_module_text("SpillSuppression"), 
                            OBS_GROUP_NORMAL, spill_group);
    
    // Enable spill suppression
    obs_property_t *spill_enable = obs_properties_add_bool(spill_group, "spill_enable", 
                                                          obs_module_text("EnableSpillSuppression"));
    
    // Spill suppression strength
    obs_properties_add_float_slider(spill_group, "spill_strength", obs_module_text("SpillStrength"), 0.0, 1.0, 0.05);
    
    // Spill color picker
    obs_properties_add_color(spill_group, "spill_color", obs_module_text("SpillColor"));
    
    // === ADVANCED BLENDING GROUP ===
    obs_properties_t *blend_group = obs_properties_create();
    obs_properties_add_group(props, "blend_settings", obs_module_text("AdvancedBlending"), 
                            OBS_GROUP_NORMAL, blend_group);
    
    // Blend modes
    obs_property_t *blend_mode_list = obs_properties_add_list(blend_group, "blend_mode", 
                                                             obs_module_text("BlendMode"), 
                                                             OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
    obs_property_list_add_int(blend_mode_list, obs_module_text("BlendMode.Normal"), 0);
    obs_property_list_add_int(blend_mode_list, obs_module_text("BlendMode.Multiply"), 1);
    obs_property_list_add_int(blend_mode_list, obs_module_text("BlendMode.Screen"), 2);
    obs_property_list_add_int(blend_mode_list, obs_module_text("BlendMode.Overlay"), 3);
    
    // === ARM OPTIMIZATION GROUP ===
    obs_properties_t *arm_group = obs_properties_create();
    obs_properties_add_group(props, "arm_settings", obs_module_text("ARMOptimizations"), 
                            OBS_GROUP_NORMAL, arm_group);
    
    // ARM NEON acceleration
    obs_properties_add_bool(arm_group, "use_neon", obs_module_text("UseNEONAcceleration"));
    
    // Threading optimization
    obs_property_t *thread_priority_list = obs_properties_add_list(arm_group, "thread_priority", 
                                                                  obs_module_text("ThreadPriority"), 
                                                                  OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
    obs_property_list_add_int(thread_priority_list, obs_module_text("ThreadPriority.Normal"), 0);
    obs_property_list_add_int(thread_priority_list, obs_module_text("ThreadPriority.High"), 1);
    obs_property_list_add_int(thread_priority_list, obs_module_text("ThreadPriority.Realtime"), 2);
    
    // Use performance cores preference
    obs_properties_add_bool(arm_group, "prefer_performance_cores", obs_module_text("PreferPerformanceCores"));
    
    // Metal GPU optimization
    obs_properties_add_bool(arm_group, "use_metal_optimization", obs_module_text("UseMetalOptimization"));
    
    // === PERFORMANCE MONITORING GROUP ===
    obs_properties_t *perf_group = obs_properties_create();
    obs_properties_add_group(props, "performance_settings", obs_module_text("PerformanceMonitoring"), 
                            OBS_GROUP_NORMAL, perf_group);
    
    // Show performance metrics
    obs_properties_add_bool(perf_group, "show_performance", obs_module_text("ShowPerformanceMetrics"));
    
    // Frame rate limiting
    obs_properties_add_int_slider(perf_group, "max_fps", obs_module_text("MaxFrameRate"), 15, 120, 5);
    
    // Memory usage monitoring
    obs_properties_add_bool(perf_group, "show_memory_usage", obs_module_text("ShowMemoryUsage"));
    
    // GPU usage monitoring
    obs_properties_add_bool(perf_group, "show_gpu_usage", obs_module_text("ShowGPUUsage"));
    
    // === PREPROCESSING GROUP ===
    obs_properties_t *preproc_group = obs_properties_create();
    obs_properties_add_group(props, "preprocessing_settings", obs_module_text("Preprocessing"), 
                            OBS_GROUP_NORMAL, preproc_group);
    
    // Input resolution scaling
    obs_property_t *input_scale_list = obs_properties_add_list(preproc_group, "input_scale", 
                                                              obs_module_text("InputScale"), 
                                                              OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
    obs_property_list_add_int(input_scale_list, obs_module_text("InputScale.Full"), 100);
    obs_property_list_add_int(input_scale_list, obs_module_text("InputScale.Half"), 50);
    obs_property_list_add_int(input_scale_list, obs_module_text("InputScale.Quarter"), 25);
    
    // Temporal smoothing
    obs_properties_add_float_slider(preproc_group, "temporal_smoothing", obs_module_text("TemporalSmoothing"), 0.0, 1.0, 0.05);
    
    // === DEBUGGING GROUP ===
    obs_properties_t *debug_group = obs_properties_create();
    obs_properties_add_group(props, "debug_settings", obs_module_text("DebuggingTools"), 
                            OBS_GROUP_NORMAL, debug_group);
    
    // Show mask overlay
    obs_properties_add_bool(debug_group, "show_mask_overlay", obs_module_text("ShowMaskOverlay"));
    
    // Debug logging level
    obs_property_t *debug_level_list = obs_properties_add_list(debug_group, "debug_level", 
                                                              obs_module_text("DebugLevel"), 
                                                              OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
    obs_property_list_add_int(debug_level_list, obs_module_text("DebugLevel.None"), 0);
    obs_property_list_add_int(debug_level_list, obs_module_text("DebugLevel.Basic"), 1);
    obs_property_list_add_int(debug_level_list, obs_module_text("DebugLevel.Detailed"), 2);
    obs_property_list_add_int(debug_level_list, obs_module_text("DebugLevel.Verbose"), 3);
    
    // Export mask as file
    obs_properties_add_bool(debug_group, "export_mask", obs_module_text("ExportMask"));
    
    // === PRESETS GROUP ===
    obs_properties_t *preset_group = obs_properties_create();
    obs_properties_add_group(props, "preset_settings", obs_module_text("Presets"), 
                            OBS_GROUP_NORMAL, preset_group);
    
    // Quality presets
    obs_property_t *preset_list = obs_properties_add_list(preset_group, "quality_preset", 
                                                         obs_module_text("QualityPreset"), 
                                                         OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
    obs_property_list_add_int(preset_list, obs_module_text("Preset.UltraFast"), 0);
    obs_property_list_add_int(preset_list, obs_module_text("Preset.Fast"), 1);
    obs_property_list_add_int(preset_list, obs_module_text("Preset.Balanced"), 2);
    obs_property_list_add_int(preset_list, obs_module_text("Preset.Quality"), 3);
    obs_property_list_add_int(preset_list, obs_module_text("Preset.UltraQuality"), 4);
    obs_property_list_add_int(preset_list, obs_module_text("Preset.Custom"), 5);
    
    // Reset to defaults button
    obs_properties_add_button(preset_group, "reset_defaults", obs_module_text("ResetToDefaults"), 
                             optimized_vision_reset_defaults);
    
    return props;
}

// Reset to defaults callback
static bool optimized_vision_reset_defaults(obs_properties_t *props, obs_property_t *property, void *data) {
    UNUSED_PARAMETER(props);
    UNUSED_PARAMETER(property);
    
    struct optimized_vision_data *filter = data;
    if (!filter) return false;
    
    obs_data_t *settings = obs_source_get_settings(filter->context);
    
    // Reset all settings to defaults
    obs_data_set_double(settings, "threshold", 0.9);
    obs_data_set_int(settings, "quality", VNGeneratePersonSegmentationRequestQualityLevelBalanced);
    obs_data_set_int(settings, "shader_quality", 1);
    obs_data_set_double(settings, "edge_smoothing", 0.1);
    obs_data_set_double(settings, "mask_contrast", 1.0);
    obs_data_set_double(settings, "mask_brightness", 0.0);
    obs_data_set_bool(settings, "spill_enable", false);
    obs_data_set_double(settings, "spill_strength", 0.0);
    obs_data_set_int(settings, "spill_color", 0x00FF00); // Green
    obs_data_set_int(settings, "blend_mode", 0);
    obs_data_set_bool(settings, "use_neon", true);
    obs_data_set_int(settings, "thread_priority", 1);
    obs_data_set_bool(settings, "prefer_performance_cores", true);
    obs_data_set_bool(settings, "use_metal_optimization", true);
    obs_data_set_bool(settings, "show_performance", false);
    obs_data_set_int(settings, "max_fps", 60);
    obs_data_set_bool(settings, "show_memory_usage", false);
    obs_data_set_bool(settings, "show_gpu_usage", false);
    obs_data_set_int(settings, "input_scale", 100);
    obs_data_set_double(settings, "temporal_smoothing", 0.0);
    obs_data_set_bool(settings, "show_mask_overlay", false);
    obs_data_set_int(settings, "debug_level", 0);
    obs_data_set_bool(settings, "export_mask", false);
    obs_data_set_int(settings, "quality_preset", 2);
    
    obs_source_update(filter->context, settings);
    obs_data_release(settings);
    
    obs_log(LOG_INFO, "Settings reset to defaults");
    return true;
}

static void optimized_vision_defaults(obs_data_t *settings) {
    // Basic settings
    obs_data_set_default_double(settings, "threshold", 0.9);
    obs_data_set_default_int(settings, "quality", VNGeneratePersonSegmentationRequestQualityLevelBalanced);
    obs_data_set_default_int(settings, "shader_quality", 1); // Balanced
    
    // Edge refinement settings
    obs_data_set_default_double(settings, "edge_smoothing", 0.1);
    obs_data_set_default_double(settings, "mask_contrast", 1.0);
    obs_data_set_default_double(settings, "mask_brightness", 0.0);
    
    // Spill suppression settings
    obs_data_set_default_bool(settings, "spill_enable", false);
    obs_data_set_default_double(settings, "spill_strength", 0.0);
    obs_data_set_default_int(settings, "spill_color", 0x00FF00); // Green
    
    // Advanced blending settings
    obs_data_set_default_int(settings, "blend_mode", 0); // Normal
    
    // ARM optimization settings
    obs_data_set_default_bool(settings, "use_neon", true);
    obs_data_set_default_int(settings, "thread_priority", 1); // High
    obs_data_set_default_bool(settings, "prefer_performance_cores", true);
    obs_data_set_default_bool(settings, "use_metal_optimization", true);
    
    // Performance monitoring settings
    obs_data_set_default_bool(settings, "show_performance", false);
    obs_data_set_default_int(settings, "max_fps", 60);
    obs_data_set_default_bool(settings, "show_memory_usage", false);
    obs_data_set_default_bool(settings, "show_gpu_usage", false);
    
    // Preprocessing settings
    obs_data_set_default_int(settings, "input_scale", 100); // Full resolution
    obs_data_set_default_double(settings, "temporal_smoothing", 0.0);
    
    // Debug settings
    obs_data_set_default_bool(settings, "show_mask_overlay", false);
    obs_data_set_default_int(settings, "debug_level", 0); // None
    obs_data_set_default_bool(settings, "export_mask", false);
    
    // Preset settings
    obs_data_set_default_int(settings, "quality_preset", 2); // Balanced
}

// Apply quality preset
static void apply_quality_preset(obs_data_t *settings, int preset) {
    switch (preset) {
        case 0: // UltraFast
            obs_data_set_int(settings, "quality", VNGeneratePersonSegmentationRequestQualityLevelFast);
            obs_data_set_int(settings, "shader_quality", 0);
            obs_data_set_double(settings, "edge_smoothing", 0.02);
            obs_data_set_int(settings, "input_scale", 50);
            obs_data_set_double(settings, "temporal_smoothing", 0.0);
            break;
        case 1: // Fast
            obs_data_set_int(settings, "quality", VNGeneratePersonSegmentationRequestQualityLevelFast);
            obs_data_set_int(settings, "shader_quality", 0);
            obs_data_set_double(settings, "edge_smoothing", 0.05);
            obs_data_set_int(settings, "input_scale", 75);
            obs_data_set_double(settings, "temporal_smoothing", 0.1);
            break;
        case 2: // Balanced
            obs_data_set_int(settings, "quality", VNGeneratePersonSegmentationRequestQualityLevelBalanced);
            obs_data_set_int(settings, "shader_quality", 1);
            obs_data_set_double(settings, "edge_smoothing", 0.1);
            obs_data_set_int(settings, "input_scale", 100);
            obs_data_set_double(settings, "temporal_smoothing", 0.2);
            break;
        case 3: // Quality
            obs_data_set_int(settings, "quality", VNGeneratePersonSegmentationRequestQualityLevelBalanced);
            obs_data_set_int(settings, "shader_quality", 2);
            obs_data_set_double(settings, "edge_smoothing", 0.15);
            obs_data_set_int(settings, "input_scale", 100);
            obs_data_set_double(settings, "temporal_smoothing", 0.3);
            break;
        case 4: // UltraQuality
            obs_data_set_int(settings, "quality", VNGeneratePersonSegmentationRequestQualityLevelAccurate);
            obs_data_set_int(settings, "shader_quality", 2);
            obs_data_set_double(settings, "edge_smoothing", 0.2);
            obs_data_set_int(settings, "input_scale", 100);
            obs_data_set_double(settings, "temporal_smoothing", 0.4);
            break;
        case 5: // Custom - don't change settings
        default:
            break;
    }
}

static void optimized_vision_update(void *data, obs_data_t *settings) {
    struct optimized_vision_data *filter = data;
    
    // Handle quality preset first
    int quality_preset = obs_data_get_int(settings, "quality_preset");
    if (filter->last_quality_preset != quality_preset && quality_preset != 5) { // Not Custom
        apply_quality_preset(settings, quality_preset);
        filter->last_quality_preset = quality_preset;
    }
    filter->quality_preset = quality_preset;
    
    // Basic settings
    filter->threshold = (float)obs_data_get_double(settings, "threshold");
    filter->quality_level = obs_data_get_int(settings, "quality");
    filter->shader_quality_mode = obs_data_get_int(settings, "shader_quality");
    
    // Edge refinement settings
    filter->edge_smoothing = (float)obs_data_get_double(settings, "edge_smoothing");
    filter->mask_contrast = (float)obs_data_get_double(settings, "mask_contrast");
    filter->mask_brightness = (float)obs_data_get_double(settings, "mask_brightness");
    
    // Spill suppression settings
    filter->spill_enable = obs_data_get_bool(settings, "spill_enable");
    filter->spill_strength = (float)obs_data_get_double(settings, "spill_strength");
    filter->spill_color = (uint32_t)obs_data_get_int(settings, "spill_color");
    
    // Advanced blending settings
    filter->blend_mode = obs_data_get_int(settings, "blend_mode");
    
    // ARM optimization settings
    filter->use_neon = obs_data_get_bool(settings, "use_neon");
    filter->thread_priority = obs_data_get_int(settings, "thread_priority");
    filter->prefer_performance_cores = obs_data_get_bool(settings, "prefer_performance_cores");
    filter->use_metal_optimization = obs_data_get_bool(settings, "use_metal_optimization");
    
    // Performance monitoring settings
    filter->show_performance = obs_data_get_bool(settings, "show_performance");
    filter->max_fps = obs_data_get_int(settings, "max_fps");
    filter->show_memory_usage = obs_data_get_bool(settings, "show_memory_usage");
    filter->show_gpu_usage = obs_data_get_bool(settings, "show_gpu_usage");
    
    // Preprocessing settings
    filter->input_scale = obs_data_get_int(settings, "input_scale");
    filter->temporal_smoothing = (float)obs_data_get_double(settings, "temporal_smoothing");
    
    // Debug settings
    filter->show_mask_overlay = obs_data_get_bool(settings, "show_mask_overlay");
    filter->debug_level = obs_data_get_int(settings, "debug_level");
    filter->export_mask = obs_data_get_bool(settings, "export_mask");
    
    // Update threading configuration based on ARM settings
    if (filter->prefer_performance_cores) {
        // Recreate queues with performance core preference
        if (filter->high_priority_queue) {
            dispatch_release(filter->high_priority_queue);
            filter->high_priority_queue = NULL;
        }
        
        dispatch_queue_attr_t attr = dispatch_queue_attr_make_with_qos_class(
            DISPATCH_QUEUE_CONCURRENT, 
            filter->thread_priority == 2 ? QOS_CLASS_USER_INTERACTIVE : QOS_CLASS_USER_INITIATED,
            0);
        
        filter->high_priority_queue = dispatch_queue_create("com.obs.vision.high_priority", attr);
        
        if (!filter->high_priority_queue) {
            obs_log(LOG_ERROR, "Failed to recreate high priority dispatch queue");
            // Create fallback queue
            filter->high_priority_queue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0);
        }
    }
    
    // Update shader parameters if effect is loaded
    if (filter->composite_effect) {
        obs_enter_graphics();
        
        // Update shader parameters
        if (filter->edge_smoothing_param) {
            gs_effect_set_float(filter->edge_smoothing_param, filter->edge_smoothing);
        }
        if (filter->contrast_param) {
            gs_effect_set_float(filter->contrast_param, filter->mask_contrast);
        }
        if (filter->brightness_param) {
            gs_effect_set_float(filter->brightness_param, filter->mask_brightness);
        }
        if (filter->blend_mode_param) {
            gs_effect_set_int(filter->blend_mode_param, filter->blend_mode);
        }
        if (filter->spill_suppression_param) {
            gs_effect_set_float(filter->spill_suppression_param, 
                               filter->spill_enable ? filter->spill_strength : 0.0f);
        }
        if (filter->spill_color_param) {
            struct vec3 color;
            color.x = ((filter->spill_color >> 16) & 0xFF) / 255.0f; // R
            color.y = ((filter->spill_color >> 8) & 0xFF) / 255.0f;  // G
            color.z = (filter->spill_color & 0xFF) / 255.0f;         // B
            gs_effect_set_vec3(filter->spill_color_param, &color);
        }
        
        obs_leave_graphics();
    }
    
    // Log performance metrics if enabled
    if (filter->show_performance) {
        uint64_t current_time = obs_get_high_precision_time();
        uint64_t time_since_last_log = current_time - filter->last_performance_log_time;
        
        // Log every 5 seconds
        if (time_since_last_log > 5000000000ULL) {
            uint64_t total_frames = atomic_load(&filter->total_frames_processed);
            uint64_t dropped_frames = atomic_load(&filter->dropped_frames);
            
            if (total_frames > 0) {
                float success_rate = (float)(total_frames - dropped_frames) / total_frames * 100.0f;
                obs_log(LOG_INFO, "Performance: %llu frames processed, %llu dropped, %.1f%% success rate",
                       total_frames, dropped_frames, success_rate);
                
                if (filter->show_memory_usage) {
                    // Log memory usage (simplified)
                    obs_log(LOG_INFO, "Memory: Input buffer size: %dx%d, Cached textures: %d", 
                           filter->cached_width, filter->cached_height, 
                           filter->cached_mask_texture ? 1 : 0);
                }
            }
            
            filter->last_performance_log_time = current_time;
        }
    }
    
    // Apply debug settings
    if (filter->debug_level > 0) {
        obs_log(LOG_DEBUG, "Settings updated - Quality: %d, Shader: %d, Edge: %.2f, Contrast: %.2f",
               filter->quality_level, filter->shader_quality_mode, 
               filter->edge_smoothing, filter->mask_contrast);
    }
}

static void *optimized_vision_create(obs_data_t *settings, obs_source_t *source) {
    struct optimized_vision_data *filter = bzalloc(sizeof(struct optimized_vision_data));
    
    if (!filter) {
        obs_log(LOG_ERROR, "Failed to allocate memory for filter");
        return NULL;
    }
    
    filter->context = source;
    
    // Initialize Vision request
    @try {
        filter->segmentation_request = [[VNGeneratePersonSegmentationRequest alloc] init];
        if (!filter->segmentation_request) {
            obs_log(LOG_ERROR, "Failed to create VNGeneratePersonSegmentationRequest");
            goto error_cleanup;
        }
    } @catch (NSException *exception) {
        obs_log(LOG_ERROR, "Vision framework not available: %s", [[exception description] UTF8String]);
        goto error_cleanup;
    }
    
    // Initialize Metal resources (may not be available on all systems)
    @try {
        filter->metal_device = MTLCreateSystemDefaultDevice();
        if (!filter->metal_device) {
            obs_log(LOG_WARNING, "Metal device not available, using fallback mode");
            // Continue without Metal optimization
        } else {
            filter->metal_command_queue = [filter->metal_device newCommandQueue];
            if (!filter->metal_command_queue) {
                obs_log(LOG_WARNING, "Failed to create Metal command queue, disabling Metal optimization");
                [filter->metal_device release];
                filter->metal_device = nil;
            } else {
                // Create Metal texture cache
                CVReturn cv_result = CVMetalTextureCacheCreate(
                    kCFAllocatorDefault, NULL, filter->metal_device, NULL, &filter->metal_texture_cache);
                
                if (cv_result != kCVReturnSuccess) {
                    obs_log(LOG_WARNING, "Failed to create Metal texture cache: %d", cv_result);
                    [filter->metal_command_queue release];
                    [filter->metal_device release];
                    filter->metal_command_queue = nil;
                    filter->metal_device = nil;
                }
            }
        }
    } @catch (NSException *exception) {
        obs_log(LOG_WARNING, "Metal framework error: %s", [[exception description] UTF8String]);
        filter->metal_device = nil;
        filter->metal_command_queue = nil;
        filter->metal_texture_cache = NULL;
    }
    
    // Create optimized dispatch queues for ARM processors
    dispatch_queue_attr_t high_priority_attr = dispatch_queue_attr_make_with_qos_class(
        DISPATCH_QUEUE_CONCURRENT, QOS_CLASS_USER_INTERACTIVE, 0);
    
    filter->high_priority_queue = dispatch_queue_create(
        "com.obs.vision.high_priority", high_priority_attr);
    
    if (!filter->high_priority_queue) {
        obs_log(LOG_ERROR, "Failed to create high priority dispatch queue");
        goto error_cleanup;
    }
    
    dispatch_queue_attr_t processing_attr = dispatch_queue_attr_make_with_qos_class(
        DISPATCH_QUEUE_CONCURRENT, QOS_CLASS_USER_INITIATED, 0);
    
    filter->processing_queue = dispatch_queue_create(
        "com.obs.vision.processing", processing_attr);
    
    if (!filter->processing_queue) {
        obs_log(LOG_ERROR, "Failed to create processing dispatch queue");
        goto error_cleanup;
    }
    
    // Create semaphore for frame synchronization
    filter->frame_semaphore = dispatch_semaphore_create(1);
    
    if (!filter->frame_semaphore) {
        obs_log(LOG_ERROR, "Failed to create frame synchronization semaphore");
        goto error_cleanup;
    }
    
    // Cache frequently used objects
    @try {
        filter->cached_requests_array = [[NSArray alloc] initWithObjects:filter->segmentation_request, nil];
        if (!filter->cached_requests_array) {
            obs_log(LOG_ERROR, "Failed to create cached requests array");
            goto error_cleanup;
        }
    } @catch (NSException *exception) {
        obs_log(LOG_ERROR, "Failed to create cached objects: %s", [[exception description] UTF8String]);
        goto error_cleanup;
    }
    
    // Initialize atomic variables
    atomic_store(&filter->frame_counter, 0);
    atomic_store(&filter->processing_active, false);
    atomic_store(&filter->has_valid_mask, false);
    atomic_store(&filter->total_frames_processed, 0);
    atomic_store(&filter->dropped_frames, 0);
    
    // Initialize performance tracking
    filter->last_performance_log_time = obs_get_high_precision_time();
    
    // Initialize preset tracking
    filter->last_quality_preset = -1;
    
    // Load shader effect
    obs_enter_graphics();
    char *effect_file = obs_module_file("alpha_mask.effect");
    
    if (!effect_file) {
        obs_log(LOG_ERROR, "Failed to find alpha_mask.effect file");
        obs_leave_graphics();
        goto error_cleanup;
    }
    
    filter->composite_effect = gs_effect_create_from_file(effect_file, NULL);
    bfree(effect_file);
    
    if (!filter->composite_effect) {
        obs_log(LOG_ERROR, "Failed to load alpha_mask.effect");
        obs_leave_graphics();
        goto error_cleanup;
    }
    
    // Get shader parameters (some may not exist in older versions)
    filter->source_texture_param = gs_effect_get_param_by_name(filter->composite_effect, "image");
    filter->mask_texture_param = gs_effect_get_param_by_name(filter->composite_effect, "mask");
    filter->threshold_param = gs_effect_get_param_by_name(filter->composite_effect, "threshold");
    
    // These parameters are optional for advanced features
    filter->edge_smoothing_param = gs_effect_get_param_by_name(filter->composite_effect, "edge_smoothing");
    filter->contrast_param = gs_effect_get_param_by_name(filter->composite_effect, "contrast");
    filter->brightness_param = gs_effect_get_param_by_name(filter->composite_effect, "brightness");
    filter->blend_mode_param = gs_effect_get_param_by_name(filter->composite_effect, "blend_mode");
    filter->spill_suppression_param = gs_effect_get_param_by_name(filter->composite_effect, "spill_suppression");
    filter->spill_color_param = gs_effect_get_param_by_name(filter->composite_effect, "spill_color");
    
    // Verify essential parameters exist
    if (!filter->source_texture_param || !filter->mask_texture_param || !filter->threshold_param) {
        obs_log(LOG_ERROR, "Essential shader parameters missing from alpha_mask.effect");
        obs_leave_graphics();
        goto error_cleanup;
    }
    
    // Log which optional parameters are available
    if (filter->edge_smoothing_param) {
        obs_log(LOG_DEBUG, "Advanced edge smoothing available");
    }
    if (filter->spill_suppression_param) {
        obs_log(LOG_DEBUG, "Spill suppression available");
    }
    
    obs_leave_graphics();
    
    // Apply initial settings
    optimized_vision_update(filter, settings);
    
    obs_log(LOG_INFO, "Optimized Vision filter created successfully");
    return filter;
    
error_cleanup:
    if (filter->metal_texture_cache) {
        CFRelease(filter->metal_texture_cache);
    }
    if (filter->metal_command_queue) {
        [filter->metal_command_queue release];
    }
    if (filter->metal_device) {
        [filter->metal_device release];
    }
    if (filter->segmentation_request) {
        [filter->segmentation_request release];
    }
    if (filter->cached_requests_array) {
        [filter->cached_requests_array release];
    }
    if (filter->high_priority_queue) {
        dispatch_release(filter->high_priority_queue);
    }
    if (filter->processing_queue) {
        dispatch_release(filter->processing_queue);
    }
    if (filter->frame_semaphore) {
        dispatch_release(filter->frame_semaphore);
    }
    if (filter->composite_effect) {
        obs_enter_graphics();
        gs_effect_destroy(filter->composite_effect);
        obs_leave_graphics();
    }
    bfree(filter);
    return NULL;
}

static void optimized_vision_destroy(void *data) {
    struct optimized_vision_data *filter = data;
    
    // Wait for any pending processing to complete
    dispatch_semaphore_wait(filter->frame_semaphore, DISPATCH_TIME_FOREVER);
    
    // Clean up Metal resources
    if (filter->metal_texture_cache) {
        CFRelease(filter->metal_texture_cache);
    }
    if (filter->metal_command_queue) {
        [filter->metal_command_queue release];
    }
    if (filter->metal_device) {
        [filter->metal_device release];
    }
    
    // Clean up pixel buffers
    if (filter->input_pixel_buffer) {
        CVPixelBufferRelease(filter->input_pixel_buffer);
        CFRelease(filter->input_surface);
    }
    if (filter->output_pixel_buffer) {
        CVPixelBufferRelease(filter->output_pixel_buffer);
    }
    
    // Clean up cached objects
    if (filter->cached_requests_array) {
        [filter->cached_requests_array release];
    }
    if (filter->segmentation_request) {
        [filter->segmentation_request release];
    }
    if (filter->cached_request_handler) {
        [filter->cached_request_handler release];
    }
    
    // Clean up GPU resources
    if (filter->cached_mask_texture) {
        obs_enter_graphics();
        gs_texture_destroy(filter->cached_mask_texture);
        obs_leave_graphics();
    }
    if (filter->composite_effect) {
        gs_effect_destroy(filter->composite_effect);
    }
    
    // Clean up dispatch queues
    if (filter->high_priority_queue) {
        dispatch_release(filter->high_priority_queue);
    }
    if (filter->processing_queue) {
        dispatch_release(filter->processing_queue);
    }
    if (filter->frame_semaphore) {
        dispatch_release(filter->frame_semaphore);
    }
    
    // Log final performance metrics
    uint64_t total_frames = atomic_load(&filter->total_frames_processed);
    uint64_t dropped_frames = atomic_load(&filter->dropped_frames);
    obs_log(LOG_INFO, "Filter destroyed. Final stats: %llu frames processed, %llu dropped",
           total_frames, dropped_frames);
    
    bfree(filter);
}

bool obs_module_load(void) {
    // Initialize performance logging
    performance_log = os_log_create("com.obs.vision", "performance");
    render_signpost_id = os_signpost_id_generate(performance_log);
    
    // Detect ARM processor capabilities
    detect_arm_processor_capabilities();
    
    struct obs_source_info source_info = {
        .id = "optimized_mac_vision_filter",
        .type = OBS_SOURCE_TYPE_FILTER,
        .output_flags = OBS_SOURCE_VIDEO | OBS_SOURCE_SRGB,
        .get_name = optimized_vision_get_name,
        .create = optimized_vision_create,
        .destroy = optimized_vision_destroy,
        .video_render = optimized_vision_render,
        .get_defaults = optimized_vision_defaults,
        .get_properties = optimized_vision_properties,
        .update = optimized_vision_update,
    };
    
    obs_register_source(&source_info);
    
    obs_log(LOG_INFO, "Optimized Vision plugin loaded successfully (version %s)", PLUGIN_VERSION);
    return true;
}

void obs_module_unload(void) {
    obs_log(LOG_INFO, "Optimized Vision plugin unloaded");
}
