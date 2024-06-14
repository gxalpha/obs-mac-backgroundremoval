/*
OBS macOS Background Removal
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
#include <plugin-support.h>
#include <util/threading.h>
#include <Vision/Vision.h>

OBS_DECLARE_MODULE()
OBS_MODULE_USE_DEFAULT_LOCALE(PLUGIN_NAME, "en-US")

struct vision_data {
    obs_source_t *context;
    VNGeneratePersonSegmentationRequest *request;
    gs_effect_t *effect;
    gs_eparam_t *src_param;
    gs_eparam_t *mask_param;
    gs_eparam_t *threshold_param;
    float threshold;

    VNGeneratePersonSegmentationRequestQualityLevel qualityLevel;
    gs_texture_t *mask_texture;

    dispatch_queue_t mask_queue;
    pthread_mutex_t pixelBufferMutex;
    CVPixelBufferRef pixelBufferOut;
};

static const char *vision_get_name(void *)
{
    return obs_module_text("Name");
}

static void vision_render(void *filter_ptr, gs_effect_t *)
{
    struct vision_data *filter = filter_ptr;

    /* STEP ZERO: Prepare texrender */
    obs_enter_graphics();
    gs_texrender_t *render = gs_texrender_create(GS_BGRA, GS_ZS_NONE);
    obs_leave_graphics();

    obs_source_t *target = obs_filter_get_target(filter->context);
    obs_source_t *parent = obs_filter_get_parent(filter->context);

    uint32_t target_flags = obs_source_get_output_flags(target);

    bool custom_draw = (target_flags & OBS_SOURCE_CUSTOM_DRAW) != 0;
    bool async_source = (target_flags & OBS_SOURCE_ASYNC) != 0;

    uint32_t width = obs_source_get_base_width(target);
    uint32_t height = obs_source_get_base_height(target);

    /* STEP ONE: Retrieve texture */
    gs_blend_state_push();
    gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);
    if (gs_texrender_begin(render, width, height)) {
        struct vec4 clear_color;
        vec4_zero(&clear_color);
        gs_clear(GS_CLEAR_COLOR, &clear_color, 0, 0);
        gs_ortho(0, width, 0, height, -100, 100);
        if (target == parent && !custom_draw && !async_source) {
            obs_source_default_render(target);
        } else {
            obs_source_video_render(target);
        }
        gs_texrender_end(render);
    }
    gs_blend_state_pop();

    /* STEP TWO: Get source texture */
    gs_texture_t *source_texture = gs_texrender_get_texture(render);
    if (!source_texture) {
        gs_texrender_destroy(render);
        obs_source_skip_video_filter(filter->context);
        return;
    }
    enum gs_color_format format = gs_texture_get_color_format(source_texture);

    /* STEP THREE: Creation of new mask */
    /* STEP THREE point one: Create new pixel buffer from source texture */
    gs_stagesurf_t *stagesurf = gs_stagesurface_create(width, height, format);
    gs_stage_texture(stagesurf, source_texture);
    uint8_t *data;
    uint32_t linesize;
    gs_stagesurface_map(stagesurf, &data, &linesize);
    gs_stagesurface_destroy(stagesurf);
    CVPixelBufferRef pixelBufferIn;
    CVPixelBufferCreateWithBytes(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, data, linesize, nil,
                                 nil, nil, &pixelBufferIn);

    /* STEP THREE point two: Dispatch creation of new mask */
    dispatch_async(filter->mask_queue, ^{
        filter->request.qualityLevel = filter->qualityLevel;

        NSDictionary *empty = [[NSDictionary alloc] init];
        VNImageRequestHandler *handler = [[VNImageRequestHandler alloc] initWithCVPixelBuffer:pixelBufferIn
                                                                                      options:empty];

        NSArray *requests = [[NSArray alloc] initWithObjects:filter->request, nil];
        [handler performRequests:requests error:nil];
        pthread_mutex_lock(&filter->pixelBufferMutex);
        if (filter->pixelBufferOut)
            CVPixelBufferRelease(filter->pixelBufferOut);
        filter->pixelBufferOut = filter->request.results.firstObject.pixelBuffer;
        CVPixelBufferRetain(filter->pixelBufferOut);
        pthread_mutex_unlock(&filter->pixelBufferMutex);
        CVPixelBufferRelease(pixelBufferIn);
        [empty release];
        [handler release];
        [requests release];
    });

    /* Don't render if the output pixel buffer doesn't exist, like before the first frame is processed */
    if (!filter->pixelBufferOut) {
        obs_source_skip_video_filter(filter->context);
        gs_texrender_destroy(render);
        return;
    }

    /* STEP FOUR: Retrieve new mask texture */
    /* STEP FOUR point one: Destroy old mask texture */
    if (filter->mask_texture)
        gs_texture_destroy(filter->mask_texture);

    /* STEP FOUR point two: Get mask texture from pixel buffer */
    pthread_mutex_lock(&filter->pixelBufferMutex);
    CVPixelBufferLockBaseAddress(filter->pixelBufferOut, kCVPixelBufferLock_ReadOnly);
    const uint8_t *base_address = CVPixelBufferGetBaseAddress(filter->pixelBufferOut);
    filter->mask_texture = gs_texture_create((uint32_t) CVPixelBufferGetWidth(filter->pixelBufferOut),
                                             (uint32_t) CVPixelBufferGetHeight(filter->pixelBufferOut), GS_A8, 1,
                                             &base_address, 0);
    CVPixelBufferUnlockBaseAddress(filter->pixelBufferOut, kCVPixelBufferLock_ReadOnly);
    pthread_mutex_unlock(&filter->pixelBufferMutex);

    /* STEP FIVE: Render result */
    if (obs_source_process_filter_begin(filter->context, format, OBS_ALLOW_DIRECT_RENDERING)) {
        gs_effect_set_texture_srgb(filter->src_param, source_texture);
        gs_effect_set_texture_srgb(filter->mask_param, filter->mask_texture);
        gs_effect_set_float(filter->threshold_param, filter->threshold);

        gs_blend_state_push();
        obs_source_process_filter_tech_end(filter->context, filter->effect, 0, 0, "Draw");
        gs_blend_state_pop();
    }
    gs_texrender_destroy(render);
}

static obs_properties_t *vision_properties(void *)
{
    obs_properties_t *props = obs_properties_create();
    obs_properties_add_float_slider(props, "threshold", obs_module_text("Threshold"), 0, 1, 0.05);
    obs_property_t *list = obs_properties_add_list(props, "quality", obs_module_text("Quality"), OBS_COMBO_TYPE_LIST,
                                                   OBS_COMBO_FORMAT_INT);
    // "Accurate" is currently broken. The resulting mask is fine, but the image when rendered isn't.
    // obs_property_list_add_int(list, obs_module_text("Quality.Accurate"),
    //			  VNGeneratePersonSegmentationRequestQualityLevelAccurate);
    obs_property_list_add_int(list, obs_module_text("Quality.Balanced"),
                              VNGeneratePersonSegmentationRequestQualityLevelBalanced);
    obs_property_list_add_int(list, obs_module_text("Quality.Fast"),
                              VNGeneratePersonSegmentationRequestQualityLevelFast);
    return props;
}

static void vision_defaults(obs_data_t *settings)
{
    obs_data_set_default_double(settings, "threshold", 0.9);
    obs_data_set_default_int(settings, "quality", VNGeneratePersonSegmentationRequestQualityLevelBalanced);
}

static void vision_update(void *filter_ptr, obs_data_t *settings)
{
    struct vision_data *filter = filter_ptr;

    filter->threshold = obs_data_get_double(settings, "threshold");
    filter->qualityLevel = obs_data_get_int(settings, "quality");
}

static void *vision_create(obs_data_t *settings, struct obs_source *source)
{
    struct vision_data *filter = bzalloc(sizeof(struct vision_data));
    filter->context = source;
    filter->request = [[VNGeneratePersonSegmentationRequest alloc] init];
    pthread_mutex_init(&filter->pixelBufferMutex, NULL);

    /* Performing the segmentation in realtime takes too long and will lag OBS, especially at higher
	 * quality modes. As such, defer it to a different thread. This will make the mask lag behind a few
	 * frames, but is better than lagging the graphics thread. */
    filter->mask_queue = dispatch_queue_create("Filter mask dispatch queue", NULL);

    obs_enter_graphics();
    char *file = obs_module_file("alpha_mask.effect");
    filter->effect = gs_effect_create_from_file(file, NULL);
    bfree(file);
    filter->src_param = gs_effect_get_param_by_name(filter->effect, "image");
    filter->mask_param = gs_effect_get_param_by_name(filter->effect, "mask");
    filter->threshold_param = gs_effect_get_param_by_name(filter->effect, "threshold");
    obs_leave_graphics();
    vision_update(filter, settings);
    return filter;
}

static void vision_destroy(void *filter_ptr)
{
    struct vision_data *filter = filter_ptr;

    if (filter->mask_texture) {
        obs_enter_graphics();
        gs_texture_destroy(filter->mask_texture);
        obs_leave_graphics();
    }
    gs_effect_destroy(filter->effect);
    bfree(filter);
};

bool obs_module_load(void)
{
    struct obs_source_info info = {
        .id = "mac_vision_filter",
        .type = OBS_SOURCE_TYPE_FILTER,
        .output_flags = OBS_SOURCE_VIDEO | OBS_SOURCE_SRGB,
        .get_name = vision_get_name,
        .create = vision_create,
        .destroy = vision_destroy,
        .video_render = vision_render,
        .get_defaults = vision_defaults,
        .get_properties = vision_properties,
        .update = vision_update,
    };
    obs_register_source(&info);
    obs_log(LOG_INFO, "Loaded successfully (version %s)", PLUGIN_VERSION);
    return true;
}
