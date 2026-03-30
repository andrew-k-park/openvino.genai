// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/qwen3_vl/classes.hpp"
#include "utils.hpp"

// ===== TTFT Detailed Profiling for Qwen3VL =====
#include <iostream>
#include <iomanip>
#include <cstdlib>
static bool s_qwen3vl_profile_enabled = []() {
    const char* env = std::getenv("OV_VLM_TTFT_PROFILE");
    return env && std::string(env) == "1";
}();

struct Qwen3VLBlockProfiler {
    using Clock = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;
    using Duration = std::chrono::duration<double, std::milli>;

    // get_inputs_embeds breakdown
    double grid_thw_prep_ms = 0;
    double tokenization_ms = 0;        // CPU: get_encoded_input_ids
    double text_embed_infer_ms = 0;    // GPU: m_embedding->infer
    double position_ids_ms = 0;        // CPU: create_position_ids
    double merger_ms = 0;              // GPU+CPU: run_video_image_embeddings_merger
    double visual_pos_masks_ms = 0;    // CPU: create_visual_pos_masks
    double merge_text_vision_ms = 0;   // CPU: merge_text_and_video_image_embeddings
    double no_vision_setup_ms = 0;     // CPU: empty vision path
    double total_get_inputs_ms = 0;

    // run_video_image_embeddings_merger breakdown
    double reorder_embeds_ms = 0;      // CPU
    double concat_embeds_ms = 0;       // CPU
    double interp_pos_embeds_ms = 0;   // GPU+CPU (vision_embeddings_pos infer + weighted sum + permute)
    double pos_embeds_addition_ms = 0; // CPU
    double rotary_pos_emb_ms = 0;      // CPU
    double merger_set_tensors_ms = 0;  // CPU/GPU: set_tensor + get_attention_mask/cu_seqlens
    double merger_infer_ms = 0;        // GPU: vision_embeddings_merger.infer()
    double split_embeds_ms = 0;        // CPU
    double total_merger_ms = 0;

    // get_interpolated_pos_embeds breakdown
    double interp_indices_ms = 0;      // CPU
    double pos_set_tensor_ms = 0;      // CPU/GPU: set_tensor for pos model
    double pos_model_infer_ms = 0;     // GPU: vision_embeddings_pos.infer()
    double weighted_sum_ms = 0;        // CPU
    double permute_ms = 0;             // CPU
    double total_interp_ms = 0;

    void print_get_inputs_embeds() const {
        double gpu_ms = text_embed_infer_ms + merger_ms;
        double cpu_ms = grid_thw_prep_ms + tokenization_ms + position_ids_ms + visual_pos_masks_ms + merge_text_vision_ms + no_vision_setup_ms;

        std::cerr << "\n--- [Qwen3VL] get_inputs_embeds breakdown ---\n";
        std::cerr << std::fixed << std::setprecision(2);
        std::cerr << "  Total:                        " << total_get_inputs_ms << " ms\n";
        std::cerr << "  [CPU]  grid_thw_prep:          " << grid_thw_prep_ms << " ms\n";
        std::cerr << "  [CPU]  tokenization:           " << tokenization_ms << " ms\n";
        std::cerr << "  [GPU]  text_embed_infer:       " << text_embed_infer_ms << " ms\n";
        std::cerr << "  [CPU]  create_position_ids:    " << position_ids_ms << " ms\n";
        if (no_vision_setup_ms > 0) {
            std::cerr << "  [CPU]  no_vision_setup:        " << no_vision_setup_ms << " ms\n";
        }
        std::cerr << "  [GPU+CPU] merger_total:        " << merger_ms << " ms\n";
        std::cerr << "  [CPU]  visual_pos_masks:       " << visual_pos_masks_ms << " ms\n";
        std::cerr << "  [CPU]  merge_text_vision:      " << merge_text_vision_ms << " ms\n";
        std::cerr << "  ---- Summary ----\n";
        std::cerr << "  GPU total (approx):            " << gpu_ms << " ms (" << (total_get_inputs_ms > 0 ? gpu_ms/total_get_inputs_ms*100 : 0) << "%)\n";
        std::cerr << "  CPU total (approx):            " << cpu_ms << " ms (" << (total_get_inputs_ms > 0 ? cpu_ms/total_get_inputs_ms*100 : 0) << "%)\n";
        std::cerr << "-------------------------------------------\n";
    }

    void print_merger() const {
        double gpu_ms = merger_infer_ms + pos_model_infer_ms;
        double cpu_ms = reorder_embeds_ms + concat_embeds_ms + (interp_pos_embeds_ms - pos_model_infer_ms - pos_set_tensor_ms) + pos_embeds_addition_ms + rotary_pos_emb_ms + merger_set_tensors_ms + split_embeds_ms + pos_set_tensor_ms;
        if (cpu_ms < 0) cpu_ms = 0;

        std::cerr << "\n--- [Qwen3VL] run_video_image_embeddings_merger breakdown ---\n";
        std::cerr << std::fixed << std::setprecision(2);
        std::cerr << "  Total:                        " << total_merger_ms << " ms\n";
        std::cerr << "  [CPU]  reorder_embeds:         " << reorder_embeds_ms << " ms\n";
        std::cerr << "  [CPU]  concat_embeds:          " << concat_embeds_ms << " ms\n";
        std::cerr << "  [GPU+CPU] interp_pos_embeds:   " << interp_pos_embeds_ms << " ms\n";
        std::cerr << "     [CPU]  interp_indices:      " << interp_indices_ms << " ms\n";
        std::cerr << "     [CPU]  pos_set_tensor:      " << pos_set_tensor_ms << " ms\n";
        std::cerr << "     [GPU]  pos_model_infer:     " << pos_model_infer_ms << " ms\n";
        std::cerr << "     [CPU]  weighted_sum:        " << weighted_sum_ms << " ms\n";
        std::cerr << "     [CPU]  permute:             " << permute_ms << " ms\n";
        std::cerr << "  [CPU]  pos_embeds_addition:    " << pos_embeds_addition_ms << " ms\n";
        std::cerr << "  [CPU]  rotary_pos_emb:         " << rotary_pos_emb_ms << " ms\n";
        std::cerr << "  [CPU]  merger_set_tensors:     " << merger_set_tensors_ms << " ms\n";
        std::cerr << "  [GPU]  merger_infer:           " << merger_infer_ms << " ms\n";
        std::cerr << "  [CPU]  split_embeds:           " << split_embeds_ms << " ms\n";
        std::cerr << "  ---- Summary ----\n";
        std::cerr << "  GPU total:                     " << gpu_ms << " ms (" << (total_merger_ms > 0 ? gpu_ms/total_merger_ms*100 : 0) << "%)\n";
        std::cerr << "  CPU total:                     " << cpu_ms << " ms (" << (total_merger_ms > 0 ? cpu_ms/total_merger_ms*100 : 0) << "%)\n";
        std::cerr << "-------------------------------------------\n";
    }
};

// Thread-local to pass profiler from get_inputs_embeds to sub-methods
static thread_local Qwen3VLBlockProfiler* s_current_profiler = nullptr;
// ===== End Qwen3VL Profiling =====

namespace ov::genai {

namespace {

/**
 * @brief Calculates timestamps for video frames based on encoded video metadata.
 * @return Vector of float timestamps corresponding to each video frame.
 */
std::vector<float> calculate_timestamps(const VideoMetadata& video_metadata, size_t merge_size) {
    OPENVINO_ASSERT(video_metadata.fps > 0.0f, "Video metadata fps must be positive for timestamp calculation.");

    // Copy frame_indices since padding may be needed
    std::vector<size_t> frame_indices = video_metadata.frames_indices;
    if (frame_indices.size() % merge_size != 0) {
        frame_indices.resize(
            frame_indices.size() + (merge_size - frame_indices.size() % merge_size),
            frame_indices.back()
        );
    }

    std::vector<float> timestamps;
    timestamps.reserve(frame_indices.size() / merge_size);
    for (size_t i = 0; i < frame_indices.size(); i += merge_size) {
        const float timestamp = (static_cast<float>(frame_indices[i] + frame_indices[i + merge_size - 1]))
            / 2.0f / video_metadata.fps;
        timestamps.push_back(timestamp);
    }
    return timestamps;
}

/**
 * @brief Populates video metadata in encoded_video struct.
 * Computes frame sampling indices for encoded video based on video processor config.
 */
void fill_video_metadata(EncodedVideo& encoded_video, size_t total_num_frames, const VideoProcessorConfig& video_config) {
    OPENVINO_ASSERT(!(video_config.fps != 0.0f && video_config.num_frames != 0),
        "num_frames and fps are mutually exclusive video config arguments.");
    
    if (!video_config.do_sample_frames) {
        encoded_video.metadata.frames_indices.resize(total_num_frames);
        std::iota(encoded_video.metadata.frames_indices.begin(), encoded_video.metadata.frames_indices.end(), 0);
        return;
    }
    // Sample frame indices if needed
    size_t num_frames = video_config.num_frames;
    
    if (num_frames == 0 && video_config.fps != 0.0f) {
        OPENVINO_ASSERT(encoded_video.metadata.fps != 0.0f,
            "Requested to sample frames by fps but video metadata fps is not set. "
            "Provide VideoMetadata with fps or use a fixed num_frames.");

        num_frames = static_cast<size_t>(
            total_num_frames / static_cast<double>(encoded_video.metadata.fps) * static_cast<double>(video_config.fps)
        );
        num_frames = std::clamp(num_frames, video_config.min_frames, std::min(video_config.max_frames, total_num_frames));
    } else if (num_frames == 0) {
        num_frames = std::clamp(total_num_frames, video_config.min_frames, video_config.max_frames);
    }

    OPENVINO_ASSERT(num_frames > 1 && num_frames <= total_num_frames,
        "Invalid number of frames (" + std::to_string(num_frames) +") for video sampling.");

    encoded_video.metadata.frames_indices.reserve(num_frames);
    for (size_t i = 0; i < num_frames; ++i) {
        size_t frame_idx = static_cast<size_t>(std::round(
            static_cast<double>(i) * static_cast<double>(total_num_frames - 1) / static_cast<double>(num_frames - 1)
        ));
        encoded_video.metadata.frames_indices.push_back(frame_idx);
    }
}

/**
 * @brief Computes indices and weights for bilinear position embedding interpolation.
 * @return Pair of:
 *   - indices tensor [4, num_positions] - input for vision_embeddings_pos model
 *   - weights tensor [4, num_positions] - bilinear interpolation weights
 */
std::pair<ov::Tensor, ov::Tensor> get_position_interpolation_indices_and_weights(
    const std::vector<std::array<size_t, 3>>& grids_thw,
    size_t num_grid_per_side
) {
    std::vector<std::vector<int64_t>> indices_list(4);
    std::vector<std::vector<float>> weights_list(4);
    
    for (const auto& grid_thw : grids_thw) {
        const auto [t, h, w] = grid_thw;
        
        // Create linearly spaced indices for h and w
        std::vector<float> h_idxs(h), w_idxs(w);
        const float h_scale = h > 1 ? static_cast<float>(num_grid_per_side - 1) / (h - 1) : 0.0f;
        const float w_scale = w > 1 ? static_cast<float>(num_grid_per_side - 1) / (w - 1) : 0.0f;
        
        for (size_t i = 0; i < h; ++i) {
            h_idxs[i] = static_cast<float>(i) * h_scale;
        }
        for (size_t i = 0; i < w; ++i) {
            w_idxs[i] = static_cast<float>(i) * w_scale;
        }
        
        // Compute floor/ceil indices and interpolation weights
        for (size_t ti = 0; ti < t; ++ti) {
            for (size_t hi = 0; hi < h; ++hi) {
                const int64_t h_floor = static_cast<int64_t>(h_idxs[hi]);
                const int64_t h_ceil = std::min(h_floor + 1, static_cast<int64_t>(num_grid_per_side - 1));
                const float dh = h_idxs[hi] - static_cast<float>(h_floor);
                
                for (size_t wi = 0; wi < w; ++wi) {
                    const int64_t w_floor = static_cast<int64_t>(w_idxs[wi]);
                    const int64_t w_ceil = std::min(w_floor + 1, static_cast<int64_t>(num_grid_per_side - 1));
                    const float dw = w_idxs[wi] - static_cast<float>(w_floor);
                    
                    // 4 corners: (floor,floor), (floor,ceil), (ceil,floor), (ceil,ceil)
                    indices_list[0].push_back(h_floor * num_grid_per_side + w_floor);
                    indices_list[1].push_back(h_floor * num_grid_per_side + w_ceil);
                    indices_list[2].push_back(h_ceil * num_grid_per_side + w_floor);
                    indices_list[3].push_back(h_ceil * num_grid_per_side + w_ceil);
                    
                    // Bilinear weights
                    weights_list[0].push_back((1.0f - dh) * (1.0f - dw));
                    weights_list[1].push_back((1.0f - dh) * dw);
                    weights_list[2].push_back(dh * (1.0f - dw));
                    weights_list[3].push_back(dh * dw);
                }
            }
        }
    }
    
    const size_t total_positions = indices_list[0].size();
    ov::Tensor indices{ov::element::i64, {4, total_positions}};
    ov::Tensor weights{ov::element::f32, {4, total_positions}};
    
    int64_t* indices_data = indices.data<int64_t>();
    float* weights_data = weights.data<float>();
    
    for (size_t corner = 0; corner < 4; ++corner) {
        std::memcpy(indices_data + corner * total_positions,
                    indices_list[corner].data(),
                    total_positions * sizeof(int64_t));
        std::memcpy(weights_data + corner * total_positions,
                    weights_list[corner].data(),
                    total_positions * sizeof(float));
    }
    
    return {indices, weights};
}

/**
 * @brief Reorders position embeddings according to spatial merge pattern in vision encoder.
 * 
 * @param pos_embeds Interpolated position embeddings [num_positions, embed_dim]
 * @param grids_thw Grid dimensions for permutation
 * @param spatial_merge_size Spatial merge size from processor config
 * @return Permuted position embeddings [num_merged_positions, embed_dim]
 */
ov::Tensor permute_with_spatial_merge(
    const ov::Tensor& pos_embeds,
    const std::vector<std::array<size_t, 3>>& grids_thw,
    size_t spatial_merge_size
) {
    const size_t num_positions = pos_embeds.get_shape()[0];
    const size_t embed_dim = pos_embeds.get_shape()[1];
    const float* pos_embeds_data = pos_embeds.data<const float>();
    
    std::vector<float> permuted_data;
    permuted_data.reserve(num_positions * embed_dim);
    
    size_t offset = 0;
    for (const auto& grid_thw : grids_thw) {
        const auto [t, h, w] = grid_thw;
        const size_t hw = h * w;
        
        const size_t merge_h = h / spatial_merge_size;
        const size_t merge_w = w / spatial_merge_size;
        
        for (size_t ti = 0; ti < t; ++ti) {
            for (size_t mhi = 0; mhi < merge_h; ++mhi) {
                for (size_t mwi = 0; mwi < merge_w; ++mwi) {
                    for (size_t shi = 0; shi < spatial_merge_size; ++shi) {
                        for (size_t swi = 0; swi < spatial_merge_size; ++swi) {
                            const size_t src_h = mhi * spatial_merge_size + shi;
                            const size_t src_w = mwi * spatial_merge_size + swi;
                            const size_t src_idx = offset + ti * hw + src_h * w + src_w;
                            
                            const float* src = pos_embeds_data + src_idx * embed_dim;
                            permuted_data.insert(permuted_data.end(), src, src + embed_dim);
                        }
                    }
                }
            }
        }
        offset += t * hw;
    }
    
    const size_t permuted_len = permuted_data.size() / embed_dim;
    ov::Tensor result{ov::element::f32, {permuted_len, embed_dim}};
    std::memcpy(result.data<float>(), permuted_data.data(), permuted_data.size() * sizeof(float));
    
    return result;
}

/**
 * @brief Create visual position mask from input_ids by finding vision pad tokens.
 * @return Boolean tensor [batch, seq_len] with true at vision token positions
 */
ov::Tensor create_visual_pos_masks(
    const ov::Tensor& input_ids,
    int64_t image_pad_token_id,
    int64_t video_pad_token_id
) {
    const auto input_ids_shape = input_ids.get_shape();
    ov::Tensor result{ov::element::boolean, input_ids_shape};
    bool* result_data = result.data<bool>();
    const int64_t* input_ids_data = input_ids.data<const int64_t>();
    for (size_t i = 0; i < ov::shape_size(input_ids_shape); ++i) {
        result_data[i] = (input_ids_data[i] == image_pad_token_id || input_ids_data[i] == video_pad_token_id);
    }
    return result;
}

} // namespace

EncodedVideo VisionEncoderQwen3VL::encode_frames(const std::vector<ov::Tensor>& frames, const ov::AnyMap& config_map) {
    EncodedVideo encoded_video;
    
    fill_video_metadata(encoded_video, frames.size(), m_video_processor_config);

    std::vector<ov::Tensor> sampled_frames;
    if (!m_video_processor_config.do_sample_frames) {
        sampled_frames = frames;
    } else {
        sampled_frames.reserve(encoded_video.metadata.frames_indices.size());
        for (size_t idx : encoded_video.metadata.frames_indices) {
            OPENVINO_ASSERT(idx < frames.size(),
                            "Frame index ", idx, " out of range for ", frames.size(), " frames.");
            sampled_frames.push_back(frames.at(idx));
        }
    }

    VisionEncoderQwen2VL::encode_frames_with_config(encoded_video, sampled_frames, m_video_processor_config);

    return encoded_video;
}

InputsEmbedderQwen3VL::InputsEmbedderQwen3VL(
    const VLMConfig& vlm_config,
    const std::filesystem::path& model_dir,
    const std::string& device,
    const ov::AnyMap device_config
) : InputsEmbedderQwen2VL(vlm_config, model_dir, device, device_config) {
    auto pos_model = utils::singleton_core().read_model(
        model_dir / "openvino_vision_embeddings_pos_model.xml");
    auto pos_compiled = utils::singleton_core().compile_model(pos_model, device, device_config);
    
    m_ireq_queue_vision_embeddings_pos = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        pos_compiled.get_property(ov::optimal_number_of_infer_requests),
        [&pos_compiled]() -> ov::InferRequest {
            return pos_compiled.create_infer_request();
        }
    );
}

InputsEmbedderQwen3VL::InputsEmbedderQwen3VL(
    const VLMConfig& vlm_config,
    const ModelsMap& models_map,
    const Tokenizer& tokenizer,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap device_config
) : InputsEmbedderQwen2VL(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) {
    const auto& [pos_model_str, pos_weights] = 
        utils::get_model_weights_pair(models_map, "vision_embeddings_pos");
    auto pos_model = utils::singleton_core().read_model(pos_model_str, pos_weights);
    auto pos_compiled = utils::singleton_core().compile_model(pos_model, device, device_config);
    
    m_ireq_queue_vision_embeddings_pos = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        pos_compiled.get_property(ov::optimal_number_of_infer_requests),
        [&pos_compiled]() -> ov::InferRequest {
            return pos_compiled.create_infer_request();
        }
    );
}

void InputsEmbedderQwen3VL::expand_video_tags_in_prompt(
    std::string& unified_prompt,
    const std::vector<EncodedVideo>& encoded_videos,
    const std::vector<size_t>& videos_sequence,
    size_t video_base_id
) const {
    std::vector<std::array<size_t, 3>> video_grid_thw_list;
    video_grid_thw_list.reserve(encoded_videos.size());

    for (const auto& encoded_video : encoded_videos) {
        size_t grid_t = encoded_video.frame_num;
        OPENVINO_ASSERT(grid_t > 0, "Video input must contain at least one frame.");
        size_t grid_h = encoded_video.resized_source_size.height;
        size_t grid_w = encoded_video.resized_source_size.width;
        video_grid_thw_list.push_back({grid_t, grid_h, grid_w});
    }

    for (size_t video_id : videos_sequence) {
        auto [grid_t, grid_h, grid_w] = video_grid_thw_list.at(video_id - video_base_id);
        // Calculate number of video pad tokens for each frame
        const size_t num_video_pad_tokens = calc_tokens_num(1, grid_h, grid_w);

        const auto& encoded_video = encoded_videos.at(video_id - video_base_id);
        const size_t spatial_merge_size = m_vision_encoder->get_processor_config().merge_size;
        auto timestamps = calculate_timestamps(encoded_video.metadata, spatial_merge_size);
        OPENVINO_ASSERT(timestamps.size() == grid_t, "Timestamps size does not match the number of frames");

        std::string expanded_tag;
        for (size_t grid_t_idx = 0; grid_t_idx < grid_t; ++grid_t_idx) {
            std::stringstream timestamp_ss;
            timestamp_ss << std::fixed << std::setprecision(1) << timestamps[grid_t_idx];
            const std::string timestamp_str = "<" + timestamp_ss.str() + " seconds>";
            expanded_tag.append(timestamp_str);
            expanded_tag.append(m_vlm_config.vision_start_token);
            for (size_t i = 0; i < num_video_pad_tokens; ++i) {
                expanded_tag.append(m_vlm_config.video_pad_token);
            }
            expanded_tag.append(m_vlm_config.vision_end_token);
        }

        unified_prompt.replace(unified_prompt.find(NATIVE_VIDEO_TAG), NATIVE_VIDEO_TAG.length(), expanded_tag);
    }
}

ov::Tensor InputsEmbedderQwen3VL::get_interpolated_pos_embeds(
    const std::vector<std::array<size_t, 3>>& grids_thw
) {
    using Clock = Qwen3VLBlockProfiler::Clock;
    using Duration = Qwen3VLBlockProfiler::Duration;
    auto t_start = Clock::now();

    const size_t num_grid_per_side = static_cast<size_t>(
        std::sqrt(static_cast<double>(m_vlm_config.vision_config_num_position_embeddings)));
    
    auto t0 = Clock::now();
    auto [indices, weights] = get_position_interpolation_indices_and_weights(grids_thw, num_grid_per_side);
    auto t1 = Clock::now();
    if (s_current_profiler) s_current_profiler->interp_indices_ms = Duration(t1 - t0).count();
    
    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(m_ireq_queue_vision_embeddings_pos.get());
    ov::InferRequest& vision_embeddings_pos = infer_request_guard.get();

    t0 = Clock::now();
    vision_embeddings_pos.set_tensor("input", indices);
    t1 = Clock::now();
    if (s_current_profiler) s_current_profiler->pos_set_tensor_ms = Duration(t1 - t0).count();

    t0 = Clock::now();
    vision_embeddings_pos.infer();
    t1 = Clock::now();
    if (s_current_profiler) s_current_profiler->pos_model_infer_ms = Duration(t1 - t0).count();
    ov::Tensor pos_embeds = vision_embeddings_pos.get_output_tensor();

    size_t num_positions = pos_embeds.get_shape()[1];
    size_t embed_dim = pos_embeds.get_shape()[2];

    // Weighted sum over 4 corners
    t0 = Clock::now();
    ov::Tensor weighted_sum{ov::element::f32, {num_positions, embed_dim}};
    float* weighted_sum_data = weighted_sum.data<float>();
    std::fill_n(weighted_sum_data, num_positions * embed_dim, 0.0f);

    const float* pos_embeds_data = pos_embeds.data<const float>();
    const float* weights_data = weights.data<const float>();
    
    // Apply weights and sum: pos_embeds * weights[:, :, None], then sum over dim 0
    for (size_t corner = 0; corner < 4; ++corner) {
        for (size_t pos = 0; pos < num_positions; ++pos) {
            float w = weights_data[corner * num_positions + pos];
            const float* src = pos_embeds_data + (corner * num_positions + pos) * embed_dim;
            float* dst = weighted_sum_data + pos * embed_dim;
            for (size_t d = 0; d < embed_dim; ++d) {
                dst[d] += w * src[d];
            }
        }
    }
    t1 = Clock::now();
    if (s_current_profiler) s_current_profiler->weighted_sum_ms = Duration(t1 - t0).count();
    
    t0 = Clock::now();
    size_t spatial_merge_size = m_vision_encoder->get_processor_config().merge_size;
    auto result = permute_with_spatial_merge(weighted_sum, grids_thw, spatial_merge_size);
    t1 = Clock::now();
    if (s_current_profiler) s_current_profiler->permute_ms = Duration(t1 - t0).count();

    if (s_current_profiler) s_current_profiler->total_interp_ms = Duration(Clock::now() - t_start).count();
    return result;
}

std::pair<ov::Tensor, ov::Tensor> InputsEmbedderQwen3VL::run_video_image_embeddings_merger(
    const std::vector<EncodedImage>& images,
    const std::vector<size_t>& images_sequence,
    const std::vector<EncodedVideo>& videos,
    const std::vector<size_t>& videos_sequence
) {
    using Clock = Qwen3VLBlockProfiler::Clock;
    using Duration = Qwen3VLBlockProfiler::Duration;
    auto t_total_start = Clock::now();

    auto t0 = Clock::now();
    auto [reordered_image_embeds, reordered_images_grid_thw] = 
        qwen2_vl_utils::reorder_image_embeds_and_grid_thw(images, images_sequence);
    auto [reordered_video_embeds, reordered_videos_grid_thw] = 
        qwen2_vl_utils::reorder_video_embeds_and_grid_thw(videos, videos_sequence);
    auto t1 = Clock::now();
    if (s_current_profiler) s_current_profiler->reorder_embeds_ms = Duration(t1 - t0).count();
    
    t0 = Clock::now();
    ov::Tensor concatenated_embeds = 
        qwen2_vl_utils::concatenate_video_image_embeds(reordered_video_embeds, reordered_image_embeds);
    t1 = Clock::now();
    if (s_current_profiler) s_current_profiler->concat_embeds_ms = Duration(t1 - t0).count();
    
    // Combined grid for position computation
    std::vector<std::array<size_t, 3>> combined_grid_thw;
    combined_grid_thw.insert(combined_grid_thw.end(), 
        reordered_videos_grid_thw.begin(), reordered_videos_grid_thw.end());
    combined_grid_thw.insert(combined_grid_thw.end(), 
        reordered_images_grid_thw.begin(), reordered_images_grid_thw.end());
    
    if (!combined_grid_thw.empty()) {
        t0 = Clock::now();
        ov::Tensor pos_embeds = get_interpolated_pos_embeds(combined_grid_thw);
        t1 = Clock::now();
        if (s_current_profiler) s_current_profiler->interp_pos_embeds_ms = Duration(t1 - t0).count();
        
        t0 = Clock::now();
        float* concatenated_embeds_data = concatenated_embeds.data<float>();
        const float* pos_embeds_data = pos_embeds.data<const float>();
        for (size_t i = 0; i < concatenated_embeds.get_size(); ++i) {
            concatenated_embeds_data[i] += pos_embeds_data[i];
        }
        t1 = Clock::now();
        if (s_current_profiler) s_current_profiler->pos_embeds_addition_ms = Duration(t1 - t0).count();
    }
    
    t0 = Clock::now();
    ov::Tensor rotary_pos_emb = get_rotary_pos_emb(combined_grid_thw);
    t1 = Clock::now();
    if (s_current_profiler) s_current_profiler->rotary_pos_emb_ms = Duration(t1 - t0).count();
    
    t0 = Clock::now();
    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(m_ireq_queue_vision_embeddings_merger.get());
    ov::InferRequest& vision_embeddings_merger = infer_request_guard.get();
    
    vision_embeddings_merger.set_tensor("hidden_states", concatenated_embeds);
    
    if (m_with_cu_seqlens_input) {
        vision_embeddings_merger.set_tensor("cu_seq_lens", 
            qwen2_vl_utils::get_cu_seqlens(reordered_images_grid_thw, reordered_videos_grid_thw));
    } else {
        vision_embeddings_merger.set_tensor("attention_mask",
            qwen2_vl_utils::get_attention_mask(reordered_images_grid_thw, reordered_videos_grid_thw));
    }
    
    vision_embeddings_merger.set_tensor("rotary_pos_emb", rotary_pos_emb);
    t1 = Clock::now();
    if (s_current_profiler) s_current_profiler->merger_set_tensors_ms = Duration(t1 - t0).count();

    t0 = Clock::now();
    vision_embeddings_merger.infer();
    t1 = Clock::now();
    if (s_current_profiler) s_current_profiler->merger_infer_ms = Duration(t1 - t0).count();
    
    t0 = Clock::now();
    ov::Tensor vision_embeds = vision_embeddings_merger.get_tensor("last_hidden_state");
    m_lm_extra_inputs["deepstack_visual_embeds"] = vision_embeddings_merger.get_tensor("deepstack_feature_lists");
    
    auto vision_embeds_shape = vision_embeds.get_shape();
    
    // Split vision embeddings
    size_t video_tokens = calc_vec_tokens_num(reordered_videos_grid_thw);
    size_t image_tokens = calc_vec_tokens_num(reordered_images_grid_thw);
    size_t total_tokens = video_tokens + image_tokens;
    
    size_t video_count = 0;
    if (total_tokens > 0) {
        video_count = vision_embeds_shape[0] * video_tokens / total_tokens;
    }
    size_t image_count = vision_embeds_shape[0] - video_count;
    
    ov::Tensor video_embeds{vision_embeds.get_element_type(), {video_count, vision_embeds_shape[1]}};
    ov::Tensor image_embeds{vision_embeds.get_element_type(), {image_count, vision_embeds_shape[1]}};
    
    std::memcpy(video_embeds.data(), vision_embeds.data(), video_embeds.get_byte_size());
    std::memcpy(image_embeds.data(),
                static_cast<uint8_t*>(vision_embeds.data()) + video_embeds.get_byte_size(),
                image_embeds.get_byte_size());
    t1 = Clock::now();
    if (s_current_profiler) s_current_profiler->split_embeds_ms = Duration(t1 - t0).count();
    
    if (s_current_profiler) {
        s_current_profiler->total_merger_ms = Duration(Clock::now() - t_total_start).count();
        if (s_qwen3vl_profile_enabled) s_current_profiler->print_merger();
    }
    return {video_embeds, image_embeds};
}

std::vector<std::array<size_t, 3>> InputsEmbedderQwen3VL::get_vision_grid_thw_for_position_ids(
    const std::vector<std::array<size_t, 3>>& images_grid_thw,
    const std::vector<size_t>& images_sequence,
    const size_t image_id,
    const std::vector<std::array<size_t, 3>>& videos_grid_thw,
    const std::vector<size_t>& videos_sequence,
    const size_t video_id,
    const std::vector<std::pair<std::size_t, std::size_t>>& history_vision_count
) const {
    auto reordered_vision_grid_thw = InputsEmbedderQwen2VL::get_vision_grid_thw_for_position_ids(
        images_grid_thw,
        images_sequence,
        image_id,
        videos_grid_thw,
        videos_sequence,
        video_id,
        history_vision_count
    );

    // Split video grids per each frame for position_ids calculation as Qwen3-VL uses timestamp tokens between frames.
    std::vector<std::array<size_t, 3>> flattened_vision_grid_thw;

    for (const auto& vision_grid_thw : reordered_vision_grid_thw) {
        auto [grid_t, grid_h, grid_w] = vision_grid_thw;

        if (grid_t > 1) {
            for (size_t frame_idx = 0; frame_idx < grid_t; ++frame_idx) {
                flattened_vision_grid_thw.push_back({1, grid_h, grid_w});
            }
        } else {
            flattened_vision_grid_thw.push_back(vision_grid_thw);
        }
    }
    return flattened_vision_grid_thw;
}

ov::Tensor InputsEmbedderQwen3VL::get_inputs_embeds(
    const std::string& unified_prompt,
    const std::vector<ov::genai::EncodedImage>& images,
    const std::vector<ov::genai::EncodedVideo>& videos,
    ov::genai::VLMPerfMetrics& metrics,
    bool recalculate_merged_embeddings,
    const std::vector<size_t>& images_sequence,
    const std::vector<size_t>& videos_sequence,
    const std::vector<std::pair<std::size_t, std::size_t>>& history_vision_count
) {
    using Clock = Qwen3VLBlockProfiler::Clock;
    using Duration = Qwen3VLBlockProfiler::Duration;
    Qwen3VLBlockProfiler prof;
    s_current_profiler = &prof;
    auto t_total_start = Clock::now();

    auto t0 = Clock::now();
    std::vector<std::array<size_t, 3>> images_grid_thw;
    images_grid_thw.reserve(images.size());
    for (const auto& encoded_image : images) {
        images_grid_thw.push_back({
            1,
            encoded_image.resized_source_size.height,
            encoded_image.resized_source_size.width
        });
    }

    std::vector<std::array<size_t, 3>> videos_grid_thw;
    videos_grid_thw.reserve(videos.size());
    for (const auto& encoded_video : videos) {
        videos_grid_thw.push_back({
            encoded_video.frame_num,
            encoded_video.resized_source_size.height,
            encoded_video.resized_source_size.width
        });
    }
    auto t1 = Clock::now();
    prof.grid_thw_prep_ms = Duration(t1 - t0).count();

    t0 = Clock::now();
    ov::Tensor input_ids = get_encoded_input_ids(unified_prompt, metrics);
    t1 = Clock::now();
    prof.tokenization_ms = Duration(t1 - t0).count();

    t0 = Clock::now();
    CircularBufferQueueElementGuard<EmbeddingsRequest> embeddings_request_guard(m_embedding->get_request_queue().get());
    EmbeddingsRequest& req = embeddings_request_guard.get();
    ov::Tensor text_embeds = m_embedding->infer(req, input_ids);
    t1 = Clock::now();
    prof.text_embed_infer_ms = Duration(t1 - t0).count();

    int64_t vision_start_token_id = m_vision_token_ids.at("vision_start");
    int64_t image_pad_token_id = m_vision_token_ids.at("image_pad");
    int64_t video_pad_token_id = m_vision_token_ids.at("video_pad");

    t0 = Clock::now();
    m_position_ids = create_position_ids(input_ids, images_grid_thw, images_sequence, 0, 
                                         videos_grid_thw, videos_sequence, 0, 
                                         vision_start_token_id, history_vision_count);

    int64_t position_ids_max = *std::max_element(m_position_ids.data<int64_t>(), 
                                                 m_position_ids.data<int64_t>() + m_position_ids.get_size());
    m_rope_delta = position_ids_max + 1 - static_cast<int64_t>(input_ids.get_shape().at(1));
    t1 = Clock::now();
    prof.position_ids_ms = Duration(t1 - t0).count();

    if (images.empty() && videos.empty()) {
        t0 = Clock::now();
        // visual_pos_masks extra input
        const size_t batch_size = input_ids.get_shape()[0];
        ov::Tensor visual_pos_masks(ov::element::boolean, {batch_size, 1});
        std::fill_n(visual_pos_masks.data<bool>(), visual_pos_masks.get_size(), false);
        m_lm_extra_inputs["visual_pos_masks"] = std::move(visual_pos_masks);

        // deepstack_visual_embeds extra input
        const size_t num_layers = m_vlm_config.vision_config_deepstack_visual_indexes.size();
        const size_t hidden_size = text_embeds.get_shape()[2];
        ov::Tensor deepstack_visual_embeds(ov::element::f32, {num_layers, 1, hidden_size});
        std::fill_n(deepstack_visual_embeds.data<float>(), deepstack_visual_embeds.get_size(), 0.0f);
        m_lm_extra_inputs["deepstack_visual_embeds"] = std::move(deepstack_visual_embeds);

        ov::Tensor inputs_embeds(text_embeds.get_element_type(), text_embeds.get_shape());
        std::memcpy(inputs_embeds.data(), text_embeds.data(), text_embeds.get_byte_size());
        t1 = Clock::now();
        prof.no_vision_setup_ms = Duration(t1 - t0).count();
        prof.total_get_inputs_ms = Duration(Clock::now() - t_total_start).count();
        if (s_qwen3vl_profile_enabled) prof.print_get_inputs_embeds();
        s_current_profiler = nullptr;
        return inputs_embeds;
    }

    if (recalculate_merged_embeddings) {
        t0 = Clock::now();
        std::tie(m_merged_video_embeddings, m_merged_image_embeddings) = 
            run_video_image_embeddings_merger(images, images_sequence, videos, videos_sequence);
        t1 = Clock::now();
        prof.merger_ms = Duration(t1 - t0).count();
    }

    t0 = Clock::now();
    m_lm_extra_inputs["visual_pos_masks"] = create_visual_pos_masks(input_ids, image_pad_token_id, video_pad_token_id);
    t1 = Clock::now();
    prof.visual_pos_masks_ms = Duration(t1 - t0).count();

    t0 = Clock::now();
    auto result = qwen2_vl_utils::merge_text_and_video_image_embeddings(
        input_ids, text_embeds, m_merged_image_embeddings, m_merged_video_embeddings,
        image_pad_token_id, video_pad_token_id);
    t1 = Clock::now();
    prof.merge_text_vision_ms = Duration(t1 - t0).count();

    prof.total_get_inputs_ms = Duration(Clock::now() - t_total_start).count();
    if (s_qwen3vl_profile_enabled) prof.print_get_inputs_embeds();
    s_current_profiler = nullptr;
    return result;
}

void InputsEmbedderQwen3VL::start_chat(const std::string& system_message) {
    InputsEmbedderQwen2VL::start_chat(system_message);
    m_lm_extra_inputs["deepstack_visual_embeds"] = ov::Tensor();
    m_lm_extra_inputs["visual_pos_masks"] = ov::Tensor();
}

void InputsEmbedderQwen3VL::finish_chat() {
    InputsEmbedderQwen2VL::finish_chat();
    m_lm_extra_inputs["deepstack_visual_embeds"] = ov::Tensor();
    m_lm_extra_inputs["visual_pos_masks"] = ov::Tensor();
}

const std::unordered_map<std::string, ov::Tensor>& InputsEmbedderQwen3VL::get_lm_extra_inputs() const {
    return m_lm_extra_inputs;
}

} // namespace ov::genai
