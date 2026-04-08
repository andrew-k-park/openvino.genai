// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/perf_metrics.hpp"
#include "openvino/genai/visibility.hpp"


namespace ov::genai {

struct OPENVINO_GENAI_EXPORTS VLMRawPerfMetrics {
    /** @brief Duration of preparation of embeddings */
    std::vector<MicroSeconds> prepare_embeddings_durations;
    /** @brief Duration of text embedding model inference within EmbedPrep */
    std::vector<MicroSeconds> text_embed_durations;
    /** @brief Duration of position embedding model inference within EmbedPrep */
    std::vector<MicroSeconds> pos_embed_durations;
    /** @brief Duration of vision embeddings merger model inference within EmbedPrep */
    std::vector<MicroSeconds> merger_durations;
    /** @brief Total duration of vision encoder (patch embedding) inference within EmbedPrep */
    std::vector<MicroSeconds> vision_encoder_durations;
};

struct OPENVINO_GENAI_EXPORTS VLMPerfMetrics : public PerfMetrics {
    /** @brief Mean and standard deviation of preparation of embeddings in milliseconds */
    MeanStdPair prepare_embeddings_duration;

    MeanStdPair get_prepare_embeddings_duration();

    VLMPerfMetrics() = default;

    VLMPerfMetrics(PerfMetrics& perf_metrics) : PerfMetrics(perf_metrics), prepare_embeddings_duration(){};

    void evaluate_statistics(std::optional<TimePoint> start_time = std::nullopt) override;

    VLMPerfMetrics operator+(const VLMPerfMetrics& metrics) const;

    VLMRawPerfMetrics vlm_raw_metrics;
};

}
