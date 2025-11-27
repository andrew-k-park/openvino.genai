// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cmath>
#include <algorithm>
#include <random>
#include "visual_language/clip.hpp"


// Helper function to create test images
clip_image_u8 create_test_image(int width, int height, const std::string& pattern) {
    clip_image_u8 img;
    img.nx = width;
    img.ny = height;
    img.buf.resize(3 * width * height);
    
    if (pattern == "checkerboard") {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                bool white = ((x / 8) + (y / 8)) % 2 == 0;
                uint8_t val = white ? 255 : 0;
                for (int c = 0; c < 3; c++) {
                    img.buf[3 * (y * width + x) + c] = val;
                }
            }
        }
    } else if (pattern == "gradient") {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                img.buf[3 * (y * width + x) + 0] = static_cast<uint8_t>(255.0 * x / width);        // R
                img.buf[3 * (y * width + x) + 1] = static_cast<uint8_t>(255.0 * y / height);       // G
                img.buf[3 * (y * width + x) + 2] = static_cast<uint8_t>(128);                      // B
            }
        }
    } else if (pattern == "random") {
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> dist(0, 255);
        for (auto& pixel : img.buf) {
            pixel = static_cast<uint8_t>(dist(rng));
        }
    }
    
    return img;
}

// Calculate PSNR (Peak Signal-to-Noise Ratio)
double calculate_psnr(const clip_image_u8& img1, const clip_image_u8& img2) {
    EXPECT_EQ(img1.nx, img2.nx);
    EXPECT_EQ(img1.ny, img2.ny);
    EXPECT_EQ(img1.buf.size(), img2.buf.size());
    
    double mse = 0.0;
    size_t count = img1.buf.size();
    
    for (size_t i = 0; i < count; i++) {
        double diff = static_cast<double>(img1.buf[i]) - static_cast<double>(img2.buf[i]);
        mse += diff * diff;
    }
    mse /= count;
    
    if (mse == 0.0) return INFINITY;  // Perfectly identical
    return 10.0 * std::log10(255.0 * 255.0 / mse);
}

// Calculate MSE (Mean Squared Error)
double calculate_mse(const clip_image_u8& img1, const clip_image_u8& img2) {
    double mse = 0.0;
    size_t count = img1.buf.size();
    
    for (size_t i = 0; i < count; i++) {
        double diff = static_cast<double>(img1.buf[i]) - static_cast<double>(img2.buf[i]);
        mse += diff * diff;
    }
    return mse / count;
}

// Calculate maximum pixel difference
int calculate_max_diff(const clip_image_u8& img1, const clip_image_u8& img2) {
    int max_diff = 0;
    for (size_t i = 0; i < img1.buf.size(); i++) {
        int diff = std::abs(static_cast<int>(img1.buf[i]) - static_cast<int>(img2.buf[i]));
        max_diff = std::max(max_diff, diff);
    }
    return max_diff;
}

// Calculate per-pixel difference histogram
std::vector<int> calculate_diff_histogram(const clip_image_u8& img1, const clip_image_u8& img2) {
    std::vector<int> hist(256, 0);
    for (size_t i = 0; i < img1.buf.size(); i++) {
        int diff = std::abs(static_cast<int>(img1.buf[i]) - static_cast<int>(img2.buf[i]));
        hist[diff]++;
    }
    return hist;
}

class BicubicResizeAccuracyTest : public ::testing::Test {
protected:
    void CompareResizeMethods(const clip_image_u8& src, int target_w, int target_h, 
                             double min_psnr = 30.0, int max_diff_threshold = 2) {
        clip_image_u8 result_original;
        clip_image_u8 result_optimized;
        
        // Original implementation
        bicubic_resize(src, result_original, target_w, target_h);
        
        // Optimized implementation
        bicubic_resize_opt(src, result_optimized, target_w, target_h);
        
        // Validate output dimensions
        ASSERT_EQ(result_original.nx, target_w);
        ASSERT_EQ(result_original.ny, target_h);
        ASSERT_EQ(result_optimized.nx, target_w);
        ASSERT_EQ(result_optimized.ny, target_h);
        ASSERT_EQ(result_original.buf.size(), result_optimized.buf.size());
        
        // Quantitative comparison
        double psnr = calculate_psnr(result_original, result_optimized);
        double mse = calculate_mse(result_original, result_optimized);
        int max_diff = calculate_max_diff(result_original, result_optimized);
        
        std::cout << "  PSNR: " << psnr << " dB" << std::endl;
        std::cout << "  MSE: " << mse << std::endl;
        std::cout << "  Max Diff: " << max_diff << " (out of 255)" << std::endl;
        
        // Difference histogram
        auto hist = calculate_diff_histogram(result_original, result_optimized);
        int pixels_with_diff = 0;
        for (int i = 1; i < 256; i++) {
            pixels_with_diff += hist[i];
        }
        double diff_percentage = 100.0 * pixels_with_diff / result_original.buf.size();
        std::cout << "  Pixels with diff: " << diff_percentage << "%" << std::endl;
        
        // Verify tolerance thresholds
        EXPECT_GE(psnr, min_psnr) << "PSNR too low";
        EXPECT_LE(max_diff, max_diff_threshold) << "Max pixel difference too high";
        EXPECT_LE(mse, 1.0) << "MSE too high";
    }
};

TEST_F(BicubicResizeAccuracyTest, CheckerboardPattern_Downscale) {
    std::cout << "\n=== Checkerboard Downscale (256x256 -> 128x128) ===" << std::endl;
    auto src = create_test_image(256, 256, "checkerboard");
    CompareResizeMethods(src, 128, 128);
}

TEST_F(BicubicResizeAccuracyTest, CheckerboardPattern_Upscale) {
    std::cout << "\n=== Checkerboard Upscale (64x64 -> 256x256) ===" << std::endl;
    auto src = create_test_image(64, 64, "checkerboard");
    CompareResizeMethods(src, 256, 256);
}

TEST_F(BicubicResizeAccuracyTest, GradientPattern_Downscale) {
    std::cout << "\n=== Gradient Downscale (512x512 -> 224x224) ===" << std::endl;
    auto src = create_test_image(512, 512, "gradient");
    CompareResizeMethods(src, 224, 224);
}

TEST_F(BicubicResizeAccuracyTest, GradientPattern_Upscale) {
    std::cout << "\n=== Gradient Upscale (128x128 -> 336x336) ===" << std::endl;
    auto src = create_test_image(128, 128, "gradient");
    CompareResizeMethods(src, 336, 336);
}

TEST_F(BicubicResizeAccuracyTest, RandomPattern_TypicalSize) {
    std::cout << "\n=== Random Pattern (720x480 -> 224x224) ===" << std::endl;
    auto src = create_test_image(720, 480, "random");
    CompareResizeMethods(src, 224, 224, 25.0, 3);  // Wider tolerance for random patterns
}

TEST_F(BicubicResizeAccuracyTest, SmallImage) {
    std::cout << "\n=== Small Image (16x16 -> 8x8) ===" << std::endl;
    auto src = create_test_image(16, 16, "gradient");
    CompareResizeMethods(src, 8, 8);
}

TEST_F(BicubicResizeAccuracyTest, LargeImage) {
    std::cout << "\n=== Large Image (1920x1080 -> 640x360) ===" << std::endl;
    auto src = create_test_image(1920, 1080, "gradient");
    CompareResizeMethods(src, 640, 360);
}

TEST_F(BicubicResizeAccuracyTest, NonUniformScale) {
    std::cout << "\n=== Non-uniform Scale (400x300 -> 224x224) ===" << std::endl;
    auto src = create_test_image(400, 300, "gradient");
    CompareResizeMethods(src, 224, 224);
}

TEST_F(BicubicResizeAccuracyTest, IdenticalSize) {
    std::cout << "\n=== Identical Size (224x224 -> 224x224) ===" << std::endl;
    auto src = create_test_image(224, 224, "gradient");
    CompareResizeMethods(src, 224, 224, 40.0, 1);  // Should be nearly identical
}

TEST_F(BicubicResizeAccuracyTest, EdgeCases_SinglePixel) {
    std::cout << "\n=== Edge Case: Single Pixel (1x1 -> 10x10) ===" << std::endl;
    auto src = create_test_image(1, 1, "gradient");
    CompareResizeMethods(src, 10, 10, 30.0, 5);
}

// Performance comparison test (optional)
TEST_F(BicubicResizeAccuracyTest, DISABLED_PerformanceComparison) {
    auto src = create_test_image(1280, 720, "random");
    const int iterations = 100;
    
    // Original bicubic_resize
    long long total_original_us = 0;
    for (int i = 0; i < iterations; i++) {
        clip_image_u8 result;
        auto start = std::chrono::high_resolution_clock::now();
        bicubic_resize(src, result, 597, 336);
        auto end = std::chrono::high_resolution_clock::now();
        total_original_us += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }
    double mean_original_us = static_cast<double>(total_original_us) / iterations;
    
    // Optimized bicubic_resize_opt
    long long total_optimized_us = 0;
    for (int i = 0; i < iterations; i++) {
        clip_image_u8 result;
        auto start = std::chrono::high_resolution_clock::now();
        bicubic_resize_opt(src, result, 597, 336);
        auto end = std::chrono::high_resolution_clock::now();
        total_optimized_us += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }
    double mean_optimized_us = static_cast<double>(total_optimized_us) / iterations;
    
    std::cout << "\n=== Performance (" << iterations << " iterations) ===" << std::endl;
    std::cout << "  Original:  " << mean_original_us << " us (mean), " 
              << total_original_us / 1000 << " ms (total)" << std::endl;
    std::cout << "  Optimized: " << mean_optimized_us << " us (mean), " 
              << total_optimized_us / 1000 << " ms (total)" << std::endl;
    std::cout << "  Speedup:   " << mean_original_us / mean_optimized_us << "x" << std::endl;
}
