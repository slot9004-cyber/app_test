//============================================================================
// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//============================================================================

#include <jni.h>
#include <string>
#include <iostream>

#include <iterator>
#include <unordered_map>
#include <algorithm>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <hpp/inference.h>

#include "android/log.h"

#include "hpp/CheckRuntime.hpp"
#include "hpp/SetBuilderOptions.hpp"
#include "hpp/Util.hpp"
#include "LoadContainer.hpp"
#include "CreateUserBuffer.hpp"
#include "LoadInputTensor.hpp"

#include <opencv2/core/types_c.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>

std::unique_ptr<zdl::SNPE::SNPE> snpe_HRNET;
std::unique_ptr<zdl::SNPE::SNPE> snpe_BB;

std::mutex mtx;
static zdl::DlSystem::Runtime_t runtime = zdl::DlSystem::Runtime_t::CPU;
static zdl::DlSystem::RuntimeList runtimeList;
bool useUserSuppliedBuffers = true;
bool useIntBuffer = false;

zdl::DlSystem::UserBufferMap inputMap, outputMap;
std::vector <std::unique_ptr<zdl::DlSystem::IUserBuffer>> snpeUserBackedInputBuffers, snpeUserBackedOutputBuffers;
std::unordered_map <std::string, std::vector<float32_t>> applicationOutputBuffers;
std::unordered_map <std::string, std::vector<float32_t>> applicationInputBuffers;
int bitWidth = 32;


#include <android/trace.h>
#include <dlfcn.h>


std::map<int, std::string> classnamemapping =
        {
                {0, "person"},{ 1, "bicycle"},{ 2, "car"},{ 3, "motorcycle"},{ 4, "airplane"},{ 5, "bus"},{
                 6, "train"},{ 7, "truck"},{ 8, "boat"},{ 9, "traffic"},{ 10, "fire"},{ 11, "stop"},{ 12, "parking"},{
                 13, "bench"},{ 14, "bird"},{ 15, "cat"},{ 16, "dog"},{ 17, "horse"},{ 18, "sheep"},{ 19, "cow"},{
                 20, "elephant"},{ 21, "bear"},{ 22, "zebra"},{ 23, "giraffe"},{ 24, "backpack"},{ 25, "umbrella"},{
                 26, "handbag"},{ 27, "tie"},{ 28, "suitcase"},{ 29, "frisbee"},{ 30, "skis"},{ 31, "snowboard"},{
                 32, "sports"},{ 33, "kite"},{ 34, "baseball"},{ 35, "baseball"},{ 36, "skateboard"},{ 37, "surfboard"},{
                 38, "tennis"},{ 39, "bottle"},{ 40, "wine"},{ 41, "cup"},{ 42, "fork"},{ 43, "knife"},{ 44, "spoon"},{
                 45, "bowl"},{ 46, "banana"},{ 47, "apple"},{ 48, "sandwich"},{ 49, "orange"},{ 50, "broccoli"},{
                 51, "carrot"},{ 52, "hot"},{ 53, "pizza"},{ 54, "donut"},{ 55, "cake"},{ 56, "chair"},{ 57, "couch"},{
                 58, "potted"},{ 59, "bed"},{ 60, "dining"},{ 61, "toilet"},{ 62, "tv"},{ 63, "laptop"},{ 64, "mouse"},{
                 65, "remote"},{ 66, "keyboard"},{ 67, "cell"},{ 68, "microwave"},{ 69, "oven"},{ 70, "toaster"},{
                 71, "sink"},{ 72, "refrigerator"},{ 73, "book"},{ 74, "clock"},{ 75, "vase"},{ 76, "scissors"},{
                 77, "teddy"},{ 78, "hair"},{ 79, "toothbrush"}
        };

inline float ComputeIntersectionOverUnion(const BoxCornerEncoding &box_i,const BoxCornerEncoding &box_j)
{
    const float box_i_y_min = std::min<float>(box_i.y1, box_i.y2);
    const float box_i_y_max = std::max<float>(box_i.y1, box_i.y2);
    const float box_i_x_min = std::min<float>(box_i.x1, box_i.x2);
    const float box_i_x_max = std::max<float>(box_i.x1, box_i.x2);
    const float box_j_y_min = std::min<float>(box_j.y1, box_j.y2);
    const float box_j_y_max = std::max<float>(box_j.y1, box_j.y2);
    const float box_j_x_min = std::min<float>(box_j.x1, box_j.x2);
    const float box_j_x_max = std::max<float>(box_j.x1, box_j.x2);

    const float area_i =
            (box_i_y_max - box_i_y_min) * (box_i_x_max - box_i_x_min);
    const float area_j =
            (box_j_y_max - box_j_y_min) * (box_j_x_max - box_j_x_min);
    if (area_i <= 0 || area_j <= 0) return 0.0;
    const float intersection_ymax = std::min<float>(box_i_y_max, box_j_y_max);
    const float intersection_xmax = std::min<float>(box_i_x_max, box_j_x_max);
    const float intersection_ymin = std::max<float>(box_i_y_min, box_j_y_min);
    const float intersection_xmin = std::max<float>(box_i_x_min, box_j_x_min);
    const float intersection_area =
            std::max<float>(intersection_ymax - intersection_ymin, 0.0) *
            std::max<float>(intersection_xmax - intersection_xmin, 0.0);
    return intersection_area / (area_i + area_j - intersection_area);
}

std::vector<BoxCornerEncoding> NonMaxSuppression(std::vector<BoxCornerEncoding> boxes,
                           const float iou_threshold)
{

    if (boxes.size()==0) {
        return boxes;
    }

    std::sort(boxes.begin(), boxes.end(), [] (const BoxCornerEncoding& left, const BoxCornerEncoding& right) {
        if (left.score > right.score) {
            return true;
        } else {
            return false;
        }
    });


    std::vector<bool> flag(boxes.size(), false);
    for (unsigned int i = 0; i < boxes.size(); i++) {
        if (flag[i]) {
            continue;
        }

        for (unsigned int j = i + 1; j < boxes.size(); j++) {
            if (ComputeIntersectionOverUnion(boxes[i],boxes[j]) > iou_threshold) {
                flag[j] = true;
            }
        }
    }

    std::vector<BoxCornerEncoding> ret;
    for (unsigned int i = 0; i < boxes.size(); i++) {
        if (!flag[i])
            ret.push_back(boxes[i]);
    }

    return ret;
}

std::string build_network_BB(const uint8_t * dlc_buffer_BB, const size_t dlc_size_BB, const char runtime_arg)
{
    std::string outputLogger;
    bool usingInitCaching = false;  //shubham: TODO check with true

    std::unique_ptr<zdl::DlContainer::IDlContainer> container_BB = nullptr ;

    container_BB = loadContainerFromBuffer(dlc_buffer_BB, dlc_size_BB);

    if (container_BB == nullptr) {
        LOGE("Error while opening the container file.");
        return "Error while opening the container file.\n";
    }

    runtimeList.clear();
    LOGI("runtime arg %c",runtime_arg);
    zdl::DlSystem::Runtime_t runtime = zdl::DlSystem::Runtime_t::CPU;
    if (runtime_arg == 'D'){
        runtime = zdl::DlSystem::Runtime_t::DSP;
        LOGI("Added DSP");
    }
    else if (runtime_arg == 'G')
    {
        runtime = zdl::DlSystem::Runtime_t::GPU_FLOAT32_16_HYBRID; //can be written as GPU
        LOGI("Added GPU");
    }

    if(runtime != zdl::DlSystem::Runtime_t::UNSET)
    {
        bool ret = runtimeList.add(checkRuntime(runtime));
        if(ret == false){
            LOGE("Cannot set runtime");
            return outputLogger + "\nCannot set runtime";
        }
    } else {
        return outputLogger + "\nCannot set runtime";
    }


    mtx.lock();
    snpe_BB = setBuilderOptions(container_BB, runtime, runtimeList, useUserSuppliedBuffers, usingInitCaching);
    mtx.unlock();

    if (snpe_BB == nullptr) {
        LOGE("SNPE Prepare failed: Builder option failed for BB");
        outputLogger += "Model Prepare failed for BB";
        return outputLogger + "SNPE Prepare failed for BB";
    }

    outputLogger += "\nBB Model Network Prepare success !!!\n";

    //Creating Buffer
    createInputBufferMap(inputMap, applicationInputBuffers, snpeUserBackedInputBuffers, snpe_BB, useIntBuffer, bitWidth);
    createOutputBufferMap(outputMap, applicationOutputBuffers, snpeUserBackedOutputBuffers, snpe_BB, useIntBuffer, bitWidth);
    return outputLogger;
}



bool executeDLC(cv::Mat &img, int orig_width, int orig_height, int &numberofobj, std::vector<std::vector<float>> &BB_coords, std::vector<std::string> &BB_names, cv::Mat& combined_mask) {

    LOGI("execute_net_BB");
    ATrace_beginSection("preprocessing");

    struct timeval start_time, end_time;
    float milli_time, seconds, useconds;

    mtx.lock();
    assert(snpe_BB!=nullptr);

    if(!loadInputUserBuffer_BB(applicationInputBuffers, snpe_BB, img, inputMap, bitWidth))
    {
        LOGE("Failed to load Input UserBuffer");
        mtx.unlock();
        return false;
    }

    // Define output tensor names - these are guesses based on common YOLO output names.
    std::string output0_name = "output0"; // Shape: 1x116x8400 (boxes, scores, mask coeffs)
    std::string output1_name = "output1"; // Shape: 1x32x160x160 (mask prototypes)

    ATrace_endSection();
    gettimeofday(&start_time, NULL);
    ATrace_beginSection("inference time");

    bool execStatus = snpe_BB->execute(inputMap, outputMap);
    ATrace_endSection();
    ATrace_beginSection("postprocessing time");
    gettimeofday(&end_time, NULL);
    seconds = end_time.tv_sec - start_time.tv_sec; //seconds
    useconds = end_time.tv_usec - start_time.tv_usec; //milliseconds
    milli_time = ((seconds) * 1000 + useconds/1000.0);

    if(!execStatus){
        LOGE("Exec BB status is false: %s", zdl::DlSystem::getLastErrorString());
        mtx.unlock();
        return false;
    }

    // Get output tensors
    if (applicationOutputBuffers.find(output0_name) == applicationOutputBuffers.end() ||
        applicationOutputBuffers.find(output1_name) == applicationOutputBuffers.end()) {
        LOGE("Could not find output tensors with assumed names %s and %s.", output0_name.c_str(), output1_name.c_str());
        for (auto const& [key, val] : applicationOutputBuffers) {
            LOGE("Available output tensor: %s", key.c_str());
        }
        mtx.unlock();
        return false;
    }
    std::vector<float32_t>& output0_buffer = applicationOutputBuffers.at(output0_name);
    std::vector<float32_t>& output1_buffer = applicationOutputBuffers.at(output1_name);

    // Constants for post-processing
    const int NUM_PROPOSALS = 8400;
    const int CHANNELS_PER_PROPOSAL = 116;
    const int NUM_CLASSES = 80;
    const int MASK_COEFFS = 32;
    const int MASK_HEIGHT = 160;
    const int MASK_WIDTH = 160;
    const int INPUT_WIDTH = 512;
    const int INPUT_HEIGHT = 512;
    const float CONF_THRESHOLD = 0.5f;
    const float IOU_THRESHOLD = 0.5f;
    const int PERSON_CLASS_ID = 0;

    // Reshape output0 from flat buffer to 8400x116 for easier processing
    cv::Mat proposals(CHANNELS_PER_PROPOSAL, NUM_PROPOSALS, CV_32F, output0_buffer.data());
    cv::Mat proposals_t = proposals.t();

    std::vector<cv::Rect> boxes;
    std::vector<float> confs;
    std::vector<cv::Mat> proposal_mask_coeffs;

    float x_factor = (float)orig_width / INPUT_WIDTH;
    float y_factor = (float)orig_height / INPUT_HEIGHT;

    for (int i = 0; i < NUM_PROPOSALS; ++i) {
        cv::Mat proposal = proposals_t.row(i);
        cv::Mat box_coords = proposal.colRange(0, 4);
        cv::Mat class_scores = proposal.colRange(4, 4 + NUM_CLASSES);

        cv::Point class_id_point;
        double max_val;
        cv::minMaxLoc(class_scores, 0, &max_val, 0, &class_id_point);

        if (max_val > CONF_THRESHOLD && class_id_point.x == PERSON_CLASS_ID) {
            confs.push_back(max_val);

            float cx = box_coords.at<float>(0,0);
            float cy = box_coords.at<float>(0,1);
            float w = box_coords.at<float>(0,2);
            float h = box_coords.at<float>(0,3);

            int left = static_cast<int>((cx - 0.5 * w) * x_factor);
            int top = static_cast<int>((cy - 0.5 * h) * y_factor);
            int width = static_cast<int>(w * x_factor);
            int height = static_cast<int>(h * y_factor);

            boxes.push_back(cv::Rect(left, top, width, height));
            proposal_mask_coeffs.push_back(proposal.colRange(4 + NUM_CLASSES, CHANNELS_PER_PROPOSAL));
        }
    }

    // Perform Non-Maximum Suppression
    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confs, CONF_THRESHOLD, IOU_THRESHOLD, nms_result);

    numberofobj = nms_result.size();
    BB_coords.clear();
    BB_names.clear();
    combined_mask = cv::Mat::zeros(orig_height, orig_width, CV_8U);
    cv::Mat proto_masks(MASK_COEFFS, MASK_WIDTH * MASK_HEIGHT, CV_32F, output1_buffer.data());

    for (int idx : nms_result) {
        cv::Rect box = boxes[idx];
        BB_coords.push_back({(float)box.x, (float)box.y, (float)(box.x + box.width), (float)(box.y + box.height), milli_time});
        BB_names.push_back("person");

        // Reconstruct mask for the detected object
        cv::Mat mask_coeffs = proposal_mask_coeffs[idx];
        cv::Mat matmul_result;
        cv::gemm(mask_coeffs, proto_masks, 1.0, cv::Mat(), 0.0, matmul_result);
        cv::Mat final_mask = matmul_result.reshape(1, {MASK_HEIGHT, MASK_WIDTH});

        // Apply sigmoid
        cv::exp(-final_mask, final_mask);
        final_mask = 1.0 / (1.0 + final_mask);

        // Binarize the mask
        cv::Mat binary_mask = final_mask > 0.5;

        // Upscale the binary mask to original image size
        cv::Mat upscaled_mask;
        cv::resize(binary_mask, upscaled_mask, cv::Size(orig_width, orig_height), 0, 0, cv::INTER_NEAREST);

        // Combine with the main mask using bitwise OR
        combined_mask |= upscaled_mask;
    }

    ATrace_endSection();
    mtx.unlock();
    return true;
}

