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
#include <opencv2/dnn.hpp>

std::unique_ptr<zdl::SNPE::SNPE> snpe_HRNET;
std::unique_ptr<zdl::SNPE::SNPE> snpe_BB;

std::mutex mtx;
static zdl::DlSystem::Runtime_t runtime = zdl::DlSystem::Runtime_t::CPU;
static zdl::DlSystem::RuntimeList runtimeList;
bool useUserSuppliedBuffers = true;
bool useIntBuffer = false;
bool g_enable_debug = false;

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


std::string build_network_segmentation(const uint8_t * dlc_buffer_BB, const size_t dlc_size_BB, const char runtime_arg)
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



bool execute_segmentation(cv::Mat &img, int orig_width, int orig_height, int &numberofobj, std::vector<std::vector<float>> &BB_coords, std::vector<std::string> &BB_names) {

    LOGI("execute_segmentation");
    ATrace_beginSection("preprocessing");

    struct timeval start_time, end_time;
    float milli_time, seconds, useconds;

    mtx.lock();
    assert(snpe_BB!=nullptr);

    if(!loadInputUserBuffer_segmentation(applicationInputBuffers, snpe_BB, img, inputMap, bitWidth))
    {
        LOGE("Failed to load Input UserBuffer");
        mtx.unlock();
        return false;
    }

    std::string box_score_coeffs_name = "output1";
    std::string mask_prototypes_name = "output0";

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
        LOGE("Exec status is false");
        mtx.unlock();
        return false;
    }

    const auto& box_score_coeffs_buf = applicationOutputBuffers.at(box_score_coeffs_name);
    const auto& mask_prototypes_buf = applicationOutputBuffers.at(mask_prototypes_name);

    const int num_proposals = 8400;
    const int num_classes = 80;
    const int num_mask_coeffs = 32;
    const int input_width = 640;
    const int input_height = 640;
    const int mask_width = 160;
    const int mask_height = 160;
    const float conf_threshold = 0.25f;
    const float iou_threshold = 0.45f;
    const int proposal_size = 4 + num_classes + num_mask_coeffs; // 116

    std::vector<float> output0_transposed(num_proposals * proposal_size);
    for (int i = 0; i < num_proposals; ++i) {
        for (int j = 0; j < proposal_size; ++j) {
            output0_transposed[i * proposal_size + j] = box_score_coeffs_buf[j * num_proposals + i];
        }
    }

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;
    std::vector<std::vector<float>> mask_coeffs_vec;

    for (int i = 0; i < num_proposals; ++i) {
        float* proposal = output0_transposed.data() + i * proposal_size;
        float* class_scores = proposal + 4;
        auto max_it = std::max_element(class_scores, class_scores + num_classes);
        float confidence = *max_it;
        int class_id = std::distance(class_scores, max_it);

        if (confidence > conf_threshold) {
            float cx = proposal[0];
            float cy = proposal[1];
            float w = proposal[2];
            float h = proposal[3];
            boxes.emplace_back(cx - w / 2, cy - h / 2, w, h);
            confidences.push_back(confidence);
            class_ids.push_back(class_id);
            mask_coeffs_vec.emplace_back(proposal + 4 + num_classes, proposal + proposal_size);
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, iou_threshold, indices);

    numberofobj = indices.size();
    if (numberofobj == 0) {
        mtx.unlock();
        return true;
    }

    float ratio_w = (float)orig_width / input_width;
    float ratio_h = (float)orig_height / input_height;

    for (int idx : indices) {
        if (class_ids[idx] != 0) continue; // Only process "person" class

        cv::Rect box = boxes[idx];
        std::vector<float> singleboxcoords{box.x * ratio_w, box.y * ratio_h, (box.x + box.width) * ratio_w, (box.y + box.height) * ratio_h, milli_time};
        BB_coords.push_back(singleboxcoords);
        BB_names.push_back(classnamemapping[class_ids[idx]]);

        cv::Mat mat_coeffs(1, num_mask_coeffs, CV_32F, mask_coeffs_vec[idx].data());
        cv::Mat mat_prototypes(num_mask_coeffs, mask_width * mask_height, CV_32F, (float*)mask_prototypes_buf.data());
        cv::Mat mat_mul = mat_coeffs * mat_prototypes;

        cv::Mat mask_mat(mask_height, mask_width, CV_32F, mat_mul.data);
        cv::exp(-mask_mat, mask_mat);
        mask_mat = 1.0f / (1.0f + mask_mat);

        cv::Rect box_in_640 = boxes[idx];
        int x_in_160 = round(box_in_640.x / 4.0);
        int y_in_160 = round(box_in_640.y / 4.0);
        int w_in_160 = round(box_in_640.width / 4.0);
        int h_in_160 = round(box_in_640.height / 4.0);

        x_in_160 = std::max(0, x_in_160);
        y_in_160 = std::max(0, y_in_160);
        w_in_160 = std::min(mask_width - x_in_160, w_in_160);
        h_in_160 = std::min(mask_height - y_in_160, h_in_160);

        cv::Mat cropped_mask;
        if (w_in_160 > 0 && h_in_160 > 0) {
            cropped_mask = mask_mat(cv::Rect(x_in_160, y_in_160, w_in_160, h_in_160));
        } else {
            continue;
        }

        cv::Rect final_box(singleboxcoords[0], singleboxcoords[1], singleboxcoords[2]-singleboxcoords[0], singleboxcoords[3]-singleboxcoords[1]);
        if (final_box.width <= 0 || final_box.height <= 0 || final_box.x < 0 || final_box.y < 0 || final_box.x + final_box.width > img.cols || final_box.y + final_box.height > img.rows) continue;

        cv::Mat resized_mask;
        cv::resize(cropped_mask, resized_mask, final_box.size());

        cv::Mat binary_mask = resized_mask > 0.5;

        cv::Scalar color(rand() % 255, rand() % 255, rand() % 255);
        cv::Mat roi = img(final_box);

        for(int r=0; r<final_box.height; ++r) {
            for(int c=0; c<final_box.width; ++c) {
                if(binary_mask.at<uchar>(r,c) > 0) {
                    cv::Vec4b& pixel = roi.at<cv::Vec4b>(r,c);
                    pixel[0] = cv::saturate_cast<uchar>(pixel[0] * 0.5 + color[0] * 0.5); // B
                    pixel[1] = cv::saturate_cast<uchar>(pixel[1] * 0.5 + color[1] * 0.5); // G
                    pixel[2] = cv::saturate_cast<uchar>(pixel[2] * 0.5 + color[2] * 0.5); // R
                }
            }
        }
    }

    if (g_enable_debug) {
        // Save pre-processed input
        const auto& inputNamesOpt = snpe_BB->getInputTensorNames();
        const zdl::DlSystem::StringList& inputNames = *inputNamesOpt;
        std::vector<float>& inputBuffer = applicationInputBuffers.at(inputNames.at(0));
        cv::Mat preprocessed_img(input_height, input_width, CV_32FC3, inputBuffer.data());
        cv::Mat preprocessed_img_8u;
        preprocessed_img.convertTo(preprocessed_img_8u, CV_8UC3, 127.5, 127.5);
        cv::imwrite("/storage/emulated/0/Download/preprocessed_input.png", preprocessed_img_8u);

        // Save output image
        cv::imwrite("/storage/emulated/0/Download/segmentation_output.png", img);
    }

    ATrace_endSection();
    mtx.unlock();
    return true;
}

