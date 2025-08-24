//============================================================================
// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//============================================================================

#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
using namespace cv;
#include <jni.h>
#include <string>
#include <iostream>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/bitmap.h>

#include "hpp/inference.h"
#include "hpp/Util.hpp"

#include "zdl/SNPE/SNPE.hpp"
#include "zdl/SNPE/SNPEFactory.hpp"

extern "C" JNIEXPORT jstring JNICALL
Java_com_qc_objectdetectionYoloNas_SNPEHelper_queryRuntimes(
        JNIEnv* env,
        jobject /* this */,
        jstring native_dir_path) {
    const char *cstr = env->GetStringUTFChars(native_dir_path, nullptr);
    env->ReleaseStringUTFChars(native_dir_path, cstr);

    std::string runT_Status;
    std::string nativeLibPath = std::string(cstr);

//    runT_Status += "\nLibs Path : " + nativeLibPath + "\n";

    if (!SetAdspLibraryPath(nativeLibPath)) {
        __android_log_print(ANDROID_LOG_INFO, "SNPE ", "Failed to set ADSP Library Path\n");

        runT_Status += "\nFailed to set ADSP Library Path\nTerminating";
        return env->NewStringUTF(runT_Status.c_str());
    }

    // ====================================================================================== //
    runT_Status = "Querying Runtimes : \n\n";
    // DSP unsignedPD check
    if (!zdl::SNPE::SNPEFactory::isRuntimeAvailable(zdl::DlSystem::Runtime_t::DSP,zdl::DlSystem::RuntimeCheckOption_t::UNSIGNEDPD_CHECK)) {
        __android_log_print(ANDROID_LOG_INFO, "SNPE ", "UnsignedPD DSP runtime : Absent\n");
        runT_Status += "UnsignedPD DSP runtime : Absent\n";
    }
    else {
        __android_log_print(ANDROID_LOG_INFO, "SNPE ", "UnsignedPD DSP runtime : Present\n");
        runT_Status += "UnsignedPD DSP runtime : Present\n";
    }
    // DSP signedPD check
    if (!zdl::SNPE::SNPEFactory::isRuntimeAvailable(zdl::DlSystem::Runtime_t::DSP)) {
        __android_log_print(ANDROID_LOG_INFO, "SNPE ", "DSP runtime : Absent\n");
        runT_Status += "DSP runtime : Absent\n";
    }
    else {
        __android_log_print(ANDROID_LOG_INFO, "SNPE ", "DSP runtime : Present\n");
        runT_Status += "DSP runtime : Present\n";
    }
    // GPU check
    if (!zdl::SNPE::SNPEFactory::isRuntimeAvailable(zdl::DlSystem::Runtime_t::GPU)) {
        __android_log_print(ANDROID_LOG_INFO, "SNPE ", "GPU runtime : Absent\n");
        runT_Status += "GPU runtime : Absent\n";
    }
    else {
        __android_log_print(ANDROID_LOG_INFO, "SNPE ", "GPU runtime : Present\n");
        runT_Status += "GPU runtime : Present\n";
    }
    // CPU check
    if (!zdl::SNPE::SNPEFactory::isRuntimeAvailable(zdl::DlSystem::Runtime_t::CPU)) {
        __android_log_print(ANDROID_LOG_INFO, "SNPE ", "CPU runtime : Absent\n");
        runT_Status += "CPU runtime : Absent\n";
    }
    else {
        __android_log_print(ANDROID_LOG_INFO, "SNPE ", "CPU runtime : Present\n");
        runT_Status += "CPU runtime : Present\n";
    }

    return env->NewStringUTF(runT_Status.c_str());
}



//initializing network
extern "C"
JNIEXPORT jstring JNICALL
Java_com_qc_objectdetectionYoloNas_SNPEHelper_initSNPE(JNIEnv *env, jobject thiz, jobject asset_manager, jchar runtime) {
    LOGI("Reading SNPE DLC ...");
    std::string result;

    AAssetManager* mgr = AAssetManager_fromJava(env, asset_manager);
    AAsset* asset_BB = AAssetManager_open(mgr, "yolo11s-seg.dlc", AASSET_MODE_UNKNOWN);
    if (NULL == asset_BB) {
        LOGE("Failed to load ASSET, needed to load DLC\n");
        result = "Failed to load ASSET, needed to load DLC\n";
        return env->NewStringUTF(result.c_str());
    }

    long dlc_size_BB = AAsset_getLength(asset_BB);
    LOGI("DLC BB Size = %ld MB\n", dlc_size_BB / (1024*1024));
    result += "DLC BB Size = " + std::to_string(dlc_size_BB);
    char* dlc_buffer_BB = (char*) malloc(sizeof(char) * dlc_size_BB);
    AAsset_read(asset_BB, dlc_buffer_BB, dlc_size_BB);

    result += "\n\nBuilding Models DLC Network:\n";
    result += build_network_BB(reinterpret_cast<const uint8_t *>(dlc_buffer_BB), dlc_size_BB,runtime);

    return env->NewStringUTF(result.c_str());
}


//inference
extern "C"
JNIEXPORT jint JNICALL
Java_com_qc_objectdetectionYoloNas_SNPEHelper_inferSNPE(JNIEnv *env, jobject thiz, jlong inputMat, jobject maskBitmap,
                                               jobjectArray selectedBoxCoords, jint actual_width, jint actual_height,
                                               jobjectArray jboxcoords, jobjectArray objnames) {

    LOGI("infer SNPE S");

    cv::Mat &img = *(cv::Mat*) inputMat;
    std::string bs;
    int numberofobj = 0;
    std::vector<std::vector<float>> BB_coords;
    std::vector<std::string> BB_names;
    cv::Mat combined_mask;
    std::vector<cv::Rect> selected_boxes;

    int num_selected = env->GetArrayLength(selectedBoxCoords);
    for (int i = 0; i < num_selected; i++) {
        jfloatArray box_array = (jfloatArray)env->GetObjectArrayElement(selectedBoxCoords, i);
        jfloat* box_ptr = env->GetFloatArrayElements(box_array, 0);
        selected_boxes.push_back(cv::Rect(box_ptr[0], box_ptr[1], box_ptr[2] - box_ptr[0], box_ptr[3] - box_ptr[1]));
        env->ReleaseFloatArrayElements(box_array, box_ptr, 0);
    }

    bool status = executeDLC(img, actual_width, actual_height, selected_boxes, numberofobj, BB_coords, BB_names, combined_mask);

    if(numberofobj ==0)
        {
        LOGI("No object detected");
    }
    else if(status == false)
    {
        LOGE("fatal ERROR");
        return 0;
    }
    else {
        for (int z = 0; z < numberofobj; z++){
            jfloatArray boxcoords = (jfloatArray) env->GetObjectArrayElement(jboxcoords, z);
            env->SetObjectArrayElement(objnames, z,env->NewStringUTF(BB_names[z].data()));

            float tempbox[5]; //4 coords and 1 processing time
            for(int k=0;k<5;k++)
                tempbox[k]=BB_coords[z][k];
            env->SetFloatArrayRegion(boxcoords,0,5,tempbox);
        }

        // Convert cv::Mat mask to Android Bitmap
        AndroidBitmapInfo mask_info;
        void* mask_pixels;
        if (AndroidBitmap_getInfo(env, maskBitmap, &mask_info) < 0) {
            LOGE("Failed to get bitmap info");
            return 0;
        }
        if (mask_info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
            LOGE("Bitmap format is not RGBA_8888");
            return 0;
        }
        if (AndroidBitmap_lockPixels(env, maskBitmap, &mask_pixels) < 0) {
            LOGE("Failed to lock bitmap pixels");
            return 0;
        }

        cv::Mat mask_rgba(mask_info.height, mask_info.width, CV_8UC4, mask_pixels);
        cv::Mat gray_mask;
        cv::cvtColor(combined_mask, gray_mask, cv::COLOR_GRAY2BGRA); // Convert single channel to 4 channels
        gray_mask.copyTo(mask_rgba);

        AndroidBitmap_unlockPixels(env, maskBitmap);
    }
    return numberofobj;
}