package com.qc.objectdetectionYoloNas;

import android.graphics.Bitmap;

import org.opencv.core.Mat;
import org.opencv.core.CvType;
import org.opencv.android.Utils;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.Scalar;
import org.opencv.core.Core;
import org.opencv.dnn.Dnn;

public class ImageProcessor {

    public static Mat preprocess(Bitmap image, Mat mask, float maskedFill) {
        // Convert bitmap to Mat
        Mat imgMat = new Mat();
        Utils.bitmapToMat(image, imgMat);
        Imgproc.cvtColor(imgMat, imgMat, Imgproc.COLOR_RGBA2RGB);

        // Normalize image to [-1.0, 1.0]
        imgMat.convertTo(imgMat, CvType.CV_32F, 1.0 / 127.5, -1.0);

        // Convert mask to float and ensure it has 3 channels
        Mat maskF = new Mat();
        mask.convertTo(maskF, CvType.CV_32F, 1.0 / 255.0);
        Mat mask3 = new Mat();
        Core.merge(java.util.Arrays.asList(maskF, maskF, maskF), mask3);

        // Apply mask: masked = img * (1.0 - m) + fill * m
        Mat m_inv = new Mat();
        Core.subtract(new Scalar(1.0, 1.0, 1.0), mask3, m_inv);

        Mat masked_part = new Mat();
        Core.multiply(imgMat, m_inv, masked_part);

        Mat fill_part = new Mat();
        Core.multiply(new Scalar(maskedFill, maskedFill, maskedFill), mask3, fill_part);

        Mat masked = new Mat();
        Core.add(masked_part, fill_part, masked);

        return masked;
    }

    public static Bitmap postprocess(Mat ganOutput, Mat originalImage, Mat mask) {
        // Denormalize GAN output
        ganOutput.convertTo(ganOutput, CvType.CV_8UC3, 127.5, 127.5);

        // Ensure original image is 8UC3
        if (originalImage.type() != CvType.CV_8UC3) {
            originalImage.convertTo(originalImage, CvType.CV_8UC3);
        }

        // Create 3-channel mask
        Mat mask3 = new Mat();
        Core.merge(java.util.Arrays.asList(mask, mask, mask), mask3);
        mask3.convertTo(mask3, CvType.CV_32F, 1.0 / 255.0);

        // Convert images to float for calculation
        ganOutput.convertTo(ganOutput, CvType.CV_32F);
        originalImage.convertTo(originalImage, CvType.CV_32F);

        // Combine: comp = (pred * m3 + base * (1 - m3))
        Mat m_inv = new Mat();
        Core.subtract(new Scalar(1.0, 1.0, 1.0), mask3, m_inv);

        Mat comp1 = new Mat();
        Core.multiply(ganOutput, mask3, comp1);

        Mat comp2 = new Mat();
        Core.multiply(originalImage, m_inv, comp2);

        Mat comp = new Mat();
        Core.add(comp1, comp2, comp);

        // Convert to Bitmap
        comp.convertTo(comp, CvType.CV_8UC3);
        Imgproc.cvtColor(comp, comp, Imgproc.COLOR_RGB2RGBA);
        Bitmap resultBitmap = Bitmap.createBitmap(comp.cols(), comp.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(comp, resultBitmap);

        return resultBitmap;
    }
}
