//============================================================================
// Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//============================================================================

package com.qc.objectdetectionYoloNas;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Typeface;
import android.os.Trace;
import android.support.annotation.Nullable;
import android.util.AttributeSet;
import android.view.View;


import java.util.ArrayList;
import java.util.concurrent.locks.ReentrantLock;

/**
 * FragmentRender class is utility for making boxes on camera frames.
 * FragmentRender has utility in fragment_camera.xml and CameraFragment Class
 */
public class FragmentRender extends View {

    private ReentrantLock mLock = new ReentrantLock();
    private ArrayList<RectangleBox> boxlist = new ArrayList<>();

    private Paint mTextColor= new Paint();
    private Paint mBorderColor= new Paint();

    public FragmentRender(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
        init();
    }


    public void setCoordsList(ArrayList<RectangleBox> t_boxlist) {
        mLock.lock();
        if (boxlist == null) {
            boxlist = new ArrayList<>();
        }
        boxlist.clear();
        boxlist.addAll(t_boxlist);
        mLock.unlock();
        postInvalidate();
    }

    public ArrayList<RectangleBox> getBoxlist() {
        mLock.lock();
        try {
            return boxlist;
        } finally {
            mLock.unlock();
        }
    }


    private void init() {
        mTextColor.setTypeface(Typeface.DEFAULT_BOLD);

        mBorderColor.setColor(Color.TRANSPARENT);
        mBorderColor.setColor(Color.MAGENTA);
        mBorderColor.setStyle(Paint.Style.STROKE);
        mBorderColor.setStrokeWidth(6);
        mTextColor.setStyle(Paint.Style.FILL);
        mTextColor.setTextSize(50);
        mTextColor.setColor(Color.RED);
//        mPosepaint= new Paint(Paint.ANTI_ALIAS_FLAG);
//        mPosepaint.setStrokeWidth(8);
    }

    @Override
    protected void onDraw(Canvas canvas) {
        mLock.lock();
        if (boxlist == null || boxlist.isEmpty()) {
            mLock.unlock();
            return;
        }

        // Draw FPS once
        String fps_textLabel = "FPS: " + String.valueOf(boxlist.get(0).fps);
        canvas.drawText(fps_textLabel, 10, 70, mTextColor);

        for (RectangleBox rbox : boxlist) {
            if (rbox.selected) {
                mBorderColor.setColor(Color.GREEN);
            } else {
                mBorderColor.setColor(Color.MAGENTA);
            }

            String processingTimeTextLabel = rbox.processing_time + "ms";

            // Use standard coordinates: left, top, right, bottom
            canvas.drawRect(rbox.left, rbox.top, rbox.right, rbox.bottom, mBorderColor);
            canvas.drawText(rbox.label, rbox.left + 10, rbox.top + 40, mTextColor);
            canvas.drawText(processingTimeTextLabel, rbox.left + 10, rbox.top + 90, mTextColor);
        }
        mLock.unlock();
    }
}
