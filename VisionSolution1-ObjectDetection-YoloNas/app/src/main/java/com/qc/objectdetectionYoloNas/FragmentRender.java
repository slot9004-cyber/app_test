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
        postInvalidate();

        if (boxlist==null)
        {
            mLock.unlock();
            return;
        }
        boxlist.clear();
        for(int j=0;j<t_boxlist.size();j++) {
            System.out.println("writing boxList in java");
            boxlist.add(t_boxlist.get(j));
        }
        mLock.unlock();
        postInvalidate();
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
            float y_coord = rbox.left;
            float y1_coord = rbox.right;
            float x_coord = rbox.top;
            float x1_coord = rbox.bottom;

            if (rbox.selected) {
                mBorderColor.setColor(Color.GREEN);
            } else {
                mBorderColor.setColor(Color.MAGENTA);
            }

            String processingTimeTextLabel = rbox.processing_time + "ms";

            canvas.drawRect(x1_coord, y_coord, x_coord, y1_coord, mBorderColor);
            canvas.drawText(rbox.label, x1_coord + 10, y_coord + 40, mTextColor);
            canvas.drawText(processingTimeTextLabel, x1_coord + 10, y_coord + 90, mTextColor);
        }
        mLock.unlock();
    }
}
