package com.qc.objectdetectionYoloNas;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.support.v4.app.FragmentTransaction;
import android.support.v7.app.AppCompatActivity;
import android.view.WindowManager;
import android.widget.RadioGroup;

import org.opencv.android.OpenCVLoader;

public class MainActivity extends AppCompatActivity {

    static {
        System.loadLibrary("objectdetectionYoloNas");
    }

    // TODO: 預設值可依需求調整（C/G/D/N）
    public static char runtime_var = 'D';

    private static final int REQ_CAMERA = 1;

    private RadioGroup rg;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main_activity);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        OpenCVLoader.initDebug();

        rg = (RadioGroup) findViewById(R.id.rg1);
        rg.setOnCheckedChangeListener(new RadioGroup.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(RadioGroup group, int checkedId) {

                // 用 if-else 取代 switch，避免「constant expression required」
                if (checkedId == R.id.CPU) {
                    runtime_var = 'C';
                    overToCamera(runtime_var);
                    System.out.println("CPU instance running");
                } else if (checkedId == R.id.GPU) {
                    runtime_var = 'G';
                    overToCamera(runtime_var);
                    System.out.println("GPU instance running");
                } else if (checkedId == R.id.DSP) {
                    runtime_var = 'D';
                    overToCamera(runtime_var);
                    System.out.println("DSP instance running");
                } else {
                    runtime_var = 'N';
                    overToCamera(runtime_var);
                    System.out.println("Do Nothing");
                }
            }
        });
    }

    /**
     * Method to request Camera permission
     */
    private void cameraPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            requestPermissions(new String[]{Manifest.permission.CAMERA}, REQ_CAMERA);
        }
    }

    /**
     * Method to navigate to CameraFragment along with runtime choice
     */
    private void overToCamera(char runtime_value) {
        boolean passToFragment;

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            passToFragment = (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED);
        } else {
            passToFragment = true;
        }

        if (passToFragment) {
            FragmentTransaction transaction = getSupportFragmentManager().beginTransaction();
            Bundle args = new Bundle();
            args.putChar("key", runtime_value);
            transaction.replace(R.id.main_content, CameraFragment.create(args));
            transaction.commit();
        } else {
            cameraPermission();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        // 若尚未選取，維持預設 runtime_var；此處可觸發一次導向
        overToCamera(runtime_var);
    }

    @Override
    protected void onStop() {
        super.onStop();
    }

    // 如需處理權限回調，加入以下方法
    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        if (requestCode == REQ_CAMERA) {
            boolean granted = true;
            if (grantResults == null || grantResults.length == 0) {
                granted = false;
            } else {
                for (int r : grantResults) {
                    if (r != PackageManager.PERMISSION_GRANTED) {
                        granted = false;
                        break;
                    }
                }
            }
            if (granted) {
                overToCamera(runtime_var);
            }
        } else {
            super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        }
    }
}

