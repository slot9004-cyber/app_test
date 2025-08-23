package com.qc.objectdetectionYoloNas;

import android.Manifest;
import android.app.Activity;
import android.app.AlertDialog;
import android.app.Dialog;
import android.content.Context;
import android.content.DialogInterface;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Point;
import android.graphics.PorterDuff;
import android.graphics.PorterDuffXfermode;
import android.graphics.RectF;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CameraMetadata;
import android.hardware.camera2.CaptureFailure;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.CaptureResult;
import android.hardware.camera2.TotalCaptureResult;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.ImageReader;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.support.annotation.NonNull;
import android.support.v4.app.DialogFragment;
import android.support.v4.app.Fragment;
import android.support.v4.content.ContextCompat;
import android.util.Size;
import android.util.SparseIntArray;
import android.view.LayoutInflater;
import android.view.MotionEvent;
import android.view.Surface;
import android.view.TextureView;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.RelativeLayout;
import android.widget.Toast;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;

public class CameraFragment extends Fragment {

    private SNPEHelper mSnpeHelper;
    public long tic = 0, tic2 = 0;
    private boolean mNetworkLoaded;
    private FragmentRender mFragmentRender;
    private Button confirmButton;
    private Button backButton;
    private ImageView bottomImageView;
    private ArrayList<RectangleBox> selectedBoxes = new ArrayList<>();
    private boolean isConfirmed = false;

    public int fps = 0, frame_count = -1;
    public static char runtime_var;
    private String mCameraId;
    private static final SparseIntArray ORIENTATIONS = new SparseIntArray();
    private static final int REQUEST_CAMERA_PERMISSION = 1;
    private static final String FRAGMENT_DIALOG = "dialog";

    static {
        ORIENTATIONS.append(Surface.ROTATION_0, 90);
        ORIENTATIONS.append(Surface.ROTATION_90, 0);
        ORIENTATIONS.append(Surface.ROTATION_180, 270);
        ORIENTATIONS.append(Surface.ROTATION_270, 180);
    }

    private static final int STATE_PREVIEW = 0;
    private static final int STATE_WAITING_LOCK = 1;
    private static final int STATE_WAITING_PRECAPTURE = 2;
    private static final int STATE_WAITING_NON_PRECAPTURE = 3;
    private static final int STATE_PICTURE_TAKEN = 4;
    private static final int MAX_PREVIEW_WIDTH = 1920;
    private static final int MAX_PREVIEW_HEIGHT = 1080;

    private TextureView mTextureView;
    private final TextureView.SurfaceTextureListener mSurfaceTextureListener
            = new TextureView.SurfaceTextureListener() {
        @Override
        public void onSurfaceTextureAvailable(SurfaceTexture texture, int width, int height) {
            openCamera(width, height);
        }

        @Override
        public void onSurfaceTextureSizeChanged(SurfaceTexture texture, int width, int height) {
            configureTransform(width, height);
        }

        @Override
        public boolean onSurfaceTextureDestroyed(SurfaceTexture texture) {
            return true;
        }

        @Override
        public void onSurfaceTextureUpdated(SurfaceTexture texture) {
        }
    };

    private CameraCaptureSession mCaptureSession;
    private CameraDevice mCameraDevice;
    private Size mPreviewSize;
    private final CameraDevice.StateCallback mStateCallback = new CameraDevice.StateCallback() {
        @Override
        public void onOpened(@NonNull CameraDevice cameraDevice) {
            mCameraOpenCloseLock.release();
            mCameraDevice = cameraDevice;
            createCameraPreviewSession();
        }

        @Override
        public void onDisconnected(@NonNull CameraDevice cameraDevice) {
            mCameraOpenCloseLock.release();
            cameraDevice.close();
            mCameraDevice = null;
        }

        @Override
        public void onError(@NonNull CameraDevice cameraDevice, int error) {
            mCameraOpenCloseLock.release();
            cameraDevice.close();
            mCameraDevice = null;
            Activity activity = getActivity();
            if (null != activity) {
                activity.finish();
            }
        }
    };

    public static CameraFragment create(Bundle SavedInstanceState) {
        final CameraFragment fragment = new CameraFragment();
        runtime_var = SavedInstanceState.getChar("key");
        return fragment;
    }

    private HandlerThread mBackgroundThread;
    private Handler mBackgroundHandler;
    private ImageReader mImageReader;
    private File mFile;
    private final ImageReader.OnImageAvailableListener mOnImageAvailableListener
            = reader -> mBackgroundHandler.post(new ImageSaver(reader.acquireNextImage(), mFile));

    private CaptureRequest.Builder mPreviewRequestBuilder;
    private CaptureRequest mPreviewRequest;
    private int mState = STATE_PREVIEW;
    private Semaphore mCameraOpenCloseLock = new Semaphore(1);
    private boolean mFlashSupported;
    private int mSensorOrientation;
    private CameraCaptureSession.CaptureCallback mCaptureCallback
            = new CameraCaptureSession.CaptureCallback() {
        private void process(CaptureResult result) {
            // Unused for this feature
        }
        @Override
        public void onCaptureProgressed(@NonNull CameraCaptureSession session,
                                        @NonNull CaptureRequest request,
                                        @NonNull CaptureResult partialResult) {
            process(partialResult);
        }
        @Override
        public void onCaptureCompleted(@NonNull CameraCaptureSession session,
                                       @NonNull CaptureRequest request,
                                       @NonNull TotalCaptureResult result) {
            process(result);
        }
    };

    private void showToast(final String text) {
        final Activity activity = getActivity();
        if (activity != null) {
            activity.runOnUiThread(() -> Toast.makeText(activity, text, Toast.LENGTH_SHORT).show());
        }
    }

    private static Size chooseOptimalSize(Size[] choices, int textureViewWidth,
                                          int textureViewHeight, int maxWidth, int maxHeight, Size aspectRatio) {
        List<Size> bigEnough = new ArrayList<>();
        List<Size> notBigEnough = new ArrayList<>();
        int w = aspectRatio.getWidth();
        int h = aspectRatio.getHeight();
        for (Size option : choices) {
            if (option.getWidth() <= maxWidth && option.getHeight() <= maxHeight &&
                    option.getHeight() == option.getWidth() * h / w) {
                if (option.getWidth() >= textureViewWidth &&
                        option.getHeight() >= textureViewHeight) {
                    bigEnough.add(option);
                } else {
                    notBigEnough.add(option);
                }
            }
        }
        if (bigEnough.size() > 0) {
            return Collections.min(bigEnough, new CompareSizesByArea());
        } else if (notBigEnough.size() > 0) {
            return Collections.max(notBigEnough, new CompareSizesByArea());
        } else {
            return choices[0];
        }
    }

    public static CameraFragment newInstance() {
        return new CameraFragment();
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        return inflater.inflate(R.layout.fragment_camera, container, false);
    }

    @Override
    public void onViewCreated(final View view, Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        mTextureView = view.findViewById(R.id.surface);
        mTextureView.setSurfaceTextureListener(mSurfaceTextureListener);
        mFragmentRender = view.findViewById(R.id.fragmentRender);
        confirmButton = view.findViewById(R.id.confirmButton);
        backButton = view.findViewById(R.id.backButton);
        bottomImageView = view.findViewById(R.id.bottomImageView);

        confirmButton.setOnClickListener(v -> onConfirm());
        backButton.setOnClickListener(v -> onBack());
        mFragmentRender.setOnTouchListener((v, event) -> {
            if (event.getAction() == MotionEvent.ACTION_DOWN) {
                if (!isConfirmed) {
                    handleTouch(event.getX(), event.getY());
                }
            }
            return true;
        });
    }

    @Override
    public void onResume() {
        super.onResume();
        startBackgroundThread();
        if (mTextureView.isAvailable()) {
            openCamera(mTextureView.getWidth(), mTextureView.getHeight());
        } else {
            mTextureView.setSurfaceTextureListener(mSurfaceTextureListener);
        }
        ensureNetCreated();
    }

    @Override
    public void onPause() {
        closeCamera();
        stopBackgroundThread();
        super.onPause();
    }

    private void requestCameraPermission() {
        if (shouldShowRequestPermissionRationale(Manifest.permission.CAMERA)) {
            new ConfirmationDialog().show(getChildFragmentManager(), FRAGMENT_DIALOG);
        } else {
            requestPermissions(new String[]{Manifest.permission.CAMERA}, REQUEST_CAMERA_PERMISSION);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        if (requestCode == REQUEST_CAMERA_PERMISSION) {
            if (grantResults.length != 1 || grantResults[0] != PackageManager.PERMISSION_GRANTED) {
                ErrorDialog.newInstance(getString(R.string.request_permission))
                        .show(getChildFragmentManager(), FRAGMENT_DIALOG);
            }
        } else {
            super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        }
    }

    @SuppressWarnings("SuspiciousNameCombination")
    private void setUpCameraOutputs(int width, int height) {
        Activity activity = getActivity();
        CameraManager mCameraManager = (CameraManager) activity.getSystemService(Context.CAMERA_SERVICE);
        try {
            for (String cameraId : mCameraManager.getCameraIdList()) {
                CameraCharacteristics characteristics
                        = mCameraManager.getCameraCharacteristics(cameraId);
                Integer facing = characteristics.get(CameraCharacteristics.LENS_FACING);
                if (facing != null && facing == CameraCharacteristics.LENS_FACING_FRONT) {
                    continue;
                }
                StreamConfigurationMap map = characteristics.get(
                        CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
                if (map == null) {
                    continue;
                }
                Size largest = Collections.max(
                        Arrays.asList(map.getOutputSizes(ImageFormat.JPEG)),
                        new CompareSizesByArea());
                mImageReader = ImageReader.newInstance(largest.getWidth(), largest.getHeight(),
                        ImageFormat.JPEG, /*maxImages*/2);
                mImageReader.setOnImageAvailableListener(
                        mOnImageAvailableListener, mBackgroundHandler);
                int displayRotation = activity.getWindowManager().getDefaultDisplay().getRotation();
                mSensorOrientation = characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION);
                boolean swappedDimensions = false;
                switch (displayRotation) {
                    case Surface.ROTATION_0:
                    case Surface.ROTATION_180:
                        if (mSensorOrientation == 90 || mSensorOrientation == 270) {
                            swappedDimensions = true;
                        }
                        break;
                    case Surface.ROTATION_90:
                    case Surface.ROTATION_270:
                        if (mSensorOrientation == 0 || mSensorOrientation == 180) {
                            swappedDimensions = true;
                        }
                        break;
                }
                Point displaySize = new Point();
                activity.getWindowManager().getDefaultDisplay().getSize(displaySize);
                int rotatedPreviewWidth = width;
                int rotatedPreviewHeight = height;
                int maxPreviewWidth = displaySize.x;
                int maxPreviewHeight = displaySize.y;
                if (swappedDimensions) {
                    rotatedPreviewWidth = height;
                    rotatedPreviewHeight = width;
                    maxPreviewWidth = displaySize.y;
                    maxPreviewHeight = displaySize.x;
                }
                if (maxPreviewWidth > MAX_PREVIEW_WIDTH) {
                    maxPreviewWidth = MAX_PREVIEW_WIDTH;
                }
                if (maxPreviewHeight > MAX_PREVIEW_HEIGHT) {
                    maxPreviewHeight = MAX_PREVIEW_HEIGHT;
                }
                mPreviewSize = chooseOptimalSize(map.getOutputSizes(SurfaceTexture.class),
                        rotatedPreviewWidth, rotatedPreviewHeight, maxPreviewWidth,
                        maxPreviewHeight, largest);
                mFlashSupported = characteristics.get(CameraCharacteristics.FLASH_INFO_AVAILABLE) != null;
                mCameraId = cameraId;
                return;
            }
        } catch (CameraAccessException | NullPointerException e) {
            e.printStackTrace();
        }
    }

    private void openCamera(int width, int height) {
        if (ContextCompat.checkSelfPermission(getActivity(), Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            requestCameraPermission();
            return;
        }
        setUpCameraOutputs(width, height);
        configureTransform(width, height);
        Activity activity = getActivity();
        CameraManager manager = (CameraManager) activity.getSystemService(Context.CAMERA_SERVICE);
        try {
            if (!mCameraOpenCloseLock.tryAcquire(2500, TimeUnit.MILLISECONDS)) {
                throw new RuntimeException("Time out waiting to lock camera opening.");
            }
            manager.openCamera(mCameraId, mStateCallback, mBackgroundHandler);
        } catch (CameraAccessException | InterruptedException e) {
            throw new RuntimeException("Interrupted or failed to open camera.", e);
        }
    }

    private void closeCamera() {
        try {
            mCameraOpenCloseLock.acquire();
            if (null != mCaptureSession) {
                mCaptureSession.close();
                mCaptureSession = null;
            }
            if (null != mCameraDevice) {
                mCameraDevice.close();
                mCameraDevice = null;
            }
            if (null != mImageReader) {
                mImageReader.close();
                mImageReader = null;
            }
        } catch (InterruptedException e) {
            throw new RuntimeException("Interrupted while trying to lock camera closing.", e);
        } finally {
            mCameraOpenCloseLock.release();
        }
    }

    private void startBackgroundThread() {
        mBackgroundThread = new HandlerThread("CameraBackground");
        mBackgroundThread.start();
        mBackgroundHandler = new Handler(mBackgroundThread.getLooper());
    }

    private void stopBackgroundThread() {
        mBackgroundThread.quitSafely();
        try {
            mBackgroundThread.join();
            mBackgroundThread = null;
            mBackgroundHandler = null;
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    private void createCameraPreviewSession() {
        try {
            SurfaceTexture texture = mTextureView.getSurfaceTexture();
            assert texture != null;
            texture.setDefaultBufferSize(mPreviewSize.getWidth(), mPreviewSize.getHeight());
            Surface surface = new Surface(texture);
            mPreviewRequestBuilder
                    = mCameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
            mPreviewRequestBuilder.addTarget(surface);
            mCameraDevice.createCaptureSession(Arrays.asList(surface), new CameraCapture(),
                    null);
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }

    private void configureTransform(int viewWidth, int viewHeight) {
        Activity activity = getActivity();
        if (null == mTextureView || null == mPreviewSize || null == activity) {
            return;
        }
        int rotation = activity.getWindowManager().getDefaultDisplay().getRotation();
        Matrix matrix = new Matrix();
        RectF viewRect = new RectF(0, 0, viewWidth, viewHeight);
        RectF bufferRect = new RectF(0, 0, mPreviewSize.getHeight(), mPreviewSize.getWidth());
        float centerX = viewRect.centerX();
        float centerY = viewRect.centerY();
        if (Surface.ROTATION_90 == rotation || Surface.ROTATION_270 == rotation) {
            bufferRect.offset(centerX - bufferRect.centerX(), centerY - bufferRect.centerY());
            matrix.setRectToRect(viewRect, bufferRect, Matrix.ScaleToFit.FILL);
            float scale = Math.max(
                    (float) viewHeight / mPreviewSize.getHeight(),
                    (float) viewWidth / mPreviewSize.getWidth());
            matrix.postScale(scale, scale, centerX, centerY);
            matrix.postRotate(90 * (rotation - 2), centerX, centerY);
        } else if (Surface.ROTATION_180 == rotation) {
            matrix.postRotate(180, centerX, centerY);
        }
        mTextureView.setTransform(matrix);
    }

    private void setAutoFlash(CaptureRequest.Builder requestBuilder) {
        if (mFlashSupported) {
            requestBuilder.set(CaptureRequest.CONTROL_AE_MODE,
                    CaptureRequest.CONTROL_AE_MODE_ON_AUTO_FLASH);
        }
    }

    static class CompareSizesByArea implements Comparator<Size> {
        @Override
        public int compare(Size lhs, Size rhs) {
            return Long.signum((long) lhs.getWidth() * lhs.getHeight() -
                    (long) rhs.getWidth() * rhs.getHeight());
        }
    }

    public static class ErrorDialog extends DialogFragment {
        private static final String ARG_MESSAGE = "message";
        public static ErrorDialog newInstance(String message) {
            ErrorDialog dialog = new ErrorDialog();
            Bundle args = new Bundle();
            args.putString(ARG_MESSAGE, message);
            dialog.setArguments(args);
            return dialog;
        }
        @NonNull
        @Override
        public Dialog onCreateDialog(Bundle savedInstanceState) {
            final Activity activity = getActivity();
            return new AlertDialog.Builder(activity)
                    .setMessage(getArguments().getString(ARG_MESSAGE))
                    .setPositiveButton(android.R.string.ok, (dialogInterface, i) -> activity.finish())
                    .create();
        }
    }

    public static class ConfirmationDialog extends DialogFragment {
        @NonNull
        @Override
        public Dialog onCreateDialog(Bundle savedInstanceState) {
            final Fragment parent = getParentFragment();
            return new AlertDialog.Builder(getActivity())
                    .setMessage(R.string.request_permission)
                    .setPositiveButton(android.R.string.ok, (dialog, which) -> parent.requestPermissions(new String[]{Manifest.permission.CAMERA},
                            REQUEST_CAMERA_PERMISSION))
                    .setNegativeButton(android.R.string.cancel,
                            (dialog, which) -> {
                                Activity activity = parent.getActivity();
                                if (activity != null) {
                                    activity.finish();
                                }
                            })
                    .create();
        }
    }

    public class CameraCapture extends CameraCaptureSession.StateCallback {
        @Override
        public void onConfigured(@NonNull CameraCaptureSession cameraCaptureSession) {
            mCaptureSession = cameraCaptureSession;
            try {
                mPreviewRequestBuilder.set(CaptureRequest.CONTROL_AF_MODE,
                        CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE);
                setAutoFlash(mPreviewRequestBuilder);
                mPreviewRequest = mPreviewRequestBuilder.build();
                cameraCaptureSession.setRepeatingRequest(mPreviewRequest, new CameraSession(),
                        mBackgroundHandler);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        @Override
        public void onConfigureFailed(@NonNull CameraCaptureSession cameraCaptureSession) {
        }
    }

    private ArrayList<RectangleBox> trackSelectedObjects(ArrayList<RectangleBox> newBoxes) {
        ArrayList<RectangleBox> nextSelectedBoxes = new ArrayList<>();
        if (!selectedBoxes.isEmpty()) {
            for (RectangleBox selectedBox : selectedBoxes) {
                float minDistance = Float.MAX_VALUE;
                RectangleBox bestMatch = null;
                for (RectangleBox newBox : newBoxes) {
                    if (newBox.label.equals("person")) {
                        float distance = getCenterDistance(selectedBox, newBox);
                        if (distance < minDistance) {
                            minDistance = distance;
                            bestMatch = newBox;
                        }
                    }
                }
                if (bestMatch != null && minDistance < 75) {
                    nextSelectedBoxes.add(bestMatch);
                }
            }
        }
        selectedBoxes = nextSelectedBoxes;

        for (RectangleBox newBox : newBoxes) {
            newBox.selected = false;
            for (RectangleBox selectedBox : selectedBoxes) {
                if (newBox == selectedBox) {
                    newBox.selected = true;
                    break;
                }
            }
        }
        return newBoxes;
    }

    private class CameraSession extends CameraCaptureSession.CaptureCallback {
        @Override
        public void onCaptureCompleted(@NonNull CameraCaptureSession session, @NonNull
                CaptureRequest request, @NonNull TotalCaptureResult result) {
            super.onCaptureCompleted(session, request, result);
            frame_count++;
            try {
                if (frame_count == 0) tic = System.currentTimeMillis();
                else {
                    tic2 = System.currentTimeMillis();
                    fps = (int) (1000 / (tic2 - tic));
                    tic = System.currentTimeMillis();
                }
            } catch (Exception e) { e.printStackTrace(); }

            if (mNetworkLoaded) {
                Bitmap mBitmap = mTextureView.getBitmap();
                if (mBitmap == null) return;

                ArrayList<RectangleBox> newBoxes = new ArrayList<>();
                mSnpeHelper.snpeInference(mBitmap, fps, newBoxes);
                ArrayList<RectangleBox> trackedBoxes = trackSelectedObjects(newBoxes);

                if (!isConfirmed) {
                    mFragmentRender.setCoordsList(trackedBoxes);
                } else {
                    Bitmap bottomBitmap = mBitmap.copy(Bitmap.Config.ARGB_8888, true);
                    Canvas canvas = new Canvas(bottomBitmap);
                    Paint whitePaint = new Paint();
                    whitePaint.setColor(Color.WHITE);
                    whitePaint.setStyle(Paint.Style.FILL);

                    Paint transparentPaint = new Paint();
                    transparentPaint.setColor(Color.TRANSPARENT);
                    transparentPaint.setXfermode(new PorterDuffXfermode(PorterDuff.Mode.SRC));
                    canvas.drawRect(0, 0, bottomBitmap.getWidth(), bottomBitmap.getHeight() / 2, transparentPaint);

                    for (RectangleBox box : trackedBoxes) {
                        if (!box.selected && box.label.equals("person")) {
                            canvas.drawRect(box.left, box.top, box.right, box.bottom, whitePaint);
                        }
                    }
                    final Bitmap finalBottomBitmap = bottomBitmap;
                    if (getActivity() != null) {
                        getActivity().runOnUiThread(() -> bottomImageView.setImageBitmap(finalBottomBitmap));
                    }
                }
            }
        }
    }

    private boolean ensureNetCreated() {
        if (mSnpeHelper == null) {
            mSnpeHelper = new SNPEHelper(getActivity().getApplication());
            new Thread(() -> mNetworkLoaded = mSnpeHelper.loadingMODELS(runtime_var)).start();
        }
        return mNetworkLoaded;
    }

    private void onConfirm() {
        if (selectedBoxes.isEmpty()) {
            showToast("Please select at least one person.");
            return;
        }
        isConfirmed = true;
        confirmButton.setVisibility(View.GONE);
        backButton.setVisibility(View.VISIBLE);
        bottomImageView.setVisibility(View.VISIBLE);
        mFragmentRender.setVisibility(View.GONE);
    }

    private void onBack() {
        isConfirmed = false;
        confirmButton.setVisibility(View.VISIBLE);
        backButton.setVisibility(View.GONE);
        bottomImageView.setVisibility(View.GONE);
        mFragmentRender.setVisibility(View.VISIBLE);
        selectedBoxes.clear();
    }

    private void handleTouch(float x, float y) {
        ArrayList<RectangleBox> currentBoxes = mFragmentRender.getBoxlist();
        if (currentBoxes == null) return;

        RectangleBox tappedBox = null;
        for (RectangleBox box : currentBoxes) {
            if (x >= box.left && x <= box.right && y >= box.top && y <= box.bottom) {
                tappedBox = box;
                break;
            }
        }

        if (tappedBox != null) {
            boolean isAlreadySelected = false;
            int removeIndex = -1;
            for (int i = 0; i < selectedBoxes.size(); i++) {
                if (getCenterDistance(selectedBoxes.get(i), tappedBox) < 20) {
                    isAlreadySelected = true;
                    removeIndex = i;
                    break;
                }
            }

            if (isAlreadySelected) {
                selectedBoxes.remove(removeIndex);
            } else {
                selectedBoxes.add(tappedBox);
            }
        }
    }

    private float getCenterDistance(RectangleBox b1, RectangleBox b2) {
        float b1_cx = (b1.left + b1.right) / 2;
        float b1_cy = (b1.top + b1.bottom) / 2;
        float b2_cx = (b2.left + b2.right) / 2;
        float b2_cy = (b2.top + b2.bottom) / 2;
        return (float) Math.sqrt(Math.pow(b1_cx - b2_cx, 2) + Math.pow(b1_cy - b2_cy, 2));
    }
}
