# Setting Up All Artifacts details

## Give appropriate permission to the directory "FOLDER_WITH_ARTIFACTS" you are working with
import os
os.environ['SNPE_ROOT']="/mnt/c/Qualcomm/AIStack/QAIRT/2.31.0.250130" #set up your snpe path here.
os.environ['RAW_FILE_FOLDER']="raw"
os.environ['DLC32']="models/yolo_nas_fp32.dlc"
os.environ['DLC8']="models/yolo_nas_w8a8.dlc"
os.environ['TARGET_INPUT_LIST']="input.txt"
os.environ['ONDEVICE_FOLDER']="yolonas"
os.environ['DEVICE_HOST']="localhost"
os.environ['DEVICE_ID']="de41ee15" #change with your device-id. Use command "adb devices" to get devices names.
os.environ['SNPE_TARGET_ARCH']="aarch64-android"
os.environ['SNPE_TARGET_STL']="libc++_shared.so"

## Note- Use python3.8 or above for generating onnx
# !pip install super-gradients==3.1.2
import torch
from super_gradients.training import models
from super_gradients.common.object_names import Models
import cv2
import numpy as np
import os
import onnxruntime as ort

## Getting The dataset
# Please, fill coco dataset link in below code block. You might need 10-15 images for quantization.
# !wget http://images.cocodataset.org/zips/val2017.zip -q --show-progress
# !unzip val2017.zip
# !mkdir "raw"

files = os.listdir('val2017') #val2017 is the datatset folder path. Keeping only 15 images.
for file in files[15:]:
    os.remove("val2017/"+file)

# Remove zip file
import subprocess
subprocess.run(["rm", "-rf", "val2017.zip"])

## Getting the ONNX Model
os.makedirs('models', exist_ok=True)

# YOLO_NAS_S Model has issues while download. To overcome, follow below steps:
# In your python enviroment, update super gradients package (3.1.2) with the modified YOLO_NAS_S download link.
# Code Change 1#:
# In /usr/venv/lib/python3.10/site-packages/super_gradients/training/pretrained_models.py line 47
# Existing Code: "yolo_nas_s_coco": "https://sghub.deci.ai/models/yolo_nas_s_coco.pth",
# Modify: "yolo_nas_s_coco": "https://sg-hub-nv.s3.amazonaws.com/models/yolo_nas_s_coco.pth",
# Code Change 2#:
# In /usr/venv/lib/python3.10/site-packages/super_gradient/training/utils/checkpoint_utils.py line 316, Function Name:load_pretrained_weightsline
# Existing code: unique_filename = url.split("https://sghub.deci.ai/models/")[1].replace("/", "_").replace(" ", "_")
# Modify: unique_filename = url.split("https://sg-hub-nv.s3.amazonaws.com/models/")[1].replace("/", "_").replace(" ", "_")

model = models.get(Models.YOLO_NAS_S, pretrained_weights="coco")
# Prpare model for conversion
# Input size is in format of [Batch x Channels x Width x Height] where 640 is the standard COCO dataset dimensions
model.eval()
model.prep_model_for_conversion(input_size=[1, 3, 320, 320])
# Create dummy_input
dummy_input = torch.randn([1, 3, 320, 320], device="cpu")
# Convert model to onnx
torch.onnx.export(model, dummy_input, "models/yolo_nas_s.onnx", opset_version=11)

#### Getting the FP32 Model
# The following bash commands assume you are running in a shell environment
# If running in a notebook, use %%bash magic or subprocess as needed
# Example with subprocess:
subprocess.run(["bash", "-c", "source $SNPE_ROOT/bin/envsetup.sh && snpe-onnx-to-dlc -i models/yolo_nas_s.onnx -o app/src/main/assets/yolo_nas_s.dlc --out_node 887 --out_node 879"])

## Preprocessing
def preprocess(original_image):
    resized_image = cv2.resize(original_image, (320, 320))
    resized_image = resized_image/255
    return resized_image

##Please download Coco2014 dataset and give the path here
dataset_path = "val2017/"
os.makedirs("rawYoloNAS", exist_ok=True)
filenames=[]
for path in os.listdir(dataset_path):
    # check if current path is a file
    if os.path.isfile(os.path.join(dataset_path, path)):
        filenames.append(os.path.join(dataset_path, path))
for filename in filenames:
    original_image = cv2.imread(filename)
    img = preprocess(original_image)
    img = img.astype(np.float32)
    img.tofile("raw/"+filename.split("/")[-1].split(".")[0]+".raw")

# Generate input.txt
subprocess.run(["bash", "-c", "find raw -name *.raw > input.txt"])

## Quantize the DLC
subprocess.run(["bash", "-c", "source $SNPE_ROOT/bin/envsetup.sh && snpe-dlc-quantize --input_dlc app/src/main/assets/yolo_nas_s.dlc --input_list input.txt --output_dlc app/src/main/assets/Quant_intermediate_yoloNas_s_320.dlc"])

# - Based on the device where you will execute the model set --htp_socs to sm8750 or sm8650 or sm8550

subprocess.run(["bash", "-c", "source $SNPE_ROOT/bin/envsetup.sh && snpe-dlc-graph-prepare --input_dlc app/src/main/assets/Quant_intermediate_yoloNas_s_320.dlc --set_output_tensors=887,879 --htp_socs=sm8650 --output_dlc=app/src/main/assets/Quant_yoloNas_s_320.dlc"])

# Load ONNX model
onnx_model_path = "models/yolo_nas_s.onnx"
session = ort.InferenceSession(onnx_model_path)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Directory to save outputs
os.makedirs("outputs", exist_ok=True)

def yolo_postprocess(output_boxes, output_scores, original_image, input_image_shape=(320, 320), conf_threshold=0.5, nms_threshold=0.4):
    """
    Postprocess YOLO model outputs and annotate the input image.
    Args:
        output_boxes: Model output for bounding boxes (numpy array).
        output_scores: Model output for class scores (numpy array).
        original_image: Original input image to annotate.
        input_image_shape: Shape of the input image to the model (default: 320x320).
        conf_threshold: Confidence threshold to filter weak detections.
        nms_threshold: IoU threshold for non-maximum suppression.
    Returns:
        List of detected objects with bounding boxes, class IDs, and confidences.
    """
    import cv2

    original_image_shape = original_image.shape[:2]
    height_ratio = original_image_shape[0] / input_image_shape[0]
    width_ratio = original_image_shape[1] / input_image_shape[1]

    boxes = []
    confidences = []
    class_ids = []

    for box, scores in zip(output_boxes, output_scores):
        if len(scores) == 0:
            continue  # Skip this detection

        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.25:  # Print detections with confidence > 0.25
            print(f"Detection: Class {class_id}, Confidence: {confidence:.2f}, Box: {box}")

        if confidence > conf_threshold:
            center_x, center_y, width, height = box * [input_image_shape[1], input_image_shape[0], input_image_shape[1], input_image_shape[0]]
            x = int((center_x - width / 2) * width_ratio)
            y = int((center_y - height / 2) * height_ratio)
            w = int(width * width_ratio)
            h = int(height * height_ratio)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

    # Apply Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    results = []
    for i in indices:
        i = i[0]
        box = boxes[i]
        x, y, w, h = box
        confidence = confidences[i]
        class_id = class_ids[i]

        # Annotate the image
        cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"Class {class_id}: {confidence:.2f}"
        cv2.putText(original_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        results.append({
            "box": box,
            "confidence": confidence,
            "class_id": class_id
        })

    return results

# Run inference on dataset
for filename in filenames:
    original_image = cv2.imread(filename)
    img = preprocess(original_image)
    img = img.astype(np.float32)
    img = np.expand_dims(img.transpose(2, 0, 1), axis=0)  # Convert to [1, 3, 320, 320]

    # Run inference
    outputs = session.run([output_name], {input_name: img})

    # Debugging: Print the outputs length and content
    print(f"Number of outputs: {len(outputs)}")
    for i, output in enumerate(outputs):
        print(f"Output[{i}] shape: {output.shape}")

    # Adjust postprocess call based on available outputs
    if len(outputs) == 1:
        detections = yolo_postprocess(outputs[0].squeeze(), None, original_image, input_image_shape=(320, 320))
    elif len(outputs) > 1:
        detections = yolo_postprocess(outputs[0].squeeze(), outputs[1].squeeze(), original_image, input_image_shape=(320, 320))
    else:
        raise ValueError("No outputs returned from the model.")

    # # Save the image with detections
    # output_file = os.path.join("rawYoloNAS", filename.split("/")[-1].replace(".jpg", "_output.jpg"))
    # cv2.imwrite(output_file, original_image)

    # # Save bounding boxes and class labels to separate files
    # boxes_file = os.path.join("rawYoloNAS", filename.split("/")[-1].replace(".jpg", "_boxes.txt"))
    # classes_file = os.path.join("rawYoloNAS", filename.split("/")[-1].replace(".jpg", "_classes.txt"))

    # with open(boxes_file, "w") as bf, open(classes_file, "w") as cf:
    #     for detection in detections:
    #         x, y, w, h = detection["box"]
    #         confidence = detection["confidence"]
    #         class_id = detection["class_id"]

    #         # Write bounding box coordinates
    #         bf.write(f"{x} {y} {w} {h}\n")

    #         # Write class ID and confidence
    #         cf.write(f"{class_id} {confidence:.2f}\n")