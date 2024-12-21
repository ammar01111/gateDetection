from ultralytics import YOLO
import cv2

# Paths
model_path = r'C:\Users\PC\Desktop\gateDetection-20241221T173614Z-001\gateDetection\weights\last.pt'
image_path = r"C:\Users\PC\Desktop\ICBJ.v1-versi-24-september-2022-19.40.yolov11\train\images\Dataset_Panitia_Main_Gate_mp4-193_jpg.rf.870af137efe19ff128bea43455438350.jpg"

# Load image
img = cv2.imread(image_path)
if img is None:
    raise ValueError("Image could not be loaded. Please check the image path.")

# Load YOLO model
model = YOLO(model_path)

# Run inference
results = model(image_path)

# Process results (assuming pose estimation model)
for result in results:
    if hasattr(result, 'keypoints') and result.keypoints is not None:
        keypoints_array = result.keypoints.xy.cpu().numpy()  # Convert to NumPy array
        print("Keypoints Array Shape:", keypoints_array.shape)  # Debugging shape
        print("Keypoints Array Data:", keypoints_array)        # Debugging content

        # Iterate through detections
        for detection in keypoints_array:
            # Iterate through keypoints in a detection
            for keypoint_indx, keypoint in enumerate(detection):
                x, y = keypoint  # Extract x and y
                x, y = int(x), int(y)  # Convert to integers
                cv2.putText(img, str(keypoint_indx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(img, (x, y), 3, (255, 0, 0), -1)
    else:
        print("No keypoints found in this result.")

# Display image with keypoints
cv2.imshow('Pose Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
