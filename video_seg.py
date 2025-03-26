import cv2

from yoloseg import YOLOSeg

# # Initialize video
cap = cv2.VideoCapture("../file.avi")
cap.set(cv2.CAP_PROP_FPS, 20)

# Initialize YOLOv5 Instance Segmentator
model_path = "../best.dlc"
yoloseg = YOLOSeg(model_path, conf_thres=0.25, iou_thres=0.3)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output2.mp4", fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
i = 0
while cap.isOpened():

    # Read frame from the video
    ret, frame = cap.read()
    if not ret:
        break
    if i == 20:
        break
    # Update object localizer
    boxes, scores, class_ids, masks = yoloseg(frame)

    combined_img = yoloseg.draw_masks(frame, mask_alpha=0.4)
    out.write(combined_img)
    i += 1
    # cv2.imshow("Detected Objects", combined_img)

cap.release()
out.release()