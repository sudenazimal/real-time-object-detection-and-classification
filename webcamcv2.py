import cv2
import torch

# load the trained model 
model = torch.hub.load("ultralytics/yolov5", "custom", path=r"C:\Users\sude_\Masaüstü\webcam_stream\best.pt",force_reload=True)

# initialize webcam
cap = cv2.VideoCapture(0)  # 0 is the default webcam

# define the class names 
classes = ['without_mask', 'with_mask', 'mask_weared_incorrect']

while True:
    # capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break


    results = model(frame)

    # parse the results 
    predictions = results.pred[0]  # extract the prediction from the model

    for det in predictions:
        if det[4] > 0.5:  
            x1, y1, x2, y2 = map(int, det[:4])  
            class_id = int(det[5])  
            label = classes[class_id]  
            confidence = det[4]  

            # draw bounding box and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # display the resulting frame with detections
    cv2.imshow("Real-time Face Mask Detection", frame)

    # exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
