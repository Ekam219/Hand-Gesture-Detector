import cv2
import os

# Define the number of images to collect per gesture
num_images = 100
gesture_name = "wave"
output_dir = os.path.join("data", gesture_name)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

cap = cv2.VideoCapture(0)
i = 0

while i < num_images:
    ret, frame = cap.read()
    if not ret:
        break

    img_name = os.path.join(output_dir, f"{gesture_name}_{i}.jpg")
    cv2.imwrite(img_name, frame)
    cv2.imshow("Frame", frame)
    
    i += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
