import cv2
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("gesture_model.h5")

# Load class names
class_names = os.listdir("data")

# Define image dimensions
img_height, img_width = 224, 224

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    img = cv2.resize(frame, (img_height, img_width))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Make prediction
    predictions = model.predict(img)
    predicted_class = class_names[np.argmax(predictions)]

    # Display the result
    cv2.putText(frame, f"Gesture: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
