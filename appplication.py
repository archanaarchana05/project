import cv2
import numpy as np
from sklearn import svm

# Load the Haar Cascade classifiers for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Initialize SVM classifier
svm_classifier = svm.SVC()

# Function to extract features from the eye region
def extract_eye_features(eye_image):
    # Implement your feature extraction logic here
    return eye_image.flatten()

# Function to preprocess the dataset and labels for training SVM
def preprocess_dataset(dataset, labels):
    # Implement any preprocessing steps here
    return dataset, labels

# Load your dataset and labels for training SVM
dataset, labels = np.load('dataset.npy'), np.load('labels.npy')

# Preprocess the dataset and labels
dataset, labels = preprocess_dataset(dataset, labels)

# Train the SVM classifier
svm_classifier.fit(dataset, labels)

# Initialize video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect eyes in the face region
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        for (ex,ey,ew,eh) in eyes:
            eye_image = roi_gray[ey:ey+eh, ex:ex+ew]
            eye_features = extract_eye_features(eye_image)
            predicted_label = svm_classifier.predict([eye_features])[0]
            
            # Do something based on the predicted label (e.g., perform an action corresponding to the blink pattern)

            # Draw rectangle around the eyes
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    # Display the frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
