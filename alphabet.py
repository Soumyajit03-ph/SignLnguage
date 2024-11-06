import tensorflow as tf
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import TensorBoard
import mediapipe as mp

# Mediapipe setup for hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def mediapipe_hand_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False  # Image is no longer writeable
    results = model.process(image)  # Make prediction
    image.flags.writeable = True  # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_hand_landmarks(image, results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

def extract_hand_keypoints(results):
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]  # Use the first detected hand
        hand_keypoints = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()
    else:
        hand_keypoints = np.zeros(21 * 3)  # 21 hand landmarks
    return hand_keypoints

# Data path and actions
DATA_PATH = os.path.join('MP_Data_2')
actions = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O']
#['O','P','Q','R','S','T','U','V','W','X','Y','Z']
no_sequences = 50
sequence_length = 50

# Prepare directories for data collection
for action in actions:  # Only loop over new actions
    for sequence in range(no_sequences):
        path = os.path.join(DATA_PATH, action, str(sequence))
        if not os.path.exists(path):
            os.makedirs(path)

# #Data collection
# cap = cv2.VideoCapture(0)
# with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
#     for action in actions:  # Only collect for new actions
#         for sequence in range(no_sequences):
#             for frame_num in range(sequence_length):
#                 ret, frame = cap.read()
#                 image, results = mediapipe_hand_detection(frame, hands)
#                 draw_hand_landmarks(image, results)
#
#                 if frame_num == 0:
#                     cv2.putText(image, 'STARTING COLLECTION', (120, 200),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
#                     cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}',
#                                 (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
#                     cv2.imshow('OpenCV Feed', image)
#                     cv2.waitKey(2000)
#                 else:
#                     cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}',
#                                 (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
#                     cv2.imshow('OpenCV Feed', image)
#
#                 keypoints = extract_hand_keypoints(results)
#                 np.save(os.path.join(DATA_PATH, action, str(sequence), str(frame_num)), keypoints)
#
#                 if cv2.waitKey(10) & 0xFF == ord('m'):
#                     break
# cap.release()
# cv2.destroyAllWindows()


# Prepare data for training
label_map = {label: num for num, label in enumerate(actions)}
sequences, labels = [], []


for action in actions:
    for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)


model_path = 'action.h5'

# Check if model file exists
if os.path.exists(model_path):
    # Load the pre-trained model
    print("Loading existing model...")
    model = load_model(model_path)
else:
    # LSTM model training
    log_dir = os.path.join('Logs')
    tb_callback = TensorBoard(log_dir=log_dir)

    model = Sequential()
    model.add(LSTM(264, return_sequences=True, activation='relu', input_shape=(sequence_length, 63)))  # 63 for hand keypoints
    # model.add(BatchNormalization())
    # model.add(Dropout(0.4))
    model.add(LSTM(264, return_sequences=True, activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.4))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.3))
    model.add(LSTM(128, return_sequences=False, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(len(actions), activation='softmax'))

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback])
    model.save(model_path)

#Evaluation during training
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")


#Evaluation matrices
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Generate predictions
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

# Print classification report
print("Classification Report:")
print(classification_report(y_true_labels, y_pred_labels, target_names=actions))

# Plot confusion matrix
conf_matrix = confusion_matrix(y_true_labels, y_pred_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=actions, yticklabels=actions)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

print("Unique classes in y_true_labels:", np.unique(y_true_labels))
print("Unique classes in y_pred_labels:", np.unique(y_pred_labels))

# Detection in real time with consistency threshold
sequence = []
sentence = []
predictions = []
threshold = 0.3
consistency_threshold = 5  # Number of consecutive frames with the same prediction
consistent_count = 0
current_action = None

cap = cv2.VideoCapture(0)
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_hand_detection(frame, hands)
        draw_hand_landmarks(image, results)

        # Extract hand keypoints
        keypoints = extract_hand_keypoints(results)
        sequence.insert(0, keypoints)
        sequence = sequence[:50]

        if len(sequence) == 50:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            predicted_action = actions[np.argmax(res)]
            confidence = res[np.argmax(res)]

            # Check if the confidence is above the threshold
            if confidence > threshold:
                if predicted_action == current_action:
                    # Increase consistency count if the same action is predicted
                    consistent_count += 1
                else:
                    # Reset for a new action
                    consistent_count = 1
                    current_action = predicted_action

                # Only update the sentence if consistent for enough frames
                if consistent_count >= consistency_threshold:
                    # Clear the sentence and add the new consistent action
                    sentence = [current_action]

        # Display the current consistent action on the screen
        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('m'):
            break
    cap.release()
    cv2.destroyAllWindows()


