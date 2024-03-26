import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras import layers, models

# Define the preprocessing function to handle GIF images
def preprocess_image(image_path, target_size):
    gif = cv2.VideoCapture(image_path)
    ret, frame = gif.read()
    gif.release()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, target_size)
    img = img / 255.0  # Normalize pixel values to range [0, 1]
    return img

# Load and preprocess the dataset
def load_dataset(dataset_path, target_size, test_size=0.2):
    X = []
    y = []
    for folder_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder_name)
        if os.path.isdir(folder_path):
            label = int(folder_name)
            label -= 100  # Adjust label to match the range [0, 899]
            for filename in os.listdir(folder_path):
                if filename.endswith('.gif'):
                    image_path = os.path.join(folder_path, filename)
                    img = preprocess_image(image_path, target_size)
                    X.append(img)
                    y.append(label)
    X = np.array(X).reshape(-1, target_size[0], target_size[1], 1)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

# Define the CNN model
def create_model(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),  # Increased complexity
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),  # Added dropout layer for regularization
        layers.Dense(256, activation='relu'),  # Increased complexity
        layers.Dense(900, activation='softmax')  # 900 classes for numbers 100 to 999
    ])
    return model

# Load the dataset
dataset_path = r'C:\Users\Brahim\Desktop\model'
target_size = (80, 150)  # Updated target size to match the image dimensions
X_train, X_test, y_train, y_test = load_dataset(dataset_path, target_size)

# Define the CNN model
model = create_model(input_shape=(target_size[0], target_size[1], 1))

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=40, batch_size=16, validation_data=(X_test, y_test))

# Save the trained model
model.save('model.h5')

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)

# Make predictions
def predict_number(image_path):
    img = preprocess_image(image_path, target_size)
    img = img.reshape(1, target_size[0], target_size[1], 1)
    prediction = model.predict(img)
    predicted_number = np.argmax(prediction) + 100  # Convert predicted index to number between 100 and 999
    return predicted_number

# Example usage:
input_image_path = r"C:\Users\Brahim\Desktop\885\huyg.gif"
predicted_number = predict_number(input_image_path)
print("Predicted Number:", predicted_number)
