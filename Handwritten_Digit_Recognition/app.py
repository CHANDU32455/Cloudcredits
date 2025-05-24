import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import cv2
from utils import preprocess_image
from whiteboard import WhiteboardCanvas

model = tf.keras.models.load_model('models/mnist_cnn_model.h5')

def predict_digit(img):
    img = preprocess_image(img)
    prediction = model.predict(img)
    return np.argmax(prediction), max(prediction[0])

def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path).convert('L')
        img_resized = img.resize((200, 200))
        img_display = ImageTk.PhotoImage(img_resized)
        panel.config(image=img_display)
        panel.image = img_display

        digit, prob = predict_digit(img)
        result_label.config(text=f'Predicted Digit: {digit} ({prob:.2f} confidence)')

def predict_from_whiteboard():
    img_pil = whiteboard.get_image()
    digit, prob = predict_digit(img_pil)
    result_label.config(text=f'Predicted Digit: {digit} ({prob:.2f} confidence)')

    img_display = img_pil.resize((200, 200))
    panel_img = ImageTk.PhotoImage(img_display)
    panel.config(image=panel_img)
    panel.image = panel_img


# GUI Setup
root = tk.Tk()
root.title("MNIST Digit Recognizer")
root.geometry("500x650")  # Set a bigger window size

# Style Configuration
btn_font = ("Helvetica", 14)
label_font = ("Helvetica", 18, "bold")

# Button Frame
btn_frame = tk.Frame(root)
btn_frame.pack(pady=20)

upload_btn = tk.Button(btn_frame, text="Upload Image", command=upload_image, font=btn_font, width=20)
upload_btn.pack(side=tk.LEFT, padx=15)

predict_btn = tk.Button(btn_frame, text="Predict from Whiteboard", command=predict_from_whiteboard, font=btn_font, width=25)
predict_btn.pack(side=tk.LEFT, padx=15)

# Image Display Panel
panel = tk.Label(root)
panel.pack(pady=20)

# Prediction Result
result_label = tk.Label(root, text="Prediction Result", font=label_font, fg="blue")
result_label.pack(pady=10)

# Whiteboard Canvas
whiteboard = WhiteboardCanvas(root, width=300, height=300)
whiteboard.pack(pady=20)

root.mainloop()
