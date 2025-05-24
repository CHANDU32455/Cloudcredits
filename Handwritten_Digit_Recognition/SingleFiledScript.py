import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import tensorflow as tf
import cv2

# ---- WhiteboardCanvas class ----
class WhiteboardCanvas(tk.Canvas):
    def __init__(self, master=None, width=280, height=280):
        super().__init__(master, width=width, height=height, bg='white', cursor='cross')
        self.width = width
        self.height = height
        self.old_x = None
        self.old_y = None

        self.image = Image.new("L", (width, height), 255)  # white background
        self.draw = ImageDraw.Draw(self.image)

        self.bind("<B1-Motion>", self.paint)
        self.bind("<ButtonRelease-1>", self.reset)

        clear_btn = tk.Button(master, text="Clear Whiteboard", command=self.clear)
        clear_btn.pack()

        save_btn = tk.Button(master, text="Save Image", command=self.save_image_dialog)
        save_btn.pack()

    # ---- Save image dialog function ----
    def save_image_dialog(self):
            filepath = filedialog.asksaveasfilename(defaultextension=".png",
                                             filetypes=[("PNG files", "*.png")])
            if filepath:
                self.save(filepath)
    
    def save(self, filepath='digit.png'):
        self.image.save(filepath)

    def paint(self, event):
        if self.old_x and self.old_y:
            self.create_line(self.old_x, self.old_y, event.x, event.y,
                             width=12, fill='black', capstyle=tk.ROUND, smooth=tk.TRUE)
            self.draw.line([self.old_x, self.old_y, event.x, event.y],
                           fill=0, width=12)
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x = None
        self.old_y = None

    def clear(self):
        self.delete("all")
        self.image = Image.new("L", (self.width, self.height), 255)
        self.draw = ImageDraw.Draw(self.image)

    def get_image(self):
        return self.image.copy()

def preprocess_image(img):
    if isinstance(img, Image.Image):
        img = img.convert('L')
        img = np.array(img)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = 255 - img  # invert

    # Threshold to binary image
    _, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)

    coords = cv2.findNonZero(img)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        img = img[y:y+h, x:x+w]

        # Resize to fit in 20x20 box while maintaining aspect ratio
        max_dim = max(w, h)
        scale = 20 / max_dim
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Create 28x28 image and paste the resized digit centered
        padded_img = np.zeros((28, 28), dtype=np.uint8)
        x_offset = (28 - new_w) // 2
        y_offset = (28 - new_h) // 2
        padded_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img

        padded_img = padded_img.astype('float32') / 255.0
        return padded_img.reshape(1, 28, 28, 1)
    
    # fallback if blank image
    blank = np.zeros((28, 28), dtype='float32')
    return blank.reshape(1, 28, 28, 1)

# ---- Load model ----
model = tf.keras.models.load_model('models/mnist_cnn_model.h5')

# ---- Prediction function ----
def predict_digit(img):
    img = preprocess_image(img)
    prediction = model.predict(img)
    return np.argmax(prediction), max(prediction[0])

# ---- Upload image function ----
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

# ---- Predict from whiteboard function ----
def predict_from_whiteboard():
    img_pil = whiteboard.get_image()
    
    # Check if the image is effectively blank (all white)
    if np.array(img_pil).mean() > 250:  # almost all pixels white
        result_label.config(text="please draw Visibible...")
        panel.config(image=None)
        panel.image = None
        return

    digit, prob = predict_digit(img_pil)
    result_label.config(text=f'Predicted Digit: {digit} ({prob:.2f} confidence)')

    img_display = img_pil.resize((200, 200))
    panel_img = ImageTk.PhotoImage(img_display)
    panel.config(image=panel_img)
    panel.image = panel_img


# ---- GUI Setup ----
root = tk.Tk()
root.title("MNIST Digit Recognizer")
root.geometry("500x650")

btn_font = ("Helvetica", 14)
label_font = ("Helvetica", 18, "bold")

btn_frame = tk.Frame(root)
btn_frame.pack(pady=20)

upload_btn = tk.Button(btn_frame, text="Upload Image", command=upload_image, font=btn_font, width=20)
upload_btn.pack(side=tk.LEFT, padx=15)

predict_btn = tk.Button(btn_frame, text="Predict from Whiteboard", command=predict_from_whiteboard, font=btn_font, width=25)
predict_btn.pack(side=tk.LEFT, padx=15)

panel = tk.Label(root)
panel.pack(pady=20)

result_label = tk.Label(root, text="Prediction Result", font=label_font, fg="blue")
result_label.pack(pady=10)

whiteboard = WhiteboardCanvas(root, width=300, height=300)
whiteboard.pack(pady=20)


root.mainloop()

