import cv2
import numpy as np
from PIL import Image

def preprocess_image(img):
    if isinstance(img, Image.Image):
        img = img.convert('L')  # grayscale
        img = np.array(img)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Invert image: MNIST format is white background, black digit
    img = 255 - img

    # Crop to bounding box of the digit
    coords = cv2.findNonZero(img)  # get all non-zero points (digit)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)  # get bounding box
        img = img[y:y+h, x:x+w]  # crop the digit region

    # Resize to 28x28 while keeping aspect ratio
    img = cv2.resize(img, (20, 20), interpolation=cv2.INTER_AREA)
    padded_img = np.pad(img, ((4, 4), (4, 4)), mode='constant', constant_values=0)

    # Normalize
    padded_img = padded_img.astype('float32') / 255.0
    padded_img = padded_img.reshape(1, 28, 28, 1)

    return padded_img
