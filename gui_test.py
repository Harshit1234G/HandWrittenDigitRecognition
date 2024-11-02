import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import joblib
import numpy as np


# loading model
model = joblib.load('trial_model.pkl')

# Initialize main window
root = tk.Tk()
root.title("Handwritten Digit Recognition")

# Create canvas
canvas_size = 280
canvas = tk.Canvas(root, width=canvas_size, height=canvas_size, bg="white")
canvas.grid(row=0, column=0, pady=20, padx=20)

# Prepare to draw
draw_image = Image.new("L", (canvas_size, canvas_size), 255)
draw = ImageDraw.Draw(draw_image)

def draw_digit(event):
    x, y = event.x, event.y
    r = 10  # Brush size
    # Draw on both canvas and the image
    canvas.create_oval(x - r, y - r, x + r, y + r, fill="black", width=0)
    draw.ellipse([x - r, y - r, x + r, y + r], fill=0)

def clear_canvas():
    canvas.delete("all")
    draw.rectangle([0, 0, canvas_size, canvas_size], fill=255)

def process_digit():
    # Resize to 28x28 for model input
    small_image = draw_image.resize((28, 28), Image.LANCZOS)
    small_image = ImageOps.invert(small_image)  # Invert to match MNIST black-on-white
    # small_image.show()  # Show the resized image for testing

    # Convert to pixel data (array format) if needed
    pixel_data = list(small_image.getdata())
    return pixel_data

def predict():
    img_digit = process_digit()
    np_img = np.array(img_digit).reshape((1, -1))
    # print(type(img_digit))
    print(model.predict(np_img))

# Bind drawing function
canvas.bind("<B1-Motion>", draw_digit)

# Add buttons
tk.Button(root, text="Clear", command=clear_canvas).grid(row=1, column=0, sticky="we", padx=10, pady=10)
tk.Button(root, text="Predict", command=predict).grid(row=2, column=0, sticky="we", padx=10, pady=10)

root.mainloop()
