import customtkinter as ctk
from PIL import Image, ImageDraw, ImageOps
import joblib
import numpy as np


# loading model
model = joblib.load('stacking_clf.pkl')

root = ctk.CTk()
root.title("Handwritten Digit Recognition")
ctk.set_appearance_mode('light')

# Creating canvas
canvas_size = 280
canvas = ctk.CTkCanvas(root, width= canvas_size, height= canvas_size, bg= "white")
canvas.grid(row= 0, column= 0, pady= 20, padx= 20)

# drawing objects
draw_image = Image.new(
    mode= "L", 
    size= (canvas_size, canvas_size), 
    color= 255
)
draw = ImageDraw.Draw(draw_image)

photo_image = None

def draw_digit(event):
    x, y = event.x, event.y
    r = 10  # Brush size
    # Draw on both canvas and the image
    canvas.create_rectangle(x - r, y - r, x + r, y + r, fill= "black", width= 0)
    draw.rectangle([x - r, y - r, x + r, y + r], fill= 0)

def clear_canvas():
    canvas.delete("all")
    draw.rectangle([0, 0, canvas_size, canvas_size], fill= 255)
    prediction_label.configure(text= 'Prediction: ')
    root.update()

def process_digit():
    global photo_image
    # Resize to 28x28 for model input
    small_image = draw_image.resize((28, 28), Image.LANCZOS)
    # Invert to match MNIST black-on-white
    small_image = ImageOps.invert(small_image)  
    # small_image.show()

    pixel_data = list(small_image.getdata())
    return pixel_data

def predict():
    img_digit = process_digit()
    np_img = np.array(img_digit).reshape((1, -1))
    
    probabilities = {str(i): proba for i, proba in enumerate(model.predict_proba(np_img)[0])}
    print(probabilities)

    prediction = model.predict(np_img)[0]
    prediction_label.configure(text= f'Prediction: {prediction}')
    root.update()


canvas.bind("<B1-Motion>", draw_digit)

prediction_label = ctk.CTkLabel(root, text= 'Prediction: ')
prediction_label.grid(row= 1, column= 0, sticky=  'nsew')

ctk.CTkButton(root, text= "Clear", command= clear_canvas).grid(row= 2, column= 0, sticky= "we", padx= 10, pady= 10)
ctk.CTkButton(root, text= "Predict", command= predict).grid(row= 3, column= 0, sticky= "we", padx= 10, pady= 10)

root.mainloop()
