import customtkinter as ctk
import tensorflow as tf

class GUI(ctk.CTk):
    model = tf.keras.models.load_model('kaggle/working/handwritten_digit_rec.keras')
    def __init__(self) -> None:
        ...