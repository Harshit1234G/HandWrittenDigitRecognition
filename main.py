import customtkinter as ctk
import os
from PIL import ImageTk
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
from GUI.draw_frame import DrawFrame
from GUI.metrics_frame import MetricsFrame
from GUI.statusbar import StatusBar
from utils.common import NDArrayFloat


class MainWindow(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        # setting some basic things
        ctk.set_appearance_mode('light')

        self.geometry('1366x768')
        self.title('Hand Written Digit Recognition')

        self.imagepath = ImageTk.PhotoImage(file= os.path.join('icons', 'app.png'))
        self.wm_iconbitmap()
        self.iconphoto(False, self.imagepath)

        # loading model
        self.model = tf.keras.models.load_model('kaggle/working/handwritten_digit_rec.keras')

        # status bar
        self.statusbar = StatusBar(
            self,
            height= 30,
            corner_radius= 15
        )
        self.statusbar.pack(
            fill= 'x',
            side= 'bottom',
            anchor= 's',
            padx= 7
        )

        # this frame would contain all the graphs and other stuffs.
        self.metrics_frame = MetricsFrame(
            self, 
            corner_radius= 15
        )
        self.metrics_frame.pack(
            side= 'right',
            fill= 'both',
            expand= True,
            anchor= 's',
            pady= 7,
            padx= (0, 7)
        )

        # this frame would contain drawing area and buttons
        self.draw_frame = DrawFrame(
            self,
            width= 320,
            corner_radius= 15
        )
        self.draw_frame.pack(
            fill= 'y',
            expand= True,
            anchor= 'w',
            pady= 7,
            padx= 7
        )

        # configuring draw_frame
        self.draw_frame.clear_button.configure(command= self.clear)
        self.draw_frame.predict_button.configure(command= self.predict)

        # Bind the close event to the on_closing function
        self.protocol("WM_DELETE_WINDOW", self.on_closing)


    def predict(self) -> None:
        # processing digit
        np_img = self.draw_frame.process_digit()

        # predicting
        probas: NDArrayFloat = self.model.predict(np_img)[0]

        # setting attributes of MetricsFrame class
        self.metrics_frame.original_image = self.draw_frame.draw_image
        self.metrics_frame.resized_image = np_img
        self.metrics_frame.probabilities = probas
        self.metrics_frame.prediction = probas.argmax()

        self.metrics_frame.update_all()


    def clear(self) -> None:
        self.draw_frame.clear_canvas()
        self.metrics_frame.clear_prediction()


    def on_closing(self):
        plt.close("all")   # Close any Matplotlib figures
        self.destroy()     # Destroy the Tkinter window
        sys.exit()         # Exit the program completely
        

if __name__ == '__main__':
    app = MainWindow()
    app.mainloop()