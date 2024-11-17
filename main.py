import customtkinter as ctk
import os
from PIL import ImageTk
import tensorflow as tf
import pandas as pd
from GUI.draw_frame import DrawFrame
from GUI.metrics_frame import MetricsFrame
from GUI.statusbar import StatusBar
from utils.common import button_kwargs, button_grid_kwargs, NDArrayFloat


class MainWindow(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()

        self.current_probabilities: NDArrayFloat | None = None

        # setting some basic things
        ctk.set_appearance_mode('light')

        self.geometry('1300x600')
        self.title('Hand Written Digit Recognition')

        self.imagepath = ImageTk.PhotoImage(file= os.path.join('icons', 'app.png'))
        self.wm_iconbitmap()
        self.iconphoto(False, self.imagepath)

        # loading model
        self.model = tf.keras.models.load_model('kaggle/working/handwritten_digit_rec.keras')

        # status bar
        self.statusbar: ctk.CTkFrame = StatusBar(
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
        self.metrics_frame: ctk.CTkScrollableFrame = MetricsFrame(
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
        self.draw_frame: ctk.CTkFrame = DrawFrame(
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

        # predict button
        self.predict_button = ctk.CTkButton(
            self.draw_frame, 
            **button_kwargs,
            text= 'Predict', 
            command= self.predict)
        self.predict_button.grid(
            row= 2, 
            **button_grid_kwargs
        )

        # clear button
        self.clear_button = ctk.CTkButton(
            self.draw_frame, 
            **button_kwargs,
            text= 'Clear', 
            command= self.draw_frame.clear_canvas
        )
        self.clear_button.grid(
            row= 3, 
            **button_grid_kwargs
        )

        # Import Button
        self.import_button = ctk.CTkButton(
            self.draw_frame, 
            **button_kwargs,
            text= 'Import', 
            command= ...)
        self.import_button.grid(
            row= 5, 
            **button_grid_kwargs
        )

        # Export Button
        self.export_button = ctk.CTkButton(
            self.draw_frame, 
            **button_kwargs,
            text= 'Export', 
            command= ...)
        self.export_button.grid(
            row= 6, 
            **button_grid_kwargs
        )

    def predict(self) -> None:
        np_img = self.draw_frame.process_digit()
        self.current_probabilities = self.model.predict(np_img)[0]
        print(self.current_probabilities.argmax())
        print(self.current_probabilities.round(2) * 100)
        

if __name__ == '__main__':
    app = MainWindow()
    app.mainloop()