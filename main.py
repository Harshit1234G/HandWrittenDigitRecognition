import customtkinter as ctk
from PIL import ImageTk
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
from GUI.draw_frame import DrawFrame
from GUI.metrics_frame import MetricsFrame
from GUI.statusbar import StatusBar
from utils.common import NDArrayFloat
from utils.export_top_level import ExportWindow


class MainWindow(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()

        # setting some basic things
        ctk.set_appearance_mode('light')

        self.geometry('1366x768')
        self.title('Hand Written Digit Recognition')

        self.imagepath = ImageTk.PhotoImage(file= 'icons/app.png')
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
            statusbar= self.statusbar,
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
            statusbar= self.statusbar,
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
        self.draw_frame.export_button.configure(command= self.create_export_window)

        # configuring metrics_frame
        self.metrics_frame.load_data_button.configure(command= self.load_data_from_history)

        # Bind the close event to the on_closing function
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # shortcut binding
        # basic actions
        self.bind('<Control-p>', self.predict)
        self.bind('<Control-Delete>', self.clear)
        self.bind('<Control-s>', self.create_export_window)

        # History and shortcut panel
        self.bind('<Control-Shift-L>', self.load_data_from_history)
        self.bind('<Control-Shift-T>', self.metrics_frame.clear_all_history)
        self.bind('<Control-period>', self.statusbar.create_shortcut_window)

        # Metrics toggels
        self.bind('<Control-m><Key-1>', lambda _: self.metrics_frame.checkbox_shortcut_callback(name= 'accuracy'))
        self.bind('<Control-m><Key-2>', lambda _: self.metrics_frame.checkbox_shortcut_callback(name= 'confidence'))
        self.bind('<Control-m><Key-3>', lambda _: self.metrics_frame.checkbox_shortcut_callback(name= 'cm'))
        self.bind('<Control-m><Key-4>', lambda _: self.metrics_frame.checkbox_shortcut_callback(name= 'count'))

        # Correction actions
        self.bind('<Control-Shift-C>', lambda _: self.metrics_frame.correct_wrong_callback(value= 'Correct'))
        self.bind('<Control-Shift-W>', lambda _: self.metrics_frame.correct_wrong_callback(value= 'Wrong'))
        self.bind('<Control-Shift-S>', self.metrics_frame.update_history)

        # updating status
        self.statusbar.status.update('Program loaded successfully')


    def predict(self, event: any = None) -> None:
        # if button is disabled then preventing shortcut key to work
        if self.draw_frame.predict_button.cget('state') == 'disabled':
            return None

        self.statusbar.status.update('Predicting...')

        # processing digit
        np_img: NDArrayFloat = self.draw_frame.process_digit()

        # predicting
        probas: NDArrayFloat = self.model.predict(np_img)[0]

        # setting attributes of MetricsFrame class
        self.metrics_frame.original_image = self.draw_frame.draw_image
        self.metrics_frame.probabilities = probas
        self.metrics_frame.prediction = probas.argmax()

        self.metrics_frame.update_all()
        self.statusbar.status.update('Prediction completed')


    def clear(self, event: any = None) -> None:
        self.draw_frame.clear_canvas()
        self.metrics_frame.clear_prediction()
        self.statusbar.status.update('Drawing canvas and current prediction is cleared')

    
    def load_data_from_history(self, event: any = None) -> None:
        if self.metrics_frame.history.empty:
            self.statusbar.status.update('No data in history')
            return None
        
        # loading the selected data and updating the metrics_frame class attributes
        self.metrics_frame.update_attributes()

        # drawing the original image on canvas
        self.draw_frame.draw_image_on_canvas(self.metrics_frame.original_image)


    def create_export_window(self, event: any = None) -> None:
        ExportWindow(master= self)


    def on_closing(self):
        plt.close("all")   # Close any Matplotlib figures
        self.destroy()     # Destroy the Tkinter window
        sys.exit()         # Exit the program completely
        

if __name__ == '__main__':
    app = MainWindow()
    app.mainloop()
    