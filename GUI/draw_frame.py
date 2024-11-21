import customtkinter as ctk
from PIL import Image, ImageDraw, ImageOps, ImageTk
import utils.common as common
from typing import Callable
import numpy as np


class DrawFrame(ctk.CTkFrame):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.noise_is_added: bool = False

        ctk.CTkLabel(
            self,
            text= 'Draw a number between 0 to 9 in the box:',
            text_color= common.grey_color,
            anchor= 'w',
            font= (common.font, 12)
        ).grid(row= 0, column= 0, sticky= 'w', padx= 20)

        # Creating canvas
        self.canvas_size: int = 280
        self.canvas = ctk.CTkCanvas(
            self, 
            width= self.canvas_size, 
            height= self.canvas_size, 
            bg= 'white',
            cursor= 'tcross'
        )
        self.canvas.grid(
            row= 1, 
            column= 0, 
            pady= (0, 20), 
            padx= 20
        )

        # drawing objects
        self.draw_image = Image.new(
            mode= 'L', 
            size= (self.canvas_size, self.canvas_size), 
            color= 255
        )
        self.draw = ImageDraw.Draw(self.draw_image)

        # binding mouse movement
        self.canvas.bind('<B1-Motion>', self.draw_digit)

        # setting default layouts
        self.default_draw_frame_layout()

        # for storing the original image
        self.set_original_image()

    
    def default_draw_frame_layout(self) -> None:
        # predict button
        self.predict_button = ctk.CTkButton(
            self, 
            **common.button_kwargs,
            text= 'Predict',
            state= 'disabled'
            )
        self.predict_button.grid(
            row= 2,
            **common.button_grid_kwargs
        )

        # clear button
        self.clear_button = ctk.CTkButton(
            self, 
            **common.button_kwargs,
            text= 'Clear'
        )
        self.clear_button.grid(
            row= 3, 
            **common.button_grid_kwargs
        )

        common.create_canvas_and_line(self, row= 4)

        # Import Button
        self.import_button = ctk.CTkButton(
            self, 
            **common.button_kwargs,
            text= 'Import'
            )
        self.import_button.grid(
            row= 5, 
            **common.button_grid_kwargs
        )

        # Export Button
        self.export_button = ctk.CTkButton(
            self, 
            **common.button_kwargs,
            text= 'Export'
            )
        self.export_button.grid(
            row= 6, 
            **common.button_grid_kwargs
        )
        
        common.create_canvas_and_line(self, row= 7)

        # text variables
        self.noise_var = ctk.DoubleVar(self, value= 0.0)
        self.pensize_var = ctk.IntVar(self, value= 10)

        # noise slider
        self.noise_slider = self.create_slider_frame(
            label= 'Noise     ',
            from_= 0.0,
            to= 0.5,
            variable= self.noise_var,
            command= self.add_noise
        )
        self.noise_slider.grid(
            row= 8,
            column= 0,
            padx= 20,
            pady= (0, 10),
            sticky= 'nsew'
        )

        # pensize slider
        self.pensize_slider = self.create_slider_frame(
            label= 'Pen Size',
            from_= 5,
            to= 20,
            variable= self.pensize_var
        )
        self.pensize_slider.grid(
            row= 9,
            column= 0,
            padx= 20,
            pady= (0, 10),
            sticky= 'nsew'
        )
        

    def draw_digit(self, event: any) -> None:
        x, y = event.x, event.y
        pen_size: int = self.pensize_var.get()

        # Draw on both canvas and the image
        self.canvas.create_rectangle(
            x - pen_size, y - pen_size, x + pen_size, y + pen_size, 
            fill= 'black'
        )
        self.draw.rectangle(
            [x - pen_size, y - pen_size, x + pen_size, y + pen_size], 
            fill= 0
        )

        self.set_original_image()
        self.predict_button.configure(state= 'normal')


    def clear_canvas(self) -> None:
        self.canvas.delete('all')
        self.draw.rectangle(
            [0, 0, self.canvas_size, self.canvas_size], 
            fill= 255
        )

        self.set_original_image()
        self.noise_var.set(0.0)
        self.noise_is_added = False
        self.update()
        self.predict_button.configure(state= 'disabled')


    def process_digit(self) -> common.NDArrayFloat:
        # Resize to 28x28 for model input
        small_image = self.draw_image.resize((28, 28), Image.LANCZOS)
        # Invert to match MNIST black-on-white
        small_image = ImageOps.invert(small_image)

        pixel_data = list(small_image.getdata())
        # reshpaing and normalization
        np_img = np.array(pixel_data).reshape(-1, 28, 28, 1) / 255
        return np_img


    def create_slider_frame(
        self,
        *,
        label: str,
        from_: float,
        to: float,
        variable: ctk.IntVar | ctk.DoubleVar,
        command: Callable | None = None
    ) -> ctk.CTkFrame:
        slider_frame = ctk.CTkFrame(
            self, 
            width= 280,
            height= 45,
            fg_color= common.upper_frame_color
        )

        ctk.CTkLabel(
            master= slider_frame,
            text= label,
            text_color= common.grey_color,
            anchor= 'w',
            font= (common.font, 12)
        ).grid(row= 0, column= 0, padx= 5, pady= 5, sticky= 'w')

        slider = ctk.CTkSlider(
            master= slider_frame,
            button_color= common.fg_color,
            button_hover_color= common.hover_color,
            progress_color= common.fg_color, 
            from_= from_,
            to= to,
            variable= variable,
            command= command
        )
        slider.grid(row= 0, column= 1, padx= (0, 5), pady= 5, sticky= 'e')

        return slider_frame
    

    def set_original_image(self) -> None:
        if not self.noise_is_added:
            self.original_image: Image.Image = ImageOps.invert(self.draw_image.copy())
    

    def add_noise(self, event: any) -> None:
        self.noise_is_added = True

        # getting noise level from slider
        noise_level: float = self.noise_var.get()

        # converting to numpy
        image_array = np.array(self.original_image, dtype= np.int32)
        noise = np.random.normal(scale= noise_level * 255, size= image_array.shape)
    
        # Add noise to the image
        noisy_image = image_array + noise

        # Clip values to stay in the range [0, 255]
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

        # updating draw_image
        self.draw_image = ImageOps.invert(Image.fromarray(noisy_image))
        self.draw = ImageDraw.Draw(self.draw_image)

        # updating canvas
        self.canvas.delete("all")
        self.noisy_image_tk = ImageTk.PhotoImage(self.draw_image)
        self.canvas.create_image(0, 0, anchor= 'nw', image= self.noisy_image_tk)
        self.canvas.update()
        self.update()


    def draw_image_on_canvas(self, image: Image.Image) -> None:
        self.canvas.delete("all")
        self.loaded_img_tk = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor= 'nw', image= self.loaded_img_tk)
        self.canvas.update()
        self.update()
