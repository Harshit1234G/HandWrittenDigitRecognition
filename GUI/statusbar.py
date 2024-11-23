import customtkinter as ctk
from PIL import Image
from utils.common import upper_frame_color, grey_color, font
from utils.shortcut_top_level import ShortcutWindow


class StatusBar(ctk.CTkFrame):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # shortcut button
        self.shortcut_button = ctk.CTkButton(
            master= self,
            width= 24, 
            height= 24,
            corner_radius= 15,
            fg_color= 'transparent',
            hover_color= upper_frame_color,
            image= ctk.CTkImage(Image.open('icons/shortcuts.png'), size= (24, 24)),
            text= '',
            command= self.create_shortcut_window
        )
        self.shortcut_button.pack(padx= 7, side= 'left')

        # status labels
        self.noise_level = StatusLabel(
            master= self, 
            main_text= 'Noise Level', 
            default_value= '0%'
        )
        self.pensize = StatusLabel(
            master= self, 
            main_text= 'Pensize', 
            default_value= 10
        )
        self.status = StatusLabel(
            master= self, 
            main_text= 'Status', 
            default_value= 'Idle'
        )
        self.draw_coords = StatusLabel(
            master= self, 
            main_text= 'Drawing Coordinates', 
            default_value= (0, 0)
        )


    def create_shortcut_window(self, event: any = None) -> None:
        ShortcutWindow(master= self)


class StatusLabel(ctk.CTkLabel):
    def __init__(
            self, 
            master: any, 
            main_text: str, 
            default_value: any
        ) -> None:
        self.main_text = main_text
        self.default_value = default_value

        super().__init__(
            master= master,
            text_color= grey_color,
            text= f'{self.main_text}: {self.default_value}',
            font= (font, 12),
        )
        self.pack(
            padx= 10,
            side= 'right'
        )

    def update(self, update_value: any) -> None:
        self.configure(text= f'{self.main_text}: {update_value}')

    
    def set_default(self) -> None:
        self.configure(text= f'{self.main_text}: {self.default_value}')