import customtkinter as ctk
from PIL import ImageTk, Image
from functools import partial
from utils.common import font, grey_color, fg_color, upper_frame_color


class MessageBox(ctk.CTkToplevel):
    def __init__(
        self,
        title_of_box: str,
        msg: str,
        type: str,
        *args,
        **kwargs
    ) -> None:

        super().__init__(*args, **kwargs)

        self.title(title_of_box)
        self.geometry('300x100')
        self.resizable(False, False)

        ctk.set_appearance_mode('light')

        # changing icon
        self.imagepath = ImageTk.PhotoImage(file= 'icons/app.png')
        self.wm_iconbitmap()
        self.after(300, lambda: self.iconphoto(False, self.imagepath))

        img_path = f'icons/{type}.png'

        self.icon = ctk.CTkImage(
            light_image= Image.open(img_path),
            size= (30, 30)
        )

        self.message = ctk.CTkLabel(
            master= self,
            image= self.icon,
            text= msg,
            compound= 'left',
            wraplength= 230,
            font= (font, 12),
            text_color= grey_color,
            anchor= 'center'
        )

        self.message.pack(padx= 10, pady= 10)

        self.ok_button = ctk.CTkButton(
            master= self,
            width= 50,
            text= 'OK',
            command= self.destroy,
            corner_radius= 5,
            font= (font, 12),
            text_color= grey_color,
            border_color= fg_color,
            fg_color= upper_frame_color,
            hover= False,
            border_width= 2
        )
        self.ok_button.pack(
            pady= 10, 
            padx= 10, 
            side= 'right', 
            anchor= 's'
        )

        self.center_window()
        self.focus_set()

        self.bind("<Escape>", lambda _: self.destroy())


    def center_window(self) -> None:
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")


# partial objects
ShowWarning = partial(MessageBox, title_of_box= 'Warning', type= 'warning')
ShowError = partial(MessageBox, title_of_box= 'Error', type= 'error')