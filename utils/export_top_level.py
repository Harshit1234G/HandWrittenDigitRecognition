import customtkinter as ctk
from PIL import ImageTk
from typing import Self
import utils.common as common


class ExportWindow(ctk.CTkToplevel):
    instance: Self | None = None  # class attribute to hold the single instance

    def __init__(self, *args, **kwargs) -> None:
        # prevent creating a new instance if one already exists
        if ExportWindow.instance is not None:
            if ExportWindow.instance.state() == 'iconic':  # check if minimized
                ExportWindow.instance.deiconify()
                
            ExportWindow.instance.focus_set()
            ExportWindow.instance.lift()
            return None

        super().__init__(*args, **kwargs)

        # save this instance
        ExportWindow.instance = self

        # when the window is closed, reset the instance
        self.protocol('WM_DELETE_WINDOW', self.on_close)

        # basic configuration
        self.geometry('500x500')
        self.title('Export Data')
        self.resizable(False, False)

        common.center_window(self)

        # setting icon
        icon = ImageTk.PhotoImage(file= 'icons/app.png')
        self.wm_iconbitmap()
        self.after(200, lambda: self.iconphoto(False, icon))

        # setting focus
        self.after(200, self.lift)
        self.after(200, self.focus_set)


    def on_close(self):
        ExportWindow.instance = None
        self.destroy()
