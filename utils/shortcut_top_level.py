import customtkinter as ctk
from PIL import ImageTk
from typing import Self
#TODO: Add a frame then create a label that shows the shortcut keys

class ShortcutWindow(ctk.CTkToplevel):
    instance: Self | None = None  # class attribute to hold the single instance

    def __init__(self, *args, **kwargs) -> None:
        # prevent creating a new instance if one already exists
        if ShortcutWindow.instance is not None:
            if ShortcutWindow.instance.state() == 'iconic':  # check if minimized
                ShortcutWindow.instance.deiconify()
                
            ShortcutWindow.instance.focus_set()
            ShortcutWindow.instance.lift()
            return

        super().__init__(*args, **kwargs)

        # save this instance
        ShortcutWindow.instance = self

        # when the window is closed, reset the instance
        self.protocol('WM_DELETE_WINDOW', self.on_close)

        # basic configuration
        self.geometry('500x500')
        self.title('Shortcuts')
        self.resizable(False, False)

        self.center_window()

        # setting icon
        icon = ImageTk.PhotoImage(file= 'icons/app.png')
        self.wm_iconbitmap()
        self.after(200, lambda: self.iconphoto(False, icon))

        # setting focus
        self.after(200, self.lift)
        self.after(200, self.focus_set)


    def center_window(self) -> None:
        self.update_idletasks()
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        window_width = self.winfo_width()
        window_height = self.winfo_height()

        # calculate position to center
        x = (screen_width // 2) - (window_width // 2)
        y = (screen_height // 2) - (window_height // 2)

        self.geometry(f"{window_width}x{window_height}+{x}+{y}")


    def on_close(self):
        ShortcutWindow.instance = None
        self.destroy()
