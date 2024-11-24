import customtkinter as ctk
from PIL import ImageTk
from typing import Self
import utils.common as common


class ShortcutWindow(ctk.CTkToplevel):
    instance: Self | None = None  # class attribute to hold the single instance

    def __init__(self, *args, **kwargs) -> None:
        # prevent creating a new instance if one already exists
        if self.already_exists():
            self.focus_existing()
            return None

        super().__init__(*args, **kwargs)

        # save this instance
        self.set_instance(self)

        # when the window is closed, reset the instance
        self.protocol('WM_DELETE_WINDOW', self.on_close)

        # basic configuration
        self.geometry('500x500')
        self.title('Shortcuts Panel')
        self.resizable(False, False)

        common.center_window(toplevel_window= self)

        # setting icon
        icon = ImageTk.PhotoImage(file= 'icons/app.png')
        self.wm_iconbitmap()
        self.after(200, lambda: self.iconphoto(False, icon))

        # frame for all shortcuts
        self.main_frame = ctk.CTkScrollableFrame(
            master= self,
            corner_radius= 15
        )
        self.main_frame.pack(
            padx= 7,
            pady= 7,
            fill= 'both', 
            expand= True
        )

        # adding shortcuts
        shortcuts: dict[str, str | None] = {
            'Basic Actions': None,
            'Predict': 'ctrl + p',
            'Clear (clearing canvas)': 'ctrl + delete',
            'Import': 'ctrl + o',
            'Export': 'ctrl + s',
            'History & Shortcut Panel': None,
            'Clear All Data': 'ctrl + shift + T',
            'Load (from history)': 'ctrl + shift + L',
            'Shortcuts Panel': 'ctrl + .',
            'Correction Actions': None,
            'Correct button': 'ctrl + shift + C',
            'Wrong button': 'ctrl + shift + W',
            'Submit button (appears after clicking "Wrong")': 'ctrl + shift + S',
            'Metrics Toggles': None,
            'Accuracy Trend per Prediction': 'ctrl + m + 1',
            'Confidence Trend per Prediction': 'ctrl + m + 2',
            'Confusion Matrix': 'ctrl + m + 3',
            'Correct V/S Wrong': 'ctrl + m + 4'
        }

        for name, key in shortcuts.items():
            if key is None:
                self.add_heading(name)

            else:
                self.add_shortcut(name, key)

        # setting focus
        self.after(200, self.lift)
        self.after(200, self.focus_set)


    @classmethod
    def already_exists(cls) -> bool:
        return cls.instance is not None


    @classmethod
    def set_instance(cls, instance: Self | None) -> None:
        cls.instance = instance


    @classmethod
    def focus_existing(cls) -> None:
        if cls.instance.state() == 'iconic':  # check if minimized
            cls.instance.deiconify()

        cls.instance.focus_set()
        cls.instance.lift()


    def on_close(self):
        self.set_instance(self)
        self.destroy()

    
    def add_heading(self, heading: str) -> None:
        ctk.CTkLabel(
            master= self.main_frame,
            text= heading,
            text_color= common.grey_color,
            font= (common.font, 16, 'bold')
        ).pack(padx= 7, pady= (7, 0), side= 'top', anchor= 'w')

    
    def add_shortcut(self, name: str, key: str) -> None:
        shortcut_frame = ctk.CTkFrame(
            master= self.main_frame,
            corner_radius= 5,
            fg_color= common.upper_frame_color
        )
        shortcut_frame.pack(
            padx= 7, 
            pady= (7, 0), 
            fill= 'x',
            side= 'top'
        )

        # setting name of shortcut
        ctk.CTkLabel(
            master= shortcut_frame,
            text= name,
            text_color= common.grey_color,
            font= (common.font, 12)
        ).pack(side= 'left', padx= 7)

        # setting shortcut key
        ctk.CTkLabel(
            master= shortcut_frame,
            text= key,
            text_color= common.grey_color,
            font= (common.font, 12, 'bold')
        ).pack(side= 'right', padx= 7)
