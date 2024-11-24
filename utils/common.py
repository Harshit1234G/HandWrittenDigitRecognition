import customtkinter as ctk
from numpy import int_, float_
from numpy.typing import NDArray


fg_color: str = '#FE6402'
bg_color: str = '#DBDBDB'
hover_color: str = '#aa3600'
grey_color: str = '#6d707f'
upper_frame_color: str = '#EBEBEB'
font: str = 'sans serif'
table_alternate_colors: tuple[str, str] = ('#ffe5d6', '#fffaf7')
red_color: str = '#FF4C4C'

NDArrayInt = NDArray[int_]
NDArrayFloat = NDArray[float_]

button_kwargs: dict[str, any] = {
    'width': 280,
    'height': 35,
    'corner_radius': 5,
    'font': (font, 18),
    'fg_color': fg_color,
    'hover_color': hover_color
}

button_grid_kwargs: dict[str, any] = {
    'column': 0, 
    'sticky': 'we', 
    'padx': 20, 
    'pady': (0, 10)
}

plot_pack_kwargs: dict[str, any] = {
    'padx': 7, 
    'pady': 7, 
    'side': 'left', 
    'fill': 'both',
    'expand': True
}


def create_canvas_and_line(
    master: ctk.CTkFrame, 
    row: int, 
    *,
    column: int = 0, 
    coordinates: list[int] = [10, 3, 270, 3]
    ) -> None:
    canvas_widget = ctk.CTkCanvas(
        master= master,
        width= coordinates[0] + coordinates[2],
        height= 6,
        bg= bg_color,
        highlightthickness= 0
    )
    canvas_widget.grid(row= row, column= column, pady= (0, 10))

    canvas_widget.create_line(*coordinates, fill= grey_color)


def clear_widgets(master: ctk.CTkFrame) -> None:
    for widget in master.winfo_children():
        widget.destroy()

    
def center_window(toplevel_window: ctk.CTkToplevel) -> None:
        toplevel_window.update_idletasks()
        screen_width = toplevel_window.winfo_screenwidth()
        screen_height = toplevel_window.winfo_screenheight()
        window_width = toplevel_window.winfo_width()
        window_height = toplevel_window.winfo_height()

        # calculate position to center
        x = (screen_width // 2) - (window_width // 2)
        y = (screen_height // 2) - (window_height // 2)

        toplevel_window.geometry(f"{window_width}x{window_height}+{x}+{y}")
