import customtkinter as ctk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image
from tkinter import ttk
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
import utils.common as common


class MetricsFrame(ctk.CTkFrame):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # data variables
        self.original_image: common.NDArrayFloat | None = None
        self.resized_image: common.NDArrayFloat | None = None
        self.probabilities: common.NDArrayFloat | None = None
        self.prediction: int | None = None
        self.history = pd.DataFrame(
            columns= [
                'original_image', 
                'resized_image', 
                'prediction', 
                'probabilities', 
                'y_true',
                'confidence',
                'correctness',
                'acc_score'
            ],
            index= pd.RangeIndex(start= 1, stop= 1, step= 1)
        )
        # text variables
        self.pred_var = ctk.StringVar(value= '')

        # info image
        self.info_img = ctk.CTkImage(
            light_image= Image.open('icons/info.png'),
            size= (50, 50)
        )

        # frame where all the metrics are displayed
        self.all_metrics_frame = ctk.CTkFrame(
            self,
            fg_color= common.upper_frame_color
        )
        self.all_metrics_frame.pack(
            side= 'bottom',
            anchor= 'e',
            fill= 'both',
            expand= True,
            padx= 20,
            pady= (10, 20)
        )

        # the frame where are the checkboxes for metrics
        self.checkbox_frame = ctk.CTkFrame(
            self, 
            height= 45,
            fg_color= common.upper_frame_color
        )
        self.checkbox_frame.pack(
            side= 'bottom',
            anchor= 'e',
            fill= 'x',
            padx= 20
        )

        # history frame
        self.history_frame = ctk.CTkFrame(
            self, 
            height= 310,
            fg_color= common.upper_frame_color
        )

        self.history_frame.pack(
            side= 'right',
            anchor= 'n',
            fill= 'x',
            expand= True,
            padx= (0, 20),
            pady= (20, 10)
        )
        self.history_frame.pack_propagate(False)

        # bar graph of probabilities frame
        self.bar_frame = ctk.CTkFrame(
            self,
            width= 420,
            height= 220,
            fg_color= common.upper_frame_color
        )

        self.bar_frame.pack(
            side= 'top',
            anchor= 'w',
            padx= (20, 10),
            pady= (20, 10)
        )
        self.bar_frame.pack_propagate(False)

        # frame to display prediction, and correctness check
        self.result_frame = ctk.CTkFrame(
            self,
            width= 420,
            height= 80,
            fg_color= common.upper_frame_color
        )
        self.result_frame.pack(
            side= 'top',
            anchor= 'w',
            padx= (20, 10),
            pady= (0, 10)
        )
        self.result_frame.grid_propagate(False)

        # setting default layouts
        self.default_result_frame_layout()
        self.default_bar_frame_layout()
        self.default_history_frame_layout()
        

    def default_result_frame_layout(self) -> None:
        # label
        self.pred_label = ctk.CTkLabel(
            self.result_frame,
            text= 'Prediction:',
            font= (common.font, 12),
            text_color= common.grey_color
        )
        self.pred_label.grid(row= 0, column= 0, padx= (10, 0), pady= 7, sticky= 'w')
        
        # result frame
        self.result_pred = ctk.CTkEntry(
            self.result_frame,
            textvariable= self.pred_var,
            state= 'disabled',
            corner_radius= 5,
            height= 10,
            border_color= common.grey_color
        )
        self.result_pred.grid(
            row= 0, 
            column= 1,
            pady= 7,
            padx= (0, 10),
            sticky= 'w'
        )

        # adding a separator
        ctk.CTkLabel(
            self.result_frame,
            text= '|',
            font= (common.font, 24),
            text_color= common.grey_color
        ).grid(row= 0, column= 2, padx= (15, 0), sticky= 'w')

        # segmented button
        self.correct_wrong_button = ctk.CTkSegmentedButton(
            self.result_frame,
            height= 10,
            fg_color= common.upper_frame_color,
            selected_color= common.fg_color,
            selected_hover_color= common.hover_color,
            unselected_color= common.grey_color,
            unselected_hover_color= common.fg_color,
            values= ['Correct', 'Wrong'],
            command= self.correct_wrong_callback,
            state= 'disabled'
        )
        self.correct_wrong_button.grid(
            row= 0,
            column= 3,
            pady= 7,
            padx= 10,
            sticky= 'e'
        )

        # description label
        ctk.CTkLabel(
            self.result_frame,
            text= "● If the model's prediction is correct, click 'Correct'; otherwise, click 'Wrong'. ● Enter correct number after clicking 'Wrong'.",
            font= (common.font, 12),
            text_color= common.grey_color,
            wraplength= 418,
            justify= 'left'
        ).grid(row= 1, column= 0, padx= 10, columnspan= 4)


    def default_bar_frame_layout(self) -> None:
        # removing previous plot
        common.clear_widgets(self.bar_frame)

        ctk.CTkLabel(
            master= self.bar_frame,
            width= 400,
            height= 200,
            text= "Draw a digit and click 'Predict' to see its probability distribution.",
            image= self.info_img,
            text_color= common.grey_color,
            font= (common.font, 12),
            wraplength= 150,
            compound= 'top'
        ).pack(anchor= 'center', padx= 20, pady= 20)


    def default_history_frame_layout(self) -> None:
        ctk.CTkLabel(
            master= self.history_frame,
            height= 300,
            text= "No predictions yet. After drawing a number and clicking 'Predict,' click 'Correct' or 'Wrong' to add it to the history.",
            image= self.info_img,
            text_color= common.grey_color,
            font= (common.font, 12),
            wraplength= 250,
            compound= 'top'
        ).pack(anchor= 'center', padx= 20, pady= 20)


    def bar_plot_from_proba(self) -> None:
        # removing previous plot
        common.clear_widgets(self.bar_frame)

        # Processing probabilities
        probas: common.NDArrayInt = (self.probabilities.round(2) * 100).astype(np.uint8)

        # plotting probabilities
        fig, ax = plt.subplots(figsize= (4.05, 2.05))
        ax.bar(
            x= range(10),
            height= probas,
            color= common.fg_color,
            edgecolor= common.grey_color
        )
        ax.set_title('Probability Distribution')
        ax.set_xlabel('Number')
        ax.set_ylabel('Probability (%)')
        ax.set_xticks(range(10))
        ax.set_yticks(range(0, 101, 25))

        fig.tight_layout()

        # Setting to frame
        canvas = FigureCanvasTkAgg(fig, master= self.bar_frame)  
        canvas.draw()
        canvas.get_tk_widget().pack(padx= 7, pady= 7)


    def correct_wrong_callback(self, value: str) -> None:
        match value:
            case 'Correct':
                self.append_to_history()

            case 'Wrong':
                # self.append_to_history_for_wrong()
                self.pred_label.configure(text= 'Correct Number:')
                self.result_pred.configure(
                    state= 'normal', 
                    border_color= common.fg_color
                )
                self.result_pred.focus_set()
                self.pred_var.set('')
                self.update()

        self.correct_wrong_button.configure(state= 'disabled')
        self.correct_wrong_button.set('')
        self.result_frame.update()
        self.update_history_display()

    
    def append_to_history(self) -> None:
        data: dict[str, any] = {
            'original_image': self.original_image,
            'resized_image': self.resized_image,
            'prediction': self.prediction,
            'probabilities': self.probabilities,
            'y_true': int(self.pred_var.get()),
            'confidence': self.probabilities.max(),
            'correctness': True
        }
        index: int = len(self.history.index)

        self.history.loc[index] = data

        y_true, y_pred = self.history['y_true'], self.history['prediction']

        acc_score: float = accuracy_score(y_true, y_pred)
        self.history.loc[-1, 'acc_score'] = acc_score



    def append_to_history_for_wrong(self) -> None:
        ...


    def update_history_display(self) -> None:
        # # columns to display
        # display_columns: list[str] = [
        #     'S.No.'
        #     'resized_image', 
        #     'prediction',
        #     'y_true',
        #     'confidence'
        # ]
        # # creating tree view
        # tree_view = ttk.Treeview(
        #     master= self.history_frame,
        #     columns= display_columns,
        #     show= 'headings'
        # )

        # # adding headings
        # for column in display_columns:
        #     tree_view.heading(column, text= column)
        #     tree_view.column(column, anchor= 'center', width= 100)

        # # adding data
        ...

    def register_validate(self):
        validate_cmd = (self.register(self.validate_input), '%P')
        self.result_pred.configure(validate= 'key', validatecommand= validate_cmd)

    def validate_input(self, input_value: str) -> bool:
        if input_value.isdigit() and 0 <= int(input_value) <= 9:
            return True
        
        elif input_value == '':  # Allow clearing the input during typing
            return True
        
        else:
            return False

    
    def update_all(self) -> None:
        # if the user click prediction again without clearing, then the correct and wrong button is stuck to previous state so setting it '' initially.
        self.correct_wrong_button.set('')

        self.pred_var.set(self.prediction)
        self.bar_plot_from_proba()
        self.correct_wrong_button.configure(state= 'normal')
        self.update()


    def clear_prediction(self) -> None:
        # setting default layout
        self.default_bar_frame_layout()
        # resetting vars and getting back to normal layout
        self.pred_var.set('')
        self.correct_wrong_button.configure(state= 'disabled')
        self.correct_wrong_button.set('')
        self.pred_label.configure(text= 'Prediction:')
        self.result_pred.configure(
            state= 'disabled', 
            border_color= common.grey_color
        )
        self.update()
