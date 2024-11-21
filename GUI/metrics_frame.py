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
        self.probabilities: common.NDArrayFloat | None = None
        self.prediction: int | None = None
        self.history = pd.DataFrame(
            columns= [
                'original_image', 
                'Prediction', 
                'probabilities', 
                'Correct Number',
                'Confidence (%)',
                'correctness',
                'acc_score'
            ]
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
            border_color= common.grey_color,
            # input validation
            validate= 'key',
            validatecommand= (self.register(self.validate_input), '%P')
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
            text_color= common.grey_color,
            justify= 'left'
        ).grid(row= 0, column= 3, padx= (15, 0), sticky= 'w')

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
            column= 4,
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
        ).grid(row= 1, column= 0, padx= 10, columnspan= 5)


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
            text_color= common.grey_color,
            text= 'History',
            font= (common.font, 16),
            justify= 'left'
        ).pack(padx= 7, pady= (7, 0), side= 'top', anchor= 'w')

        # columns to display
        self.display_columns: list[str] = [
            'S.No.',
            'Prediction',
            'Correct Number',
            'Confidence (%)'
        ]

        # setting the style for tables
        style = ttk.Style(self.history_frame)
        style.theme_use('xpnative')

        style.configure(
            style= 'Custom.Treeview.Heading',
            font= (common.font, 12, 'bold'),
            foreground= common.grey_color
        )

        # making table
        self.tree_view = ttk.Treeview(
            master= self.history_frame,
            columns= self.display_columns,
            show= 'headings',
            style= 'Custom.Treeview'
        )

        for column, min_width in zip(self.display_columns, [50, 90, 130, 130]):
            self.tree_view.heading(
                column, 
                text= column
            )
            self.tree_view.column(
                column, 
                anchor= 'center', 
                width= min_width, 
                minwidth= min_width
            )

        # setting tags
        self.tree_view.tag_configure('evenrow', background= common.table_alternate_colors[0])
        self.tree_view.tag_configure('oddrow', background= common.table_alternate_colors[1])
        self.tree_view.tag_configure('wrong', background= common.red_color)


        # Add a vertical scrollbar to the frame
        scrollbar = ttk.Scrollbar(
            master= self.tree_view, 
            orient= 'vertical', 
            command= self.tree_view.yview
        )
        self.tree_view.configure(yscroll= scrollbar.set)
        scrollbar.pack(side= 'right', fill= 'y')

        # Pack the Treeview widget inside the CTkScrollableFrame
        self.tree_view.pack(
            side= 'top', 
            padx= 7, 
            pady= 7, 
            fill= 'both',
            expand= True
        )

        # setting initial label becuase no data is present
        self.default_history_label = ctk.CTkLabel(
            master= self.history_frame,
            height= 250,
            text= "No predictions yet. After drawing a number and clicking 'Predict,' click 'Correct' or 'Wrong' to add it to the history.",
            image= self.info_img,
            text_color= common.grey_color,
            font= (common.font, 12),
            wraplength= 200,
            compound= 'top'
        )
        self.default_history_label.pack(
            anchor= 'center', 
            padx= 20, 
            pady= 20
        )

        common.button_kwargs['width'] = 270

        # delete row button
        self.delete_row_button = ctk.CTkButton(
            master= self.history_frame,
            **common.button_kwargs,
            text= 'Delete Row',
            state= 'disabled'
        )
        self.delete_row_button.pack(
            side= 'left',
            anchor= 'sw',
            padx= 7,
            pady= (0, 7)
        )
        # Clear all button
        self.clear_all_button = ctk.CTkButton(
            master= self.history_frame,
            **common.button_kwargs,
            text= 'Clear All'
        )
        self.clear_all_button.pack(
            side= 'right',
            anchor= 'se',
            padx= (0, 7),
            pady= (0, 7)
        )


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
                self.update_history()

            case 'Wrong':
                self.pred_label.configure(text= 'Correct Number:')
                self.result_pred.configure(
                    state= 'normal', 
                    border_color= common.fg_color
                )
                self.result_pred.focus_set()
                self.pred_var.set('')
                self.result_frame.update()

                self.result_pred.configure(width= 60)

                self.submit_correct_num_button = ctk.CTkButton(
                    master= self.result_frame,
                    width= 50,
                    height= 6,
                    corner_radius= 2,
                    fg_color= common.grey_color,
                    hover_color= common.fg_color,
                    text= 'Submit',
                    font= (common.font, 12),
                    command= self.update_history
                )
                self.submit_correct_num_button.grid(
                    row= 0, 
                    column= 2,
                    pady= 7,
                    padx= (0, 10),
                    sticky= 'w'
                )

        self.correct_wrong_button.configure(state= 'disabled')
        self.correct_wrong_button.set('')
        self.result_frame.update()

    
    def append_to_history(self) -> None:
        correct_number = int(self.pred_var.get())
        data: dict[str, any] = {
            'original_image': self.original_image,
            'Prediction': self.prediction,
            'probabilities': self.probabilities,
            'Correct Number': correct_number,
            'Confidence (%)': int(round((self.probabilities.max()), 2) * 100),
            'correctness': self.prediction == correct_number
        }
        index: int = len(self.history.index)

        self.history.loc[index] = data

        y_true, y_pred = self.history['Correct Number'], self.history['Prediction']

        acc_score: float = accuracy_score(y_true, y_pred)
        self.history.loc[index, 'acc_score'] = acc_score


    def insert_row_to_treeview(self) -> None:
        # deleting default message
        if self.default_history_label.winfo_exists():
            self.default_history_label.destroy()

        # taking the row
        sr_no: int = len(self.history) 
        row = self.history.loc[sr_no - 1, self.display_columns[1:]]

        # selecting tag
        if not self.history.loc[sr_no - 1, 'correctness']:
            tags = ('wrong',)

        elif sr_no % 2 == 0:
            tags: tuple[str] = ('evenrow')

        else:
            tags = ('oddrow',)

        # inserting to table
        self.tree_view.insert(
            parent= '',
            index= 'end',
            text= '',
            values= (sr_no, row['Prediction'], row['Correct Number'], row['Confidence (%)']),
            tags= tags
        )
        self.history_frame.update()

    def update_history(self) -> None:
        self.append_to_history()
        self.insert_row_to_treeview()

        if hasattr(self, 'submit_correct_num_button'):
            self.submit_correct_num_button.destroy()

        self.result_pred.configure(width= 140)
        self.result_pred.configure(
            state= 'disabled', 
            border_color= common.grey_color
        )


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
        self.pred_label.configure(text= 'Prediction:')
        self.result_pred.configure(
            state= 'disabled', 
            border_color= common.grey_color
        )
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
