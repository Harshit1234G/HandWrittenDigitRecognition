import pickle
import numpy as np
import os
import pandas as pd
from tkinter.filedialog import asksaveasfilename


def export_data(data: pd.DataFrame) -> None | str:
    # converting Image.Image object to np.ndarray
    data['original_image'] = data['original_image'].apply(np.array)

    # filedialog
    file_path = asksaveasfilename(
        initialdir= os.getcwd(),
        initialfile= 'data',
        defaultextension= '.pkl',
        filetypes= [('Pickle Files', '*.pkl')]
    )

    if not file_path:
        return None
    
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)
            
    except Exception as e:
        return e
