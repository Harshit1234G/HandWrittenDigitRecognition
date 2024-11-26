import pickle
import os
import pandas as pd
from PIL import Image
from tkinter.filedialog import askopenfilename


def import_data() -> pd.DataFrame | str | None:
    file_path = askopenfilename(
        initialdir= os.getcwd(),
        filetypes= [('Pickle Files', '*.pkl')]
    )
    
    if not file_path:
        return None
    
    try:
        with open(file_path, 'rb') as file:
            data: pd.DataFrame = pickle.load(file)

            # converting np.ndarray to Image.Image object
            data['original_image'] = data['original_image'].apply(Image.fromarray)

        return data
    
    except Exception as e:
        return e
