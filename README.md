# Handwritten Digit Recognition

This is a software tool for recognizing handwritten digits using a trained Convolutional Neural Network model. The project provides an intuitive graphical user interface (GUI) to draw digits, predict their values, and analyze metrics such as accuracy, confidence, and more. Users can also import/export datasets and explore the accuracy trends of the model.

## GUI Preview
![GUI Preview](https://github.com/user-attachments/assets/1d2980ea-9138-49b2-add2-814895823e8a)

## Features
- **Digit Prediction**: Draw digits on the canvas and get predictions with confidence scores.
- **Metrics Display**: View detailed metrics including accuracy, confusion matrix, and probabilities.
- **Import/Export Data**: Load pre-existing datasets or export your predictions and data for further analysis.
- **GUI Controls**: Intuitive buttons, shortcuts, and toggles for seamless user experience.

## Technologies Used
- **Python Version**: 3.12.7 (64-bit)

### Modules and Libraries
| Module/Library       | Version       | Description                                                        |
|----------------------|---------------|--------------------------------------------------------------------|
| `tensorflow`         | `2.18.0`      | Used for building, training, and running the CNN.                  |
| `numpy`              | `1.26.4`      | Provides support for numerical operations and array manipulations. |
| `pandas`             | `2.2.2`       | Facilitates data manipulation and storage using dataframes.        |
| `Pillow`             | `10.2.0`      | Used for image processing, including working with canvas images and some GUI related tasks.|
| `customtkinter`      | `5.2.2`       | Creates an enhanced and modern-looking graphical user interface.   |
| `sklearn`            | `1.4.0`       | Used for evaluation metrics such as confusion matrix and accuracy. |
| `pyinstaller`        | `6.4.0`       | Converts the Python script into a standalone executable.           |
| `matplotlib`         | `3.8.2`       | For plotting all the graphs.                                       |
| `tkinter`            | `8.6`         | For messageboxes and filedialog.                                   |
| `pickle`             |               | For pickling data.                                                 |

## Setup
1. Download the setup file from releases, and install the software.
2. Run the executable `Hand Written Digit Recognition.exe`.
3. Explore and enjoy the features!

> **Note**: The software might take a few seconds to load. A terminal window will also run in the background; closing it will terminate the program.

## Datasets
Sample datasets are provided in the release. You can directly import these into the software for testing.

---

### Future Documentation
I will create a comprehensive documentation soon, explaining:
- The idea behind the project.
- Implementation details, including model training and GUI development.
- The process of creating the setup and executable file.
- Insights and challenges faced during development.

Stay tuned!

---

## License
This project is licensed under the Apache License 2.0. See the [LICENSE](https://github.com/Harshit1234G/HandWrittenDigitRecognition/blob/master/LICENSE.txt) file for details.
