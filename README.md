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

## Documentation
I am thrilled to share that I have completed the detailed documentation for this project as promised. It is now available under the releases section of this repository on GitHub. The documentation serves as a comprehensive guide, covering the following concepts:
1. **Acknowledgments**  
   A brief mention of those who supported the project morally and intellectually.

2. **System Requirements**  
   Detailed prerequisites for running the project, including hardware and software specifications.

3. **Table of Contents**  
   Organized structure of the documentation for ease of navigation.

4. **Project Overview**  
   An introduction to the project's objectives and its key features.

5. **Convolutional Neural Networks**  
   - Understanding human vision and the need for CNNs.  
   - Concepts of Convolution, Fourier Transform, and their applications in image processing.  
   - Explanation of Convolutional Layers, Filters, Feature Maps, Pooling Layers, and the LeNet-5 Architecture.  
   - Training methodology and memory considerations for CNNs.

6. **Code Explanation**  
   Step-by-step breakdown of the CNN implementation and GUI functionalities.

7. **Deployment**  
   Instructions for creating an executable file and a setup installer for the project.

8. **Future Plan**  
   Ideas and improvements for future iterations.

9. **Conclusion**  
   Final thoughts and reflections on the project.
   
---

## License
This project is licensed under the Apache License 2.0. See the [LICENSE](https://github.com/Harshit1234G/HandWrittenDigitRecognition/blob/master/LICENSE.txt) file for details.
