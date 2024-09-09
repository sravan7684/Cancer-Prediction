# Cancer Detection Project

This project aims to detect lung cancer and brain tumors from medical images using Convolutional Neural Network (CNN) models built with TensorFlow and Keras. The models classify images into different categories for lung cancer and brain tumors.

## Project Structure


## Setup Instructions

### Prerequisites

- Python 3.7 or higher
- TensorFlow
- Streamlit
- PIL (Pillow)
- NumPy

### Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/sravan7684/CancerPrediction.git
2. **Create a virtual environmen:**
      ```sh
     python -m venv venv
    source venv/bin/activate  # On Windows use
       `venv\Scripts\activate`

3. **Install the required packages:**
   ```sh
         pip install -r requirements.txt


4.  **Training the Models:**
    Prepare the dataset: Ensure your dataset is organized as shown in the project structure.

    Run the training script: 
      ```sh 
         python train_model_lung_cancer.py
         python train_model_brain_tumor.py
5.  **Saving the models:**
   After running the train_model files for brain and lung cancer save the modesla in the same folder
   -> model for brain tumor 
      ```sh
       brain_tumor_model.keras
   -> model for lung cancer
      ```sh
      
       brain_tumor_model.keras
            

6.  **Running the Streamlit Apps:**
       ```sh
        streamlit run file_name.py
7.  Project Files

train_model_lung_cancer.py: Script to train the CNN model for lung cancer detection.
train_model_brain_tumor.py: Script to train the CNN model for brain tumor detection.
cancer.py: Streamlit app to upload an image and get predictions from the trained lung cancer model.
brain_tumor.py: Streamlit app to upload an image and get predictions from the trained brain tumor model.

    

    

   

   

