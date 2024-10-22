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
- Dataset for Brain Tumor - https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection
- Datastets for Lung Cancer - https://www.kaggle.com/datasets/adityamahimkar/iqothnccd-lung-cancer-dataset

### Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/sravan7684/CancerPrediction.git
2. **Create a virtual environmen:**
   ```python
     python -m venv venv
     source venv/bin/activate 
     # On Windows use
     venv\Scripts\activate
   ```

3. **Install the required packages:**
   ```bash
    pip install -r requirements.txt

4.  **Training the Models:**
    Prepare the dataset: Ensure your dataset is organized as shown in the project structure.
    
     Run the training script: 
      ```python
         python train_model_lung_cancer.py
         python train_model_brain_tumor.py
       ```
5.  **Saving the models:**
   After running the train_model files for brain and lung cancer save the modesla in the same folder

   - model for brain tumor 

           brain_tumor_model.keras
    
   - model for lung cancer
   
         brain_tumor_model.keras
6.  **Running the Streamlit Apps:**
       
      streamlit run file_name.py
7.  **Project Files:**

   -  __train_model_lung_cancer.py__ : Script to train the CNN model for lung cancer detection.
   
   -  __train_model_brain_tumor.py__ : Script to train the CNN model for brain tumor detection.
   
   - __cancer.py :__ Streamlit app to upload an image and get predictions from the trained lung cancer model.
   
   - __brain_tumor.py :__ Streamlit app to upload an image and get predictions from the trained brain tumor model.
  
  
  **Model Architectures:**


8.  **Lung Cancer Model:**
   The model is a Convolutional Neural Network (CNN) with the following layers:

          Convolutional layers with ReLU activation
          MaxPooling layers
          Flatten layer
          Dense layers with ReLU activation
          Dropout layer
          Output layer with softmax activation for multi-class classification
       

9. **Brain Tumor Model:**
    The model is a Convolutional Neural Network (CNN) with the following layers:

         Convolutional layers with ReLU activation
         MaxPooling layers
         Flatten layer
         Dense layers with ReLU activation
         Dropout layer
         Output layer with sigmoid activation for binary classification

   
11. **License:**
   This project is licensed under the MIT License. See the LICENSE file for details.

12. **Acknowledgements:**

      The datasets used for training the models.
      TensorFlow and Keras for providing the deep learning framework.
      Streamlit for providing the web app framework.

13. **Contact:**
   
      For any questions or suggestions, please contact sravankumar7684@gmail.com.
  
   






