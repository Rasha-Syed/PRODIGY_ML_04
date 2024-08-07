# Task 03 : Hand Gesture Recognition Model

## Objective
Develop a Convolutional Neural Network (CNN) model to accurately identify and classify different hand gestures from image data, enabling intuitive human-computer interaction and gesture-based control systems.

## Dataset
- `arch.zip`: Contains image data of various hand gestures organized into folders.
  - Inside `leapGestRecog` folder, there are ten subfolders (00 to 09), each containing ten gesture folders (01_palm, 02_l, 03_fist, etc.).
  - Each gesture folder contains around 200 `.png` files of hand gesture images.
- `Dataset Link : https://www.kaggle.com/gti-upm/leapgestrecog

## Steps Implemented

### 1. Data Loading and Exploration
- **Unzipping the Dataset**: Extracted images from `arch.zip` to access the dataset.
- **Exploration**: Reviewed the dataset structure and confirmed the presence of images categorized into different hand gesture folders.

### 2. Data Preprocessing
- **Image Loading**: Loaded images from the extracted directories.
- **Resizing and Normalization**: Resized images to 64x64 pixels and normalized pixel values to [0, 1].
- **Label Encoding**: Encoded gesture labels into numerical values and converted them to categorical format using one-hot encoding.

### 3. Model Building
- **Data Augmentation**: Applied data augmentation techniques including rotation, width and height shift, shear, zoom, and horizontal flip to enhance the training dataset.
- **CNN Model Architecture**:
  - Convolutional layers: Extracted spatial features from the images.
  - Max Pooling layers: Reduced spatial dimensions.
  - Fully Connected (Dense) layers: Learned complex representations.
  - Dropout layer: Prevented overfitting.
- **Model Compilation**: Used Adam optimizer and categorical cross-entropy loss function for training the model.

### 4. Model Training
- **Training and Validation Split**: Split the data into training and test sets using `train_test_split`.
- **Training with Data Augmentation**: Trained the CNN model using the augmented data.

### 5. Model Evaluation
- **Validation**: Evaluated the CNN model on the test set.
- **Metrics**:
  - Calculated accuracy score.
  - Displayed a confusion matrix to visualize model performance.
  - Generated a classification report to detail precision, recall, and F1-score.

### 6. Visualization
- **Confusion Matrix**: Visualized model performance using a confusion matrix.
- **Image Classification**: Classified and displayed specific images with predicted labels.
- **Random Samples**: Displayed random images from the dataset with their true and predicted labels.

## Model Evaluation Metrics
- **Test Accuracy**: Accuracy score on the test data.

## Files Included
- `arch.zip`: Zip file containing gesture images.

## Usage
1. **Unzip the Dataset**: Run the script to extract images from `arch.zip`.
2. **Load and Preprocess Images**: Use `load_images_from_folder()` to load and preprocess images.
3. **Train the Model**: Execute the script to train the CNN model with data augmentation.
4. **Evaluate and Test**: Assess model performance on the test dataset.
5. **Classify and Visualize**: Use `plot_confusion_matrix()` to visualize model performance. Use `plot_images()` to view random samples with true and predicted labels.

## Requirements
- `numpy`
- `scikit-learn`
- `opencv-python`
- `tensorflow`
- `matplotlib`
- `seaborn`
