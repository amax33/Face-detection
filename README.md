# Face Detection Model

This project trains CNN models for gender and age classification based on face image data. The workflow includes data loading, preprocessing, image augmentation, model construction, and loss function customization.

### 1. **Imports and Setup**
   - Essential libraries are imported, including:
     - **TensorFlow** for model building, data processing, and training.
     - **Pandas** and **NumPy** for data manipulation.

### 2. **Data Inspection**
   - Displays the dataset structure and checks data types.
   - The dataset has 49 columns, including the image filename and attributes like gender, age categories, ethnicity, and other facial features.

### 3. **Data Preprocessing**
   - **Label Mapping**: Converts `-1` and `1` labels to `0` and `1` for binary columns (e.g., gender).
   - **Age Label Creation**:
     - Maps age categories (`Young`, `Middle_Aged`, `Senior`) to labels: 0 for Young, 1 for Middle-Aged, and 2 for Senior.
     - Removes rows with no age category for clear labeling.
   - **Data Splitting**: The data is split into training and validation sets for gender and age using `StratifiedShuffleSplit` to ensure balanced class representation.
   - **Oversampling**: `RandomOverSampler` is used to balance the training set by oversampling minority classes.

### 4. **Image Loading and Processing**
   - **Image Loading**: Defines a function, `load_and_preprocess_image`, to read, resize, and normalize each image.
   - **Image Augmentation**:
     - Applies random transformations (flip, brightness, contrast, saturation) to images to prevent overfitting and increase model generalization.
   - **Dataset Preparation**:
     - Creates TensorFlow datasets for gender and age by batching, shuffling, and applying augmentation.
     - Two functions, `prepare_dataset_gender` and `prepare_dataset_age`, prepare separate datasets for each classification task.

### 5. **Model Architecture**
   - **Gender Classification Model**:
     - A Convolutional Neural Network (CNN) model with layers:
       - `Conv2D` and `MaxPooling2D` for feature extraction.
       - `BatchNormalization` for normalization.
       - `Flatten` and `Dense` for fully connected layers.
       - `Dropout` to reduce overfitting.
     - Final layer with a sigmoid activation for binary classification.
   - **Age Classification Model**:
     - A CNN with a similar structure but with a softmax output layer for multiclass classification (3 age groups).

### 6. **Loss Functions**
   - **Binary Focal Loss**:
     - Used for gender classification to address class imbalance.
     - Includes parameters:
       - **`gamma`**: focusing parameter to reduce loss for well-classified examples.
       - **`alpha`**: balancing parameter to adjust for class imbalance.
   - **Sparse Categorical Focal Loss**:
     - Used for age classification to mitigate class imbalance across multiple classes.
     - Similar structure as binary focal loss but adapted for multiclass classification with one-hot encoding.
