<img width="200" height="200" alt="463395010-2204bb58-c11a-42b9-90b0-3f4870b8faf7" src="https://github.com/user-attachments/assets/932caf31-0b62-4775-bef6-4d8d7f0530bf" />


# Plant Disease Classification using Transfer Learning

This repository applies supervised deep learning models ResNet50, EfficientNetB0, and MobileNetV2 to classify plant leaf diseases using the [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease).

---

# Overview
Task Definition
Classify RGB images of plant leaves into 8 disease categories based on visual symptoms using image classification models.

Early detection of plant diseases is critical for sustainable agriculture and food security.  
This project tackles the challenge of classifying plant leaf images into disease categories using modern deep learning techniques.  
Our goal is to build a reliable image classifier that can assist farmers and researchers in identifying plant diseases based on leaf images

# Approach
We formulated this as a multiclass supervised classification problem, leveraging transfer learning with pre trained CNNs. We compared three architectures — ResNet50, EfficientNetB0, and MobileNetV2  and applied data augmentation to improve generalization.

# Performance Summary
**ResNet50** achieved 95% validation accuracy with strong ROC-AUC scores.

**EfficientNetB0** reached 90% validation accuracy.

**MobileNetV2 performed** at 78% validation accuracy, suitable for mobile deployment.



---

## Summary of Work Done

### Data

* **Type:** RGB images (224x224)
* **Classes:** 8 disease categories
* **Dataset Size:** 992 images total 794 for training, 198 for validation
* **Preprocessing:** Resizing, normalization, augmentation (rotation, zoom, flip)

### Data Visualization

* Sample images displayed per class to ensure correctness.

* <img width="807" height="812" alt="download" src="https://github.com/user-attachments/assets/ffecf96c-3101-4c0b-8718-5e0f9de93ee0" />

---

## Problem Formulation

### Input / Output

* **Input:** RGB image (224x224x3)
* **Output:** Predicted disease category (one of 8)

### Models Tried

* **EfficientNetB0**  Efficient model with balanced performance.
* **ResNet50** Best performer in accuracy and class separation.
* **MobileNetV2**  Lightweight model suitable for mobile.

### Hyperparameters

* **Optimizer:** Adam
* **Loss:** Sparse Categorical Crossentropy
* **Epochs:** 30
* **Batch Size:** Auto from tf.data pipelines

---

## Training

* **Platform:** Google Colab with GPU
* **Framework:** TensorFlow 2.x, Keras
* **Training Time:** \~30 mins per model
* **Epoch Control:** Fixed 30 epochs
* **Augmentation:** Enhanced with ImageDataGenerator

### Training Results

* ROC curves and confusion matrices analyzed post-training


<img width="776" height="601" alt="download" src="https://github.com/user-attachments/assets/c0788a7e-1113-4721-9c92-6587d8b55b44" />
<img width="691" height="547" alt="download" src="https://github.com/user-attachments/assets/687a779b-f840-47ed-b5a7-f5c3e000cb52" />
<img width="700" height="547" alt="download" src="https://github.com/user-attachments/assets/6ddf77c4-5d32-4b28-b593-7781e35fb7f5" />
<img width="702" height="601" alt="download" src="https://github.com/user-attachments/assets/f67fd6ee-6359-43cb-a749-3456e2de66a9" />

# Comparing Models With and Without Augmentation

## Models Compared:
- Baseline Model (Without Augmentation)
- Model with Data Augmentation
<img width="1288" height="701" alt="download" src="https://github.com/user-attachments/assets/c5ba5320-62a8-44c2-be27-0e8dfc9b2c26" />

##  ROC Curve Comparison — Observations

-  **Perfect Classification for Disease Classes (AUC = 1.00)**  
  Both the **Base Model** and **Augmented Model** achieved near-perfect ROC curves for:
  - `Tomato_Bacterial_spot`
  - `Tomato_Early_blight`
  - `Tomato_Late_blight`
  - `Tomato_Tomato_mosaic_virus`
  
  This reflects **excellent model performance** in detecting these diseases.

-  **Slightly Lower Performance on Healthy Class**
  - **Base Model AUC:** 0.95
  - **Augmented Model AUC:** 0.94
  - The classification of **Healthy leaves** is slightly less accurate, indicating potential feature overlap with diseased samples.

-  **Effect of Augmentation**
  - **Data Augmentation** did **not degrade ROC performance**.
  - It maintained or marginally enhanced model robustness across most classes.

-  **ROC Curve Behavior**
  - Most curves tightly hug the **top-left corner**, indicating **high sensitivity and specificity**.
  - The dotted diagonal line serves as a reference for random guessing.

---

###  **Conclusion**
- The model exhibits **strong discriminative ability** for disease detection.
- **Healthy class detection** shows minor performance lag — could benefit from further analysis or targeted augmentation.
- **Data Augmentation** strategy proved effective and reliable across all tested classes.





## Performance Comparison

<img width="702" height="601" alt="download" src="https://github.com/user-attachments/assets/8c4c3af4-d35e-4b55-b926-57151d10bb46" />
<img width="702" height="601" alt="download" src="https://github.com/user-attachments/assets/600861bb-38e0-4189-9f7d-016ac37780c5" />




| Model          | Validation Accuracy | Macro F1-Score |
| -------------- | ------------------- | -------------- |
| ResNet50       | 95%                 | 0.94           |
| EfficientNetB0 | 90%                 | 0.89           |
| MobileNetV2    | 78%                 | 0.76           |

### Visual Analysis

<img width="1173" height="790" alt="download" src="https://github.com/user-attachments/assets/add5c4e3-8135-42a7-bde7-bab91f7fb13d" />

##  ROC Curve Comparison Observations – EfficientNetB0 vs ResNet50 vs MobileNetV2

###  EfficientNetB0:
- Achieved **AUC = 1.00** for most classes (perfect classification on key diseases).
- **Tomato_healthy** slightly lower at **AUC = 0.95**, still excellent.
- Demonstrated the most consistently high ROC performance.

###  ResNet50:
- Matched EfficientNetB0 in most categories with **AUC = 1.00**.
- **Tomato_healthy** slightly better at **AUC = 0.96**.
- Overall showed marginally stronger generalization on critical classes.

###  MobileNetV2:
- Performed well but slightly behind the other two models.
- **AUC values ranged between 0.91 and 0.99**.
- Lower AUCs observed particularly for **Potato___Late_blight (0.94)** and **Tomato_healthy (0.91)**.
- Indicates some class-wise weaknesses compared to ResNet50 and EfficientNetB0.

---

###  Overall Insights:
- **ResNet50 and EfficientNetB0 consistently delivered top-tier ROC scores across most classes**, making them better suited for high accuracy scenarios.
- **MobileNetV2, while lighter, showed slightly lower AUCs**, reflecting its trade-off between model complexity and classification strength.
- All three models demonstrated strong performance, but **ResNet50 edges slightly ahead in multiclass discrimination**.
- ROC curves confirm the ranking observed from accuracy and confusion matrix evaluations.

---

---


## Conclusion:
- Transfer learning with pre-trained models like **ResNet50**, **EfficientNetB0**, and **MobileNetV2** proved effective for plant disease classification on the PlantVillage subset.
- **ResNet50 consistently achieved the highest validation accuracy (95%) and superior AUC scores**, making it the most reliable model among those tested.
- **EfficientNetB0** performed nearly as well, showing strong generalization with slightly lower complexity.
- **MobileNetV2**, while lightweight, underperformed slightly in both accuracy and ROC-AUC, but remains a viable option for resource-constrained environments.
- Evaluation metrics and visual analysis confirmed that the models sometimes confuse diseases with similar visual symptoms (e.g., blight categories).
- Data augmentation improved overall performance and helped mitigate overfitting, especially on EfficientNetB0.




##  Future Work:
- **Fine-tune pre-trained layers** (unfreeze and retrain) to improve feature extraction on disease-specific patterns.
- **Experiment with advanced augmentation techniques** (e.g., color jitter, cutout, mixup) to improve robustness.
- **Expand dataset size** or include more diverse samples to address class confusion, especially between blight categories.
- **Try ensemble models** or voting systems combining the strengths of ResNet50 and EfficientNetB0.
- **Deploy lightweight models like MobileNetV2** for mobile or edge devices with real-time inference.
- Explore explainable AI (XAI) methods (e.g., Grad-CAM) to visualize what the model focuses on during prediction.

---

## How to Reproduce Results

### Clone Repository

```bash
git clone https://github.com/yourusername/plant-disease-classification.git
cd plant-disease-classification
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Download Dataset

* Get it from [PlantVillage Kaggle Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
* Place under `./plantdisease_subset/`

### Train Models

* `Train_EfficientNetB0.ipynb`
* `Train_ResNet50.ipynb`
* `Train_MobileNetV2.ipynb`

### Evaluate Models

* Use `Model_Evaluation.ipynb` for confusion matrices, ROC, and classification reports.

---

## Overview of Repository Files

- **Data_louder (1).ipynb**  Loads and prepares the dataset for training and validation.

- **TrainBaseMode.ipynb** Trains the baseline EfficientNetB0 model without data augmentation.

- **TrainBaseModelAugmentation.ipynb** Trains the EfficientNetB0 model with data augmentation applied.

- **Train_ResNet50.ipynb** Implements and trains the ResNet50 model using transfer learning.

- **Train_MobileNet.ipynb** Implements and trains the MobileNetV2 model using transfer learning.

- **CompareAugmentation.ipynb**  Compares model performance with and without data augmentation using ROC curves and confusion matrices.

- **CompareModels (1).ipynb**  Compares ResNet50, EfficientNetB0, and MobileNetV2 models side by side based on metrics and ROC analysis.

- **Qualitative_Analysis_-_Sample_Predictions.ipynb** Tests the trained models on individual samples and visualizes predictions.

- **README.md** Project documentation and instructions.

---

## Software Setup

* Python 3.10+
* TensorFlow 2.x
* scikit-learn
* Matplotlib, NumPy

---

## Data Source

* [PlantVillage Dataset (Kaggle)](https://www.kaggle.com/datasets/emmarex/plantdisease)

---

## How to Train & Evaluate

* Run training notebooks sequentially.
* Evaluate using provided evaluation notebook.

---

## References

* [PlantVillage Dataset Publication](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0243243)
* TensorFlow & Keras Official Documentation
