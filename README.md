

# Real-Time Indian Sign Language (ISL) Gesture Recognition System

## Overview
The Real-Time Indian Sign Language (ISL) Gesture Recognition System aims to develop an advanced solution for identifying ISL gestures using machine learning techniques. Specifically designed for 26 English alphabet gesture sets, this system operates in real-time to enhance communication accessibility for individuals with hearing impairments.

## Key Features
- **Real-Time Recognition:** Utilizes machine learning models to recognize ISL gestures instantaneously, facilitating seamless communication.
- **Targeted Gesture Sets:** Focuses on 26 English alphabet gesture sets, covering essential communication needs.
- **Near-Perfect Performance:** Despite challenges, the system achieves near-perfect performance in practical scenarios, ensuring reliable and accurate gesture recognition.


## Module Use
- Tensorflow 
- Medipipe
- OpenCV
- scikit-learn





## Classifier Performance

| Classifier    | Accuracy (one hand) | Accuracy (two hands) |
|---------------|---------------------|----------------------|
| DNN           | 99.55%              | 99.94%               |
| k-NN          | 99.94%              | 99.91%               |
| SVM           | 99.89%              | 99.99%               |
| RandomForest  | 99.89%              | 99.98%               |

From the table, the K-NN model is deployed for one-handed gestures, while SVM is used for two-handed gestures.


## How to Use
1. **Clone the Repository**
    ```bash
    git clone https://github.com/yourusername/hand-sign-recognition-system.git
    cd hand-sign-recognition-system
    ```

2. **Install the Requirements**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the System**
   - Run `app.py` to start the system and initiate real-time Indian Sign Language (ISL) gesture recognition.

## Additional Information
This system is a multimodal solution combining computer vision (Medipipe and OpenCV) with machine learning (KNN and SVM) to achieve robust and accurate ISL gesture recognition in real-time.






