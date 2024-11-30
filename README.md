# Computer-Vision-Project
specific code explanation is inside the python file.

## Face Verification Using FaceNet

This repository contains an implementation of a **Face Verification System** using the **FaceNet** model. The project is designed to perform face verification by comparing face embeddings generated from images and calculating their similarity. It follows a deep learning-based pipeline for face recognition, leveraging **pre-trained models** and advanced techniques like the **Triplet Loss function**.

---

### **Overview**

**Face verification** is the process of confirming the identity of an individual by matching their face to a reference image. It's commonly used in applications like unlocking mobile devices, identity verification at airports, and more.

This system is based on **FaceNet**, a deep neural network that encodes images into 128-dimensional embeddings. The closeness of these embeddings is used to verify if two images belong to the same person.

---

### **Features**

- **Face Embedding Extraction:** Uses a pre-trained FaceNet model to generate 128-dimensional vector embeddings for face images.
- **Similarity Measurement:** Calculates the Euclidean distance between embeddings to verify identity.
- **Triplet Loss Function:** Ensures embeddings for the same person are closer, and embeddings for different people are further apart.
- **Pre-trained Model Integration:** Saves time and resources by leveraging a pre-trained FaceNet model.
- **Verification Functionality:** Includes functions for:
  - Verifying an image against a known identity in a database.
  - Comparing two face images for similarity.

---

### **How It Works**

1. **Face Detection:** Detect faces in input images (not implemented here, assumes input images are cropped).
2. **Embedding Generation:** Convert face images into 128-dimensional embeddings using the pre-trained FaceNet model.
3. **Distance Calculation:** Compute the Euclidean distance between embeddings.
4. **Verification:** Determine if the distance is below a threshold (e.g., 0.7) to verify if images belong to the same person.

---

### **Core Files and Functions**

- **`triplet_loss()`**: Implements the triplet loss function for model training.
- **`img_to_encoding()`**: Converts face images into embeddings using the FaceNet model.
- **`verify_by_name()`**: Verifies an image against a known identity in the database.
- **`verify_by_image()`**: Compares two images to verify if they belong to the same person.

---

### **Requirements**

- Python 3.x
- TensorFlow
- Keras
- OpenCV
- NumPy
- Google Colab (for execution)

---

### **Sample Results**

#### **Examples**
- **Messi (Matching Images):**
  - Input Images: `Messi1.jpeg` and `Messi2.jpeg`
  - Result: *Same person*

- **Messi and Tom Hardy (Different Images):**
  - Input Images: `Messi1.jpeg` and `Tom_Hardy1.jpg`
  - Result: *Different persons*

#### **Triplet Loss Function**
The triplet loss ensures that:
- Similar images have embeddings with smaller distances.
- Dissimilar images have embeddings with larger distances.

---

### **Future Work**

- Integrate a face detection pipeline for automated input preprocessing.
- Improve performance using ArcFace or other advanced loss functions.
- Expand the system to include real-time face verification.
