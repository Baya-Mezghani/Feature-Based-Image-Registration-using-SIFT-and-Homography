# Feature-Based Image Registration using SIFT and Homography 

## Overview
This project implements a **feature-based image registration pipeline** using classical computer vision techniques. The goal is to align a source image with a target image by detecting and matching keypoints, then estimating a geometric transformation.

The pipeline uses **SIFT features**, **descriptor matching**, and **RANSAC-based homography estimation** to achieve robust alignment.

---

## Pipeline

The system follows these steps:

1. **Load images**
   - Input images are converted to grayscale

2. **Feature Detection**
   - Detect keypoints and descriptors using SIFT

3. **Feature Matching**
   - Match descriptors using Brute-Force Matcher (BFMatcher)

4. **Homography Estimation**
   - Estimate transformation using RANSAC to remove outliers

5. **Image Warping**
   - Apply homography to align the source image to the target

6. **Visualization**
   - Display:
     - Keypoints
     - Matches
     - Inlier matches after RANSAC
     - Final aligned image

7. **GIF Generation**
   - Create a smooth animation showing the registration process

---

## Technologies Used

- Python
- OpenCV
- NumPy
- Matplotlib
- ImageIO

---

## Installation

Clone the repository:

```bash
git clone git@github.com:Baya-Mezghani/Feature-Based-Image-Registration-using-SIFT-and-Homography.git
cd Feature-Based-Image-Registration-using-SIFT-and-Homography
```

Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate 
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Usage: 
```bash
python main.py
```

The script will:

* Display intermediate results

* Align the images

* Generate a GIF showing the registration process

---

## Project Structure
```bash
Feature-Based-Image-Registration-using-SIFT-and-Homography/
│
├── images/                    
│   ├── mona_source.png
│   └── mona_target.jpg
│
├── results/  # GIFs
│  
├── main.py                     
├── requirements.txt            
├── README.md                   
└── .gitignore                  
```

---

## Author

**Baya Mezghani** 

📧 baya.mezghani@ensi-uma.tn
