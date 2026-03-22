# Lip Reading AI - Hackathon project
## 👄 Overview
This project performs lip reading from video using deep learning. It extracts mouth regions from video frames and uses a CNN+LSTM model with CTC loss to predict spoken sentences.

## 🏗️ Architecture
- **Face Detection**: OpenCV Haar Cascade
- **Mouth Extraction**: Crops mouth region from each frame
- **Feature Extraction**: 4-layer CNN
- **Sequence Modeling**: Bidirectional LSTM
- **Decoding**: CTC loss with character-level encoding

## 📁 Project Structure
- \step1_check_data.py\ - Dataset exploration
- \step2_extract_mouth.py\ - Mouth extraction from videos  
- \dataset_loader.py\ - PyTorch dataset loader
- \step4_build_model.py\ - CNN+LSTM model definition
- \step5_train.py\ - Training loop
- \step6_predict.py\ - Generate predictions
- \create_train_csv.py\ - Create training metadata

## 🚀 How to Run
1. Install dependencies: \pip install -r requirements.txt\
2. Extract mouth regions: \python step2_extract_mouth.py\
3. Train model: \python step5_train.py\
4. Generate predictions: \python step6_predict.py\

## 📊 Results
- Trained on 37 videos initially
- Loss decreased from 16.7 to 0.98 in 20 epochs
- Final submission: \submission.csv\

## 👥 Team
- Elham
