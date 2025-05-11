# 🍎 Fruit & Vegetable Disease Classification using CNN and ResNet

This project focuses on building a robust image classification model to distinguish between healthy and rotten fruits and vegetables using deep learning. It is designed for industrial applications in food safety, quality assurance, and agricultural value chains.

---

## 📊 Problem Statement

Timely identification of spoiled produce is critical in supply chains to reduce food waste, ensure consumer safety, and maintain quality standards. This project addresses that by automating the classification of 28 classes across fruits and vegetables (healthy vs. rotten).

---

## 🗃️ Dataset

- **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten)
- **Classes**: 28 total (14 fruit/vegetable types × 2 conditions)
- **Size**: ~9,000 images

---

## ⚙️ Project Structure
fruit-veg-disease-detector/
│
├── notebooks/
│ ├── 01_data_exploration.ipynb
│ ├── 02_cnn_baseline.ipynb
│ └── 03_resnet_transfer_learning.ipynb
│
├── models/
│ └── best_resnet_model.pth
│
├── utils/
│ └── metrics.py
│
├── data/
├── requirements.txt
└── README.md


---

## ✅ Results

| Model        | Accuracy | Precision | Recall | F1-score |
|--------------|----------|-----------|--------|----------|
| Baseline CNN | 84.6%    | 83.8%     | 83.8%  | 83.7%    |
| ResNet-18    | **93.9%** | 94.0%     | 93.9%  | 93.9%    |

🧠 Confusion matrix and per-class reports are provided in notebooks.

---

## 📈 Key Insights

- **ResNet-18** significantly outperforms the custom CNN on both accuracy and generalization.
- Some classes like `Jujube__Rotten` and `Potato__Healthy` were harder to classify — likely due to class imbalance and visual ambiguity.
- The model can be further improved with augmentation, dataset balancing, and domain-specific preprocessing.

---

## 🚀 Future Work

- ✅ Build a simple **Streamlit app** for real-time predictions
- 🔄 Improve data augmentation
- 🧠 Add model interpretability (Grad-CAM)
- 💻 Export to ONNX or deploy via Colab

---

## 📌 How to Run

1. Clone this repo:

```bash
git clone [https://github.com/ejmwesigwa/fruit-veg-freshness-detector.git](https://github.com/ejmwesigwa/fruit-veg-freshness-detector.git)
cd fruit-veg-disease-detector



2. Install dependencies

Bash
pip install -r requirements.txt


3. Open Jupyter Notebooks in the notebooks/ folder and run step-by-step.

✍️ Author
Enock Joseph Mwesigwa
🔗 [LinkedIn](https://www.linkedin.com/in/enock-joseph-mwesigwa/)
