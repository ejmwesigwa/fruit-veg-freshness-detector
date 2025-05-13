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
- Fine-grained challenge – many mis-labels occur where even humans struggle to tell “fresh” from “slightly spoiled”.

- Grad-CAM shows the model focuses on bruises, mold spots, or color changes – confirming it learns meaningful features.

- Some rare classes (< 100 images) still confuse the model; more data or class balancing would help
- The model can be further improved with augmentation, dataset balancing, and domain-specific preprocessing.

---

## 📌 How to Run

1. **Clone this repo:**

   ```bash
   git clone https://github.com/ejmwesigwa/fruit-veg-freshness-detector.git
   cd fruit-veg-disease-detector

2. **Install dependencies:**
   
    ```Bash
    pip install -r requirements.txt


3. **Open Jupyter Notebooks in the notebooks/ folder and run step-by-step.**


---
## ✍️ Author

Enock Joseph Mwesigwa


🔗 [LinkedIn](https://www.linkedin.com/in/enock-joseph-mwesigwa/)
