

# 💰 Smart-Expense-Analyzer

Smart-Expense-Analyzer is a data-driven tool designed to **analyze expense data** from Excel (`.xlsx`) or CSV (`.csv`) files. It automatically extracts key features, generates **summary statistics**, and can be extended to perform **predictive analytics** for smarter financial planning.

---

## 🚀 Features

* 📂 **Input Support**: Upload `.csv` or `.xlsx` files of transaction/expense data.
* 📊 **Summary Statistics**:

  * Total expenses & income
  * Category-wise spending
  * Monthly/weekly breakdown
  * Highest & lowest transactions
* 🧠 **Autonomous Feature Detection**:

  * Automatically detects columns like *Date*, *Amount*, *Category*, *Payment Method*, etc.
* 🔮 **Predictive Insights** (extendable):

  * Forecast future expenses
  * Identify unusual spending behavior
* 📈 **Visualization Ready**: Generates plots for easy understanding of expense patterns.

---

## 📂 Project Structure

```
Smart-Expense-Analyzer/
│── data/                # Sample input data files  
│── notebooks/           # Jupyter notebooks for exploration  
│── src/                 # Core source code  
│   ├── preprocess.py    # Data cleaning & feature extraction  
│   ├── analysis.py      # Summary statistics generation  
│   ├── predict.py       # Predictive modeling module  
│   └── visualize.py     # Charts & graphs  
│── app.py               # Main entry point (CLI/Streamlit)  
│── requirements.txt     # Dependencies  
│── README.md            # Project documentation  
```

---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/Smart-Expense-Analyzer.git
cd Smart-Expense-Analyzer
pip install -r requirements.txt
```

---

## ▶️ Usage

### 1. Run from Command Line

```bash
python app.py --file data/expenses.csv
```

### 2. Run as a Streamlit Web App

```bash
streamlit run app.py
```

### Example Output

* Expense summary (total, mean, median)
* Top 5 categories by spending
* Monthly expense trend chart
* Forecast for next month’s expenses

---

## 🧩 Dependencies

* Python 3.8+
* pandas
* numpy
* scikit-learn
* matplotlib / seaborn
* streamlit (if using web app)

Install via:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit
```

---

## 📊 Example Input

**expenses.csv**

| Date       | Category  | Amount | Payment\_Method | Notes           |
| ---------- | --------- | ------ | --------------- | --------------- |
| 2025-01-01 | Food      | 450    | UPI             | Dinner at KFC   |
| 2025-01-03 | Transport | 120    | Cash            | Auto fare       |
| 2025-01-04 | Shopping  | 1500   | Credit Card     | Online Purchase |

---

## ✅ Roadmap

* [x] Summary statistics module
* [x] Feature detection
* [ ] Expense forecasting model
* [ ] Anomaly detection (fraudulent transactions)
* [ ] Dashboard integration

---

## 🤝 Contributing

Contributions are welcome! Please open an **issue** or submit a **pull request** for improvements.

---

## 📜 License

This project is licensed under the **MIT License**.

---
