# CS420 Final Project

## **1. Introduction**

This project fine-tunes the GPT-2 language model on a custom dataset of economics Q&A pairs. The goal is to improve GPT-2's ability to generate answers to important economics definitions. The project compares baseline GPT-2 responses with fine-tuned responses, saving the results in a CSV.

## **2. Getting Started**

This project is implemented in **Python 3.9+** and is compatible with **macOS, Linux, and Windows**.

### **2.1 Preparations**

1. Clone the repository:
```bash
git clone https://github.com/edenfitsum/csci420_finalproject.git
```

2. Navigate into the project folder:
```bash
cd csci420_finalproject
```

3. Set up and activate a virtual environment:

#### For macOS/Linux:
```bash
python -m venv ./venv/
source venv/bin/activate
```

#### For Windows:
```bash
pip install virtualenv
python -m virtualenv venv
venv\Scripts\activate
```

Once activated, your terminal prompt will show `(venv)` in front. To deactivate:
```bash
deactivate
```

### **2.2 Install Packages**

Install all required Python packages:
```bash
pip install -r requirements.txt
```

## **3. Run the Program**

To train the model and generate the CSV comparison of baseline vs fine-tuned answers:
```bash
python main.py
```
