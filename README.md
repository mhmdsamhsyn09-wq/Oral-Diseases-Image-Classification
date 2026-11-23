# Oral Disease Classifier - Gradio App

## Folder layout
```
app.py
requirements.txt
best_resnet.pth        <- your saved ResNet weights
classes.json           <- optional, e.g. ["Calculus", "Caries", ...] (same order as training)
results/               <- optional, shown in the "Model Results" tab
    metrics.json       {"accuracy": 0.95, "precision": 0.94, "recall": 0.94, "f1": 0.94}
    confusion_matrix.png
    training_curves.png
    comparison.csv     model,accuracy,f1
data/test/             <- optional ImageFolder test set for live evaluation
```

## Run
```
pip install -r requirements.txt
python app.py
```
If your weights use a different ResNet, set it: `ARCH=resnet18 MODEL_PATH=my_model.pth python app.py`
(on Windows PowerShell: `$env:ARCH="resnet18"; python app.py`).

## Run on Kaggle / Colab
Upload the files, `!pip install -q gradio`, then in `app.py` change the last line to
`demo.launch(share=True)` to get a public link.
