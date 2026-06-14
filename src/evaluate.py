from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def evaluate_model(y_test, predictions, model_name):
    acc = accuracy_score(y_test, predictions)
    cf = confusion_matrix(y_test, predictions)
    cr = classification_report(y_test, predictions)
    
    print(f"Model: {model_name}")
    print(f"Accuracy: {acc}")
    print(f"Confusion Matrix:\n{cf}")
    print(f"Classification Report:\n{cr}")

    return acc, cf, cr