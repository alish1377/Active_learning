from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from .mlflow_handler import MLFlowHandler


def get_plots(model, test_loader, args, mlflow_handler: MLFlowHandler):
    print('Making  prediction ')
    n_classes = args.n_classes
    predictions = model.predict(test_loader, steps=(test_loader.n // test_loader.batch_size + 1))
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_loader.classes  # [:len(test_loader) * args.batch_size]
    correct_count = 0.0
    for i, y1 in enumerate(y_pred):
        if y1 == y_true[i]:
            correct_count = correct_count + 1
    print(f"accuracy of prediction  is {correct_count / len(y_pred)}")
    # Metrics: Confusion Matrix
    con_mat = confusion_matrix(y_true, y_pred)
    print(con_mat)
    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
    con_mat_df = pd.DataFrame(con_mat_norm, index=[i for i in range(n_classes)], columns=[i for i in range(n_classes)])
    figure = plt.figure(figsize=(n_classes, n_classes))
    sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('confusion matrix')
    plt.show()
    mlflow_handler.add_figure(figure, 'images/confusion_matrix.png')
    report = classification_report(y_true, y_pred)
    mlflow_handler.add_report(report, 'text/report.txt')
    print(report)

    print("Evaluating model")
    test_score = model.evaluate(test_loader, batch_size=test_loader.batch_size)
    for i, score in enumerate(test_score):
        print(f"{model.metrics_names[i]}: {score}")
