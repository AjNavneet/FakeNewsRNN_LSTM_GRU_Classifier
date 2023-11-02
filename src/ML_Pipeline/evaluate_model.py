import matplotlib.pyplot as plt
from datetime import date
from os.path import exists
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Function to plot model accuracy history
def performance_history(history):
    """
    Plot model accuracy history during training.

    Args:
    history (object): Training history object containing accuracy information.

    Returns:
    None
    """
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

# Function to evaluate model and calculate test loss and accuracy
def model_evaluation(model, X_test, y_test):
    """
    Evaluate the model and calculate test loss and accuracy.

    Args:
    model (object): Trained model to evaluate.
    X_test (numpy.ndarray): Test data.
    y_test (numpy.ndarray): True labels of test data.

    Returns:
    list: Test loss and test accuracy.
    """
    score = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
    return score

# Function to generate a performance report including accuracy, precision, recall, and F1 score
def performance_report(model, testX, testy, model_name, report_dir='../output/reports/'):
    """
    Generate a performance report including accuracy, precision, recall, and F1 score.

    Args:
    model (object): Trained model.
    testX (numpy.ndarray): Test data.
    testy (numpy.ndarray): True labels of test data.
    model_name (str): Name of the model.
    report_dir (str): Directory to save the report CSV file (default is '../output/reports/').

    Returns:
    pandas.DataFrame: DataFrame containing performance metrics.
    """
    time = date.today()

    yhat_probs = model.predict(testX, verbose=0)
    yhat_classes = model.predict_classes(testX, verbose=0)
    yhat_probs = yhat_probs[:, 0]
    yhat_classes = yhat_classes[:, 0]

    accuracy = accuracy_score(testy, yhat_classes)
    print('Accuracy: %f' % accuracy)
    precision = precision_score(testy, yhat_classes)
    print('Precision: %f' % precision)
    recall = recall_score(testy, yhat_classes)
    print('Recall: %f' % recall)
    f1 = f1_score(testy, yhat_classes)
    print('F1 score: %f' % f1)

    if exists(report_dir + 'report.csv'):
        total_cost_df = pd.read_csv(report_dir + 'report.csv', index_col=0)
    else:
        total_cost_df = pd.DataFrame(columns=['time', 'name', 'Precision', 'Recall', 'f1_score', 'accuracy'])

    total_cost_df = total_cost_df.append(
            {'time': time, 'name': model_name, 'Precision': precision, 'Recall': recall, 'f1_score': f1, 'accuracy': accuracy},
            ignore_index=True)
    total_cost_df.to_csv(report_dir + 'report.csv')
    return total_cost_df
