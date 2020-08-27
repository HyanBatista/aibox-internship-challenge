import os
import urllib.request
from zipfile import ZipFile
from zipfile import BadZipFile
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from yellowbrick.classifier import ClassPredictionError
import seaborn as sns


# Defina os dados globais.
CLF_NAMES = [
    "Support vector machine (one-versus-one)",
    "Support vector machine (one-versus-rest)",
    "Stochastic gradient descent",
    "Random forest"
]
CLASSES = [-1, 0, 1]
CLASS_NAMES = ["phishing", "suspicious", "legitimate"]


DOWNLOAD_ROOT = 'https://github.com/HyanBatista/aibox-internship-challenge/blob/master/'
PHISHING_PATH = os.path.join('datasets', 'phishing')
PHISHING_URL = DOWNLOAD_ROOT + 'datasets/phishing/phishing.zip?raw=true'


# Extraia arquivos zip.
def extract_fishing_data(zip_path, dataset_path):
    try:
        # Extraia os arquivos.
        with ZipFile(zip_path) as phishing_zip:
            print("[INFO] extrating all files...")
            phishing_zip.extractall(path=dataset_path)
            print("[INFO] done!")
    except BadZipFile as bzf:
        """
        Se ocorrer alguma exceção relacionada ao estado do arquivo. Reinicie o script.
        """
        print("Error: {}".format(bzf))
        os.remove(zip_path)
        print("[INFO] restarting the script...")
        fetch_phishing_data()


# Busque os dados e os armazene em um diretório.
def fetch_phishing_data(phishing_path=PHISHING_PATH, phishing_url=PHISHING_URL):
    # Se o diretório não existir, crie-o.
    if not os.path.isdir(phishing_path):
        os.makedirs(phishing_path)

    # Crie um caminho local para o arquivo zip a ser baixado.
    zip_path = os.path.join(phishing_path, "phishing.zip")  

    # Se o arquivo zip ainda não tiver sido baixado, baixe-o.
    if not os.path.isfile(zip_path):
        print("[INFO] downloading the dataset...")
        urllib.request.urlretrieve(phishing_url, zip_path)
    
    # Extraia os arquivos.
    extract_fishing_data(zip_path, phishing_path)


# Carregue o dataset em um DataFrame.
def load_phishing_data(phishing_path=PHISHING_PATH):
    csv_path = os.path.join(phishing_path, 'Website Phishing.csv')
    return pd.read_csv(csv_path)


# Imprima as métricas precision, recall e f1_score formatadas.
def print_metrics(precision, recall, f1_score):
    sub = str.maketrans("1", "₁")
    metrics = {
        'Precision': precision,
        'Recall': recall,
        f"{'F1'.translate(sub)} score": f1_score,
    }
        
    return pd.DataFrame.from_dict(metrics, orient='index')


# Imprima a confusion matrix formatada.
def print_conf_mx(conf_mx):
    cols = ["TP", "FP", "TN", "FN"]
    table = {
        'phishing': [
            conf_mx[0, 0], 
            conf_mx[0, 1] + conf_mx[0, 2],
            conf_mx[1, 1] + conf_mx[1, 2] + conf_mx[2, 1] + conf_mx[2, 2],
            conf_mx[1, 0] + conf_mx[2, 0]
        ],
        'suspicious': [
            conf_mx[1, 1], 
            conf_mx[1, 0] + conf_mx[1, 2],
            conf_mx[0, 0] + conf_mx[0, 2] + conf_mx[2, 0] + conf_mx[2, 2],
            conf_mx[0, 1] + conf_mx[2, 1]
        ],
        'legitimate': [
            conf_mx[2, 2], 
            conf_mx[2, 0] + conf_mx[2, 1],
            conf_mx[0, 0] + conf_mx[0, 1] + conf_mx[1, 0] + conf_mx[1, 1],
            conf_mx[0, 2] + conf_mx[1, 2]
        ]
    }
    return pd.DataFrame.from_dict(table, orient='index', columns=cols)


# Imprima a acurácia formatada.
def print_accuracy(accuracy):
    table = {
        '1st folder': accuracy[0],
        '2nd folder': accuracy[1],
        '3rd folder': accuracy[2]
    }
    return pd.DataFrame.from_dict(table, orient='index')


# Plot o gráfico de class prediction error.
def show_visualizer(visualizer, X_train, X_test, y_train, y_test): 
    # Treine o visualizer.
    visualizer.fit(X_train, y_train)

    # Avalie o modelo no testing set. 
    visualizer.score(X_test, y_test)

    # Desenhe a visualização. 
    visualizer.show()


# Plote a área sob a curva da característica de operação do receptor de uma lista de classificadores.
def plot_multiclass_roc(clfs, X_test, y_test, clf_n=4, class_n=0, figsize=(12, 12)):

    y_test = label_binarize(y_test, classes=CLASSES)

    if clf_n == 4:
        y_score = [
            clfs[0].decision_function(X_test),
            clfs[1].decision_function(X_test),
            clfs[2].decision_function(X_test),
            clfs[3].predict_proba(X_test)
        ]
    
    elif clf_n == 1:
        y_score = [
            clfs[0].predict_proba(X_test)
        ]

    fpr = []
    tpr = []
    roc_auc = []
    
    for i in range(clf_n):
        fpr_aux, tpr_aux, _ = roc_curve(y_test[:, class_n], y_score[i][:, class_n])
        fpr.append(fpr_aux)
        tpr.append(tpr_aux)
        roc_auc.append(auc(fpr_aux, tpr_aux))

    plt.figure(figsize=figsize)
    for i in range(clf_n):
        if clf_n > 1:
            plt.plot(fpr[i], tpr[i], label=f"{CLF_NAMES[i]}, auc = {str(round(roc_auc[i], 3))}")
        else:
            plt.plot(fpr[i], tpr[i], label=f"{CLF_NAMES[3]}, auc = {str(round(roc_auc[i], 3))}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver operating characteristic related to {CLASS_NAMES[class_n]} class')
    plt.legend(loc="lower right")
    plt.show()