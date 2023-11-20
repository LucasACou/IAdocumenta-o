# IAdocumentacao
Importar bibliotecas<br />
python<br />
Copy code<br />
import pandas as pd<br />
from sklearn.model_selection import train_test_split<br />
from sklearn.svm import SVC<br />
from sklearn.preprocessing import StandardScaler<br />
from sklearn.metrics import accuracy_score, classification_report<br />
pandas: Para manipulação e análise de dados.<br />
train_test_split: Para dividir o conjunto de dados em conjuntos de treino e teste.<br />
SVC (Support Vector Classifier): É o classificador SVM do Scikit-learn.<br />
StandardScaler: Para padronizar (escalonar) as características.<br />
accuracy_score e classification_report: Métricas para avaliar o desempenho do modelo.<br />
Carregar os dados<br />
python<br />
Copy code<br />
data = pd.read_csv('dados_cardiacos_com_genero_idade_relacionado.csv')<br />
coluna = ['data', 'genero', 'idade']<br />
X = data[coluna].values.tolist()<br />
y = data['target'].values.tolist()<br />
pd.read_csv: Carrega os dados de um arquivo CSV para um DataFrame do pandas.<br />
coluna é uma lista das colunas que você deseja usar como características (features).<br />
X são as características que usaremos para treinar o modelo.<br />
y é a coluna alvo que o modelo tentará prever.<br />
Dividir os dados em conjuntos de treino e teste<br />
python<br />
Copy code<br />
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)<br />
train_test_split: Divide os dados em conjuntos de treino e teste.<br />
Padronizar as features<br />
python<br />
Copy code<br />
scaler = StandardScaler()<br />
X_train = scaler.fit_transform(X_train)<br />
X_test = scaler.transform(X_test)<br />
StandardScaler: Padroniza as características (escalonamento) para que todas tenham média zero e variância unitária.<br />
Inicializar o classificador SVM<br />
python<br />
Copy code<br />
svm = SVC(kernel='linear')<br />
Inicializa o classificador SVM com um kernel linear. Você pode experimentar outros kernels como 'rbf', 'poly', entre outros.<br />
Treinar o modelo SVM<br />
python<br />
Copy code<br />
svm.fit(X_train, y_train)<br />
Treina o modelo SVM nos dados de treino.<br />
Fazer previsões<br />
python<br />
Copy code<br />
predictions = svm.predict(X_test)<br />
Usa o modelo treinado para fazer previsões nos dados de teste.<br />
Avaliar o desempenho do modelo<br />
python<br />
Copy code<br />
accuracy = accuracy_score(y_test, predictions)<br />
print(f'Acurácia do modelo: {accuracy:.2f}')<br />
print(classification_report(y_test, predictions))<br />
Calcula a acurácia do modelo usando accuracy_score.<br />
Exibe um relatório detalhado de métricas de desempenho usando classification_report.<br />
