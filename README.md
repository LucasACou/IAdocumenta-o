# IAdocumentacao
Importar bibliotecas
python
Copy code
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
pandas: Para manipulação e análise de dados.
train_test_split: Para dividir o conjunto de dados em conjuntos de treino e teste.
SVC (Support Vector Classifier): É o classificador SVM do Scikit-learn.
StandardScaler: Para padronizar (escalonar) as características.
accuracy_score e classification_report: Métricas para avaliar o desempenho do modelo.
Carregar os dados
python
Copy code
data = pd.read_csv('dados_cardiacos_com_genero_idade_relacionado.csv')
coluna = ['data', 'genero', 'idade']
X = data[coluna].values.tolist()
y = data['target'].values.tolist()
pd.read_csv: Carrega os dados de um arquivo CSV para um DataFrame do pandas.
coluna é uma lista das colunas que você deseja usar como características (features).
X são as características que usaremos para treinar o modelo.
y é a coluna alvo que o modelo tentará prever.
Dividir os dados em conjuntos de treino e teste
python
Copy code
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
train_test_split: Divide os dados em conjuntos de treino e teste.
Padronizar as features
python
Copy code
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
StandardScaler: Padroniza as características (escalonamento) para que todas tenham média zero e variância unitária.
Inicializar o classificador SVM
python
Copy code
svm = SVC(kernel='linear')
Inicializa o classificador SVM com um kernel linear. Você pode experimentar outros kernels como 'rbf', 'poly', entre outros.
Treinar o modelo SVM
python
Copy code
svm.fit(X_train, y_train)
Treina o modelo SVM nos dados de treino.
Fazer previsões
python
Copy code
predictions = svm.predict(X_test)
Usa o modelo treinado para fazer previsões nos dados de teste.
Avaliar o desempenho do modelo
python
Copy code
accuracy = accuracy_score(y_test, predictions)
print(f'Acurácia do modelo: {accuracy:.2f}')
print(classification_report(y_test, predictions))
Calcula a acurácia do modelo usando accuracy_score.
Exibe um relatório detalhado de métricas de desempenho usando classification_report.
documentação
