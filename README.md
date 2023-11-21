# IAdocumentacao

## Baixar as bibliotecas

```python
pip install pandas 
pip install scikit-learn
```

## Bibliotecas Utilizadas:

- **Pandas:** Para manipulação e análise de dados.
- **Sklearn**: Para a utilização de bibliotecas de apredizado de máquina. 
	- **train_test_split:** Para dividir o conjunto de dados em conjuntos de treino e teste.
	- **SVC (Support Vector Classifier):** É o classificador SVM do Scikit-learn.
	- **StandardScaler:** Para padronizar (escalonar) as características.
	- **accuracy_score** e **classification_report:** Métricas para avaliar o desempenho do modelo.

## Declaração das bibliotecas

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
```

## Carregar os Dados

```python
data = pd.read_csv('dados_cardiacos.csv')
coluna = ['data', 'genero', 'idade']
X = data[coluna].values.tolist()
y = data['target'].values.tolist()
```

### Descrição:

- **pd.read_csv:** Carrega os dados de um arquivo CSV para um DataFrame do pandas.
- **coluna:** É uma lista das colunas que você deseja usar como características (features).
- **X:** São as características que usaremos para treinar o modelo.
- **y:** É a coluna alvo que o modelo tentará prever.

## Dividir os Dados em Conjuntos de Treino e Teste

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

### Descrição:

- **train_test_split:** Divide os dados em conjuntos de treino e teste.
- **test_size:** 30% dos dados analisados são alocados na variavel de teste.
- **random_state:** É um número que determina a randomização dos dados que são separados.

## Padronizar as Features

```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### Descrição:

- **StandardScaler:** Padroniza as características (escalonamento) para que todas tenham média zero e variância unitária.

## Inicializar o Classificador SVM

```python
svm = SVC(kernel='linear')
```

### Descrição:

- Inicializa o classificador SVM com um kernel linear. Você pode experimentar outros kernels como 'rbf', 'poly', entre outros.

## Treinar o Modelo SVM

```python
svm.fit(X_train, y_train)
```

### Descrição:

- Treina o modelo SVM nos dados de treino.

## Fazer Previsões

```python
predictions = svm.predict(X_test)
```

### Descrição:

- Usa o modelo treinado para fazer previsões nos dados de teste.

## Avaliar o Desempenho do Modelo

```python
accuracy = accuracy_score(y_test, predictions)
print(f'Acurácia do modelo: {accuracy:.2f}')
print(classification_report(y_test, predictions))
```

### Descrição:

- Calcula a acurácia do modelo usando `accuracy_score`.
- Exibe um relatório detalhado de métricas de desempenho usando `classification_report`.

---
