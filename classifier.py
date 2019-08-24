import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

census_data = pd.read_csv("adult.test", delimiter=',')

# Exibe o número de linhas e o número de colunas
print("Exibe o número de linhas e o número de colunas da base de dados")
print(census_data.shape)

# Divide os dados em dois conjuntos: Atributos e Classes
attributes_data = census_data.drop('class', axis=1)
classes_data = census_data['class']

# Cria atributos "dummies" para as colunas que não são numericas no conjunto de dados
new_attributes = pd.get_dummies(attributes_data, columns=['workclass', 'education', 'marital-status', 'occupation',	'relationship',	'race',
                                                  'sex', 'native-country'],
               drop_first=True, prefix=['workclass_', 'education_', 'marital-status_', 'occupation_',	'relationship_',	'race_',
                                        'sex_', 'native-country_'])
# print(new_attributes)

# Dividir os dados aleatóriamente em conjunto para aprendizado e conjunto para testes
X_train, X_test, y_train, y_test = train_test_split(new_attributes, classes_data, test_size=0.20) #20% do tamanho do arquivo será usado para testes
# X_train: segmento dos atributos para treinamento do modelo
# X_test : segmento dos atributos para avaliação do modelo
# y_train: segmento das classes para treinamento do modelo
# y_testn: segmento das classes para avaliação do modelo


# Treinar o modelo
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

# Aplicar o modelo gerado sobre os dados separados para testes
y_pred = classifier.predict(X_test)

print(y_pred)

# Avaliar o modelo: Acurácia e matriz de contingência
print("Resultado da Avaliação do Modelo")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Classificar uma nova instância
# print("Classificar [30, 1787, 19, 10, 79, 1, -1, 0, 0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,1]")
nova_instancia=[30, 181212, 10, 0, 0, 40,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
print(classifier.predict(nova_instancia))

