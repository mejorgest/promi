datos = pd.read_csv('../../../../datos/MuestraCredito5000V2.csv', delimiter = ';', decimal = ".")
datos.info()


datos["IngresoNeto"] = datos["IngresoNeto"].astype('category')
datos["CoefCreditoAvaluo"] = datos["CoefCreditoAvaluo"].astype('category')
datos["MontoCuota"] = datos["MontoCuota"].astype('category')
datos["GradoAcademico"] = datos["GradoAcademico"].astype('category')
datos.info()


X = datos.loc[:, datos.columns != 'BuenPagador']
X

y = datos.loc[:, 'BuenPagador'].to_numpy()
y[0:6]


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.85)

preprocesamiento = ColumnTransformer(
  transformers=[
    ('cat', OneHotEncoder(), ['IngresoNeto', 'CoefCreditoAvaluo', 'MontoCuota', 'GradoAcademico']),
    ('num', StandardScaler(), ['MontoCredito'])
  ]
)

modelo = Pipeline(steps=[
    ('preprocesador', preprocesamiento),
    ('clasificador', KNeighborsClassifier(n_neighbors=3, metric = "minkowski"))
])

modelo.fit(X_train, y_train)

pred = modelo.predict(X_test)
pred

labels = ["Si", "No"]
MC = confusion_matrix(y_test, pred, labels=labels)

MC

def indices_general(MC, nombres = None):
  precision_global = np.sum(MC.diagonal()) / np.sum(MC)
  error_global     = 1 - precision_global
  precision_categoria  = pd.DataFrame(MC.diagonal()/np.sum(MC,axis = 1)).T
  if nombres!=None:
    precision_categoria.columns = nombres
    return {"Matriz de Confusión":MC, 
            "Precisión Global":   precision_global, 
            "Error Global":       error_global, 
            "Precisión por categoría":precision_categoria}

indices = indices_general(MC, labels)
for k in indices:
  print("\n%s:\n%s"%(k,str(indices[k])))


datos = pd.read_csv('../../../../datos/MuestraCredito5000V2.csv', delimiter = ';', decimal = ".")
datos.info()

datos["IngresoNeto"] = datos["IngresoNeto"].astype('category')
datos["CoefCreditoAvaluo"] = datos["CoefCreditoAvaluo"].astype('category')
datos["MontoCuota"] = datos["MontoCuota"].astype('category')
datos["GradoAcademico"] = datos["GradoAcademico"].astype('category')
datos.info()

X = datos.loc[:, datos.columns != 'BuenPagador']
X

y = datos.loc[:, 'BuenPagador'].to_numpy()
y[0:6]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.85)

preprocesamiento = ColumnTransformer(
  transformers=[
    ('cat', OneHotEncoder(), ['IngresoNeto', 'CoefCreditoAvaluo', 'MontoCuota', 'GradoAcademico']),
    ('num', StandardScaler(), ['MontoCredito'])
  ]
)
modelo = Pipeline(steps=[
    ('preprocesador', preprocesamiento),
    ('clasificador', DecisionTreeClassifier(criterion='gini', max_depth=3, min_samples_split=10, min_samples_leaf=5))
])

modelo.fit(X_train, y_train)

pred = modelo.predict(X_test)
pred

columnas = modelo['preprocesador'].get_feature_names_out()
clases   = [str(clases) for clases in modelo["clasificador"].classes_]

fig, ax = plt.subplots()
no_print = plot_tree(
  modelo["clasificador"],
  filled=True,
  feature_names=columnas,
  class_names=clases,
  ax=ax
)
plt.show()

labels = ["Si", "No"]
MC = confusion_matrix(y_test, pred, labels=labels)
MC

def indices_general(MC, nombres = None):
  precision_global = np.sum(MC.diagonal()) / np.sum(MC)
  error_global     = 1 - precision_global
  precision_categoria  = pd.DataFrame(MC.diagonal()/np.sum(MC,axis = 1)).T
  if nombres!=None:
    precision_categoria.columns = nombres
    return {"Matriz de Confusión":MC, 
            "Precisión Global":   precision_global, 
            "Error Global":       error_global, 
            "Precisión por categoría":precision_categoria}

indices = indices_general(MC, labels)
for k in indices:
  print("\n%s:\n%s"%(k,str(indices[k])))



datos = pd.read_csv('../../../../datos/MuestraCredito5000V2.csv', delimiter = ';', decimal = ".")
datos.info()


datos["IngresoNeto"] = datos["IngresoNeto"].astype('category')
datos["CoefCreditoAvaluo"] = datos["CoefCreditoAvaluo"].astype('category')
datos["MontoCuota"] = datos["MontoCuota"].astype('category')
datos["GradoAcademico"] = datos["GradoAcademico"].astype('category')
datos.info()


X = datos.loc[:, datos.columns != 'BuenPagador']
X

y = datos.loc[:, 'BuenPagador'].to_numpy()
y[0:6]


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.85)

preprocesamiento = ColumnTransformer(
  transformers=[
    ('cat', OneHotEncoder(), ['IngresoNeto', 'CoefCreditoAvaluo', 'MontoCuota', 'GradoAcademico']),
    ('num', StandardScaler(), ['MontoCredito'])
  ]
)

arbol = DecisionTreeClassifier(max_depth=4)

modelo = Pipeline(steps=[
    ('preprocesador', preprocesamiento),
    ('clasificador', AdaBoostClassifier(n_estimators = 20, learning_rate = 1.0, estimator = arbol))
])

modelo.fit(X_train, y_train)

pred = modelo.predict(X_test)
pred

labels = ["Si", "No"]
MC = confusion_matrix(y_test, pred, labels=labels)
MC

def indices_general(MC, nombres = None):
  precision_global = np.sum(MC.diagonal()) / np.sum(MC)
  error_global     = 1 - precision_global
  precision_categoria  = pd.DataFrame(MC.diagonal()/np.sum(MC,axis = 1)).T
  if nombres!=None:
    precision_categoria.columns = nombres
  
  return {"Matriz de Confusión":MC, 
          "Precisión Global":   precision_global, 
          "Error Global":       error_global, 
          "Precisión por categoría":precision_categoria}

indices = indices_general(MC, labels)
for k in indices:
  print("\n%s:\n%s"%(k,str(indices[k])))



datos = pd.read_csv('../../../../datos/MuestraCredito5000V2.csv', delimiter = ';', decimal = ".")
datos.info()

datos["IngresoNeto"] = datos["IngresoNeto"].astype('category')
datos["CoefCreditoAvaluo"] = datos["CoefCreditoAvaluo"].astype('category')
datos["MontoCuota"] = datos["MontoCuota"].astype('category')
datos["GradoAcademico"] = datos["GradoAcademico"].astype('category')
datos.info()


X = datos.loc[:, datos.columns != 'BuenPagador']
X

from sklearn.preprocessing import LabelEncoder

codificacion = LabelEncoder()
y = datos.loc[:, 'BuenPagador'].to_numpy()
y = codificacion.fit_transform(y)
y[0:6]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.85)

preprocesamiento = ColumnTransformer(
  transformers=[
    ('cat', OneHotEncoder(), ['IngresoNeto', 'CoefCreditoAvaluo', 'MontoCuota', 'GradoAcademico']),
    ('num', StandardScaler(), ['MontoCredito'])
  ]
)

modelo = Pipeline(steps=[
    ('preprocesador', preprocesamiento),
    ('clasificador', xgb.XGBClassifier(n_estimators = 20, learning_rate = 1.0, max_depth = 4))
])

modelo.fit(X_train, y_train)


pred = modelo.predict(X_test)
pred

MC = confusion_matrix(y_test, pred)
MC

def indices_general(MC, nombres = None):
  precision_global = np.sum(MC.diagonal()) / np.sum(MC)
  error_global     = 1 - precision_global
  precision_categoria  = pd.DataFrame(MC.diagonal()/np.sum(MC,axis = 1)).T
  if nombres!=None:
    precision_categoria.columns = nombres
  
  return {"Matriz de Confusión":MC, 
          "Precisión Global":   precision_global, 
          "Error Global":       error_global, 
          "Precisión por categoría":precision_categoria}

indices = indices_general(MC, codificacion.classes_.tolist())
for k in indices:
  print("\n%s:\n%s"%(k,str(indices[k])))

  
