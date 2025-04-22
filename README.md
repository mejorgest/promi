datos = pd.read_csv('../../../../datos/MuestraCredito5000V2.csv', delimiter = ';', decimal = ".")
datos.info()


datos["IngresoNeto"] = datos["IngresoNeto"].astype('category')
datos["CoefCreditoAvaluo"] = datos["CoefCreditoAvaluo"].astype('category')
datos["MontoCuota"] = datos["MontoCuota"].astype('category')
datos["GradoAcademico"] = datos["GradoAcademico"].astype('category')
datos.info()


def distribucion_variable_predecir(data:pd.DataFrame, variable_predict:str):
  conteo     = data[variable_predict].value_counts()
  valores    = conteo.to_list()
  categorias = conteo.index.to_list()
  titulo     = "Distribución de la variable %s" % variable_predict
  
  fig, ax = plt.subplots()
  ax.bar(categorias, valores)
  
  ax.set_ylabel(variable_predict)
  ax.set_title(titulo)
  
  for i, valor in enumerate(valores):
    porc = round(valor / sum(valores) * 100, 2)
    text = str(valor) + "\n" + str(porc) + "%"
    ax.text(i, valor * 0.45, text, color='white', ha='center', fontsize=10)
  
  return(fig)


  fig = distribucion_variable_predecir(datos, "BuenPagador")
plt.show()

def poder_predictivo_numerica(data:pd.DataFrame, var:str, variable_predict:str):
  g = sns.FacetGrid(data, hue = variable_predict, height = 4, aspect = 1.8)
  g = g.map(sns.kdeplot, var, fill = True)
  g = g.add_legend()
  
  g.set_ylabels("Densidad")
  
  return(g)

  fig = poder_predictivo_numerica(datos, "MontoCredito", "BuenPagador")
plt.show()


def poder_predictivo_categorica(data: pd.DataFrame, var: str, variable_predict: str):
  titulo = "Distribución de la variable %s según la variable %s" % (var, variable_predict)
  
  fig, ax = plt.subplots()
  
  for cat_predictora in data[var].unique():
    conteo     = data.loc[data[var] == cat_predictora, variable_predict].value_counts()
    valores    = conteo.to_numpy()
    categorias = conteo.index.to_numpy()
    porcentaje = valores / sum(valores) * 100
    porcentaje = porcentaje.round(2)
    
    pos_text = 0
    for i in range(len(porcentaje)):
      if i == 0:
        ax.barh(cat_predictora, porcentaje[i], label=categorias[i], color = plt.cm.tab10(i))
      else:
        ax.barh(cat_predictora, porcentaje[i], left=porcentaje[i-1], label=categorias[i], color = plt.cm.tab10(i))
      
      ax.text(pos_text + porcentaje[i] / 2, cat_predictora,
              str(porcentaje[i]) + "%\n" + str(valores[i]), 
              va='center', ha='center', color='white', fontsize=10)
      pos_text = pos_text + porcentaje[i]
  
  ax.set_title(titulo)
  handles, labels = ax.get_legend_handles_labels()
  unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
  ax.legend(*zip(*unique), loc = 'upper center', bbox_to_anchor = (1.08, 1))
  
  return(fig)

  fig = poder_predictivo_categorica(datos, "IngresoNeto", "BuenPagador")
plt.show()

fig = poder_predictivo_categorica(datos, "CoefCreditoAvaluo", "BuenPagador")
plt.show()

fig = poder_predictivo_categorica(datos, "MontoCuota", "BuenPagador")
plt.show()

fig = poder_predictivo_categorica(datos, "GradoAcademico", "BuenPagador")
plt.show()

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

indices = indices_general(MC, labels)
for k in indices:
  print("\n%s:\n%s"%(k,str(indices[k])))



n = round(datos.shape[0] ** 0.5)

labels = ["Si", "No"]
ks = list(range(2, n))
errores = []

for k in ks:
  modelo = Pipeline(steps=[
    ('preprocesador', preprocesamiento),
    ('clasificador', KNeighborsClassifier(n_neighbors=k))
  ])
  no_print = modelo.fit(X_train, y_train)
  pred = modelo.predict(X_test)
  MC = confusion_matrix(y_test, pred, labels=labels)
  indices = indices_general(MC, labels)
  errores.append(indices["Error Global"])

fig, ax = plt.subplots()
no_print = ax.plot(ks, errores)
no_print = ax.set_xlabel("Valor de K")
no_print = ax.set_ylabel("Error Global")
plt.show()

datos = datos.loc[:, ["IngresoNeto", "CoefCreditoAvaluo", "MontoCuota", "GradoAcademico", "BuenPagador"]]
datos.info()

X = datos.loc[:, datos.columns != 'BuenPagador']
X
y = datos.loc[:, 'BuenPagador'].to_numpy()
y[0:6]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.85)

preprocesamiento = ColumnTransformer(
  transformers=[
    ('cat', OneHotEncoder(), ['IngresoNeto', 'CoefCreditoAvaluo', 'MontoCuota', 'GradoAcademico'])
  ]
)

modelo = Pipeline(steps=[
    ('preprocesador', preprocesamiento),
    ('clasificador', KNeighborsClassifier(n_neighbors=5))
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




