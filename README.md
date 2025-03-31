datos = pd.read_csv('../../.././../datos/EjemploEstudiantes.csv', delimiter = ';', decimal = ",", index_col = 0)
datos

escalar = StandardScaler()
datos_escalados = escalar.fit_transform(datos)
datos_escalados = pd.DataFrame(datos_escalados)
datos_escalados.columns = datos.columns
datos_escalados.index = datos.index
datos_escalados

pca = PCA(n_components = 5)
pca.fit(datos_escalados)

individuos = pca.row_coordinates(datos_escalados)
individuos

x = individuos.iloc[:, 0]
y = individuos.iloc[:, 1]

fig, ax = plt.subplots(figsize = (10, 6))

no_print = ax.scatter(x, y, color = 'steelblue')

no_print  = ax.axhline(y = 0, color = 'dimgrey', linestyle = '--')
no_print  = ax.axvline(x = 0, color = 'dimgrey', linestyle = '--')
inercia_x = round(pca.percentage_of_variance_[0], 2)
inercia_y = round(pca.percentage_of_variance_[1], 2)
no_print  = ax.set_xlabel('Componente 1' + ' (' + str(inercia_x) + '%)')
no_print  = ax.set_ylabel('Componente 2' + ' (' + str(inercia_y) + '%)')

for i in range(individuos.shape[0]):
  no_print = ax.annotate(individuos.index[i], (x[i], y[i]))

plt.show()

umap = UMAP(n_components = 2, n_neighbors = 2)
individuos = umap.fit_transform(datos_escalados)
individuos = pd.DataFrame(individuos, index=datos_escalados.index)
individuos

x = individuos.iloc[:, 0]
y = individuos.iloc[:, 1]

fig, ax = plt.subplots(figsize = (10, 6))

no_print = ax.scatter(x, y, color = 'steelblue')

no_print  = ax.axhline(y = 0, color = 'dimgrey', linestyle = '--')
no_print  = ax.axvline(x = 0, color = 'dimgrey', linestyle = '--')
no_print  = ax.set_xlabel('Componente 1')
no_print  = ax.set_ylabel('Componente 2')

for i in range(individuos.shape[0]):
  no_print = ax.annotate(individuos.index[i], (x[i], y[i]))

plt.show()

tsne = TSNE(n_components=2, perplexity=2, learning_rate='auto', init='random')
individuos = tsne.fit_transform(datos_escalados)
individuos = pd.DataFrame(individuos, index=datos_escalados.index)
individuos

x = individuos.iloc[:, 0]
y = individuos.iloc[:, 1]

fig, ax = plt.subplots(figsize = (10, 6))

no_print = ax.scatter(x, y, color = 'steelblue')

no_print  = ax.axhline(y = 0, color = 'dimgrey', linestyle = '--')
no_print  = ax.axvline(x = 0, color = 'dimgrey', linestyle = '--')
no_print  = ax.set_xlabel('Componente 1')
no_print  = ax.set_ylabel('Componente 2')

for i in range(individuos.shape[0]):
  no_print = ax.annotate(individuos.index[i], (x[i], y[i]))

plt.show()

datos = pd.read_csv('../../.././../datos/iris.csv', delimiter = ';', decimal = ",")
pred = datos["tipo"]
datos = datos.iloc[:, 0:4]

escalar = StandardScaler()
datos_escalados = escalar.fit_transform(datos)
datos_escalados = pd.DataFrame(datos_escalados)
datos_escalados.columns = datos.columns
datos_escalados.index = datos.index
datos_escalados

pca = PCA(n_components = 5)
no_print = pca.fit(datos_escalados)

individuos = pca.row_coordinates(datos_escalados)

x = individuos.iloc[:, 0]
y = individuos.iloc[:, 1]

fig, ax = plt.subplots(figsize = (10, 6))

for cat in pred.unique():
  no_print = ax.scatter(x[pred == cat], y[pred == cat], label = cat)

no_print  = ax.axhline(y = 0, color = 'dimgrey', linestyle = '--')
no_print  = ax.axvline(x = 0, color = 'dimgrey', linestyle = '--')
no_print  = ax.set_xlabel('Componente 1')
no_print  = ax.set_ylabel('Componente 2')

no_print = plt.legend()
plt.show()

umap = UMAP(n_components = 2, n_neighbors = 25)
individuos = umap.fit_transform(datos_escalados)
individuos = pd.DataFrame(individuos, index=datos_escalados.index)

x = individuos.iloc[:, 0]
y = individuos.iloc[:, 1]

fig, ax = plt.subplots(figsize = (10, 6))

for cat in pred.unique():
  no_print = ax.scatter(x[pred == cat], y[pred == cat], label = cat)

no_print  = ax.axhline(y = 0, color = 'dimgrey', linestyle = '--')
no_print  = ax.axvline(x = 0, color = 'dimgrey', linestyle = '--')
no_print  = ax.set_xlabel('Componente 1')
no_print  = ax.set_ylabel('Componente 2')

no_print = plt.legend()
plt.show()

tsne = TSNE(n_components=2, perplexity=25, learning_rate='auto', init='random')
individuos = tsne.fit_transform(datos_escalados)
individuos = pd.DataFrame(individuos, index=datos_escalados.index)

x = individuos.iloc[:, 0]
y = individuos.iloc[:, 1]

fig, ax = plt.subplots(figsize = (10, 6))

for cat in pred.unique():
  no_print = ax.scatter(x[pred == cat], y[pred == cat], label = cat)

no_print  = ax.axhline(y = 0, color = 'dimgrey', linestyle = '--')
no_print  = ax.axvline(x = 0, color = 'dimgrey', linestyle = '--')
no_print  = ax.set_xlabel('Componente 1')
no_print  = ax.set_ylabel('Componente 2')

no_print = plt.legend()
plt.show()

from sklearn import datasets

digitos = datasets.load_digits()
datos = pd.DataFrame(digitos.data)
pred  = pd.DataFrame(digitos.target)
pred  = pred.iloc[:, 0]

escalar = StandardScaler()
datos_escalados = escalar.fit_transform(datos)
datos_escalados = pd.DataFrame(datos_escalados)
datos_escalados.columns = datos.columns
datos_escalados.index = datos.index
datos_escalados

pca = PCA(n_components = 5)
no_print = pca.fit(datos_escalados)

individuos = pca.row_coordinates(datos_escalados)

x = individuos.iloc[:, 0]
y = individuos.iloc[:, 1]

fig, ax = plt.subplots(figsize = (10, 6))

for cat in pred.unique():
  no_print = ax.scatter(x[pred == cat], y[pred == cat], label = cat)

no_print  = ax.axhline(y = 0, color = 'dimgrey', linestyle = '--')
no_print  = ax.axvline(x = 0, color = 'dimgrey', linestyle = '--')
no_print  = ax.set_xlabel('Componente 1')
no_print  = ax.set_ylabel('Componente 2')

no_print = plt.legend()
plt.show()

umap = UMAP(n_components = 2, n_neighbors = 89)
individuos = umap.fit_transform(datos_escalados)
individuos = pd.DataFrame(individuos, index=datos_escalados.index)

x = individuos.iloc[:, 0]
y = individuos.iloc[:, 1]

fig, ax = plt.subplots(figsize = (10, 6))

for cat in pred.unique():
  no_print = ax.scatter(x[pred == cat], y[pred == cat], label = cat)

no_print  = ax.axhline(y = 0, color = 'dimgrey', linestyle = '--')
no_print  = ax.axvline(x = 0, color = 'dimgrey', linestyle = '--')
no_print  = ax.set_xlabel('Componente 1')
no_print  = ax.set_ylabel('Componente 2')

no_print = plt.legend()
plt.show()

tsne = TSNE(n_components=2, perplexity=89, learning_rate='auto', init='random')
individuos = tsne.fit_transform(datos_escalados)
individuos = pd.DataFrame(individuos, index=datos_escalados.index)

x = individuos.iloc[:, 0]
y = individuos.iloc[:, 1]

fig, ax = plt.subplots(figsize = (10, 6))

for cat in pred.unique():
  no_print = ax.scatter(x[pred == cat], y[pred == cat], label = cat)

no_print  = ax.axhline(y = 0, color = 'dimgrey', linestyle = '--')
no_print  = ax.axvline(x = 0, color = 'dimgrey', linestyle = '--')
no_print  = ax.set_xlabel('Componente 1')
no_print  = ax.set_ylabel('Componente 2')

no_print = plt.legend()
plt.show()
