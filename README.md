import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import ward, average, single, complete, fcluster
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class CJ:
    def __init__(self,
                 datos: pd.DataFrame,
                 cluster: int,
                 algoritmo: str = 'euclidean',
                 salto: str = 'ward'):
        
        self.__datos = datos
        self.__cluster = cluster
        self.__algoritmo = algoritmo
        self.__salto = salto

        self.__distancias = None
        self.__agregaciones = None
        self.__grupos = None
        self.__centros = None

        escalar = StandardScaler()
        self.__datos_escalados = escalar.fit_transform(self.__datos)
        self.__datos_escalados = pd.DataFrame(self.__datos_escalados)
        self.__datos_escalados.columns = self.__datos.columns
        self.__datos_escalados.index = self.__datos.index

        self.__distancias = pdist(self.__datos_escalados, metric=self.__algoritmo)

        if self.__salto == 'ward':
            self.__agregaciones = ward(self.__distancias)
        elif self.__salto == 'average':
            self.__agregaciones = average(self.__distancias)
        elif self.__salto == 'single':
            self.__agregaciones = single(self.__distancias)
        elif self.__salto == 'complete':
            self.__agregaciones = complete(self.__distancias)
        else:
            raise ValueError(f"Salto no reconocido: {self.__salto}. Use 'ward', 'average', 'single' o 'complete'.")

        grupos = fcluster(self.__agregaciones, self.__cluster, criterion='maxclust')
        self.__grupos = grupos - 1

        datos_cluster = self.__datos.copy()
        datos_cluster['cluster'] = self.__grupos
        self.__centros = datos_cluster.groupby('cluster').mean()
        
    @property
    def datos(self) -> pd.DataFrame:
        return self.__datos

    @property
    def cluster(self) -> int:
        return self.__cluster

    @property
    def algoritmo(self) -> str:
        return self.__algoritmo

    @property
    def salto(self) -> str:
        return self.__salto

    @property
    def distancias(self):
        return self.__distancias
    
    @property
    def datos_escalados(self):
        return self.__datos_escalados

    @property
    def agregaciones(self):
        return self.__agregaciones

    @property
    def grupos(self):
        return self.__grupos

    @property
    def centros(self) -> pd.DataFrame:
        return self.__centros


    @datos.setter
    def datos(self, nuevo: pd.DataFrame):

        self.__datos = nuevo


    @cluster.setter
    def cluster(self, nuevo: int):

        self.__cluster = nuevo


    @algoritmo.setter
    def algoritmo(self, nuevo: str):

        self.__algoritmo = nuevo


    @salto.setter
    def salto(self, nuevo: str):

        self.__salto = nuevo


    def datos_cluster(self) -> pd.DataFrame:

        df = self.__datos.copy()
        df['cluster'] = self.__grupos
        return df

    def inercias(self) -> dict:

        promedio_global = self.__datos.mean()

        inercia_total = 0.0
        for i in range(self.__datos.shape[0]):
            inercia_total =  inercia_total + np.sum((self.__datos.iloc[i, :] - promedio_global) ** 2)

        inercia_intra = 0.0
        for i in range(self.__datos.shape[0]):
            clus_i = self.__grupos[i]
            inercia_intra = inercia_intra + np.sum((self.__datos.iloc[i, :] - self.__centros.iloc[clus_i, :]) ** 2)


        inercia_inter = 0.0
        for clus in range(self.__centros.shape[0]):
            n_obs = np.sum(self.__grupos == clus)
            inercia_inter = inercia_inter + (n_obs * np.sum((promedio_global - self.__centros.iloc[clus, :]) ** 2))

        return {
            'total': inercia_total,
            'inter': inercia_inter,
            'intra': inercia_intra
        }

    def grafico_barra(self, estandarizar: bool = False):

        df_centros = self.__centros.copy()

        if estandarizar:
            scaler = MinMaxScaler()

            df_centros = pd.DataFrame(
                scaler.fit_transform(df_centros),
                columns=df_centros.columns,
                index=df_centros.index
            )

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        df_centros.plot(kind='bar', ax=ax)

        ax.set_xlabel('Cl  ster', fontsize=12)
        ax.set_ylabel('Promedio', fontsize=12)
        ax.set_title('Promedio de cada variable por cl  ster')

        ax.legend(title="Variable", title_fontsize=12, fontsize=10, loc='best')

        plt.tight_layout()
        return fig

    def grafico_radar(self, estandarizar: bool = False):

        df_centros = self.__centros.copy()

        if estandarizar:
            scaler = MinMaxScaler()
            df_centros = pd.DataFrame(
                scaler.fit_transform(df_centros),
                columns=df_centros.columns,
                index=df_centros.index
            )
            df_centros = df_centros * 100

        columnas = df_centros.columns
        N = len(columnas)

        theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
        theta = np.concatenate([theta, [theta[0]]])

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})

 
        ax.set_theta_zero_location("N")   
        ax.set_theta_direction(-1)     
        ax.set_rlabel_position(90)        
        ax.spines["polar"].set_color("lightgrey")

        for idx in df_centros.index:
            valores = df_centros.loc[idx, :].values.tolist()
            valores += [valores[0]]

            ax.plot(theta, valores, linewidth=2, marker="o", label=f"Cluster {idx}")
            ax.fill(theta, valores, alpha=0.25)

        if estandarizar:
            plt.yticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100"], color="black", size=10)
            plt.ylim(-10, 110) 
        else:

            pass

        plt.xticks(theta, list(columnas) + [columnas[0]], color="black", size=10)

        plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1.1))
        ax.set_title('Gr  fico Radar de Promedios por Cl  ster', fontsize=14)

        plt.tight_layout()
        return fig

    def __str__(self):
        """
        Retorna una descripci  n b  sica de la clase y de sus atributos m  s relevantes.
        """
        desc = (
            "Clase CJ para Agrupaci  n Jer  rquica\n"
            f"Datos: {self.__datos.shape[0]} registros x {self.__datos.shape[1]} variables\n"
            f"N  mero de cl  steres: {self.__cluster}\n"
            f"M  trica de distancia: {self.__algoritmo}\n"
            f"M  todo de enlace: {self.__salto}\n"
            f"Grupos: {self.__grupos}\n"
            f"Centros: {self.__centros}\n"
        )
        return desc


kmedias = KMeans(n_clusters=3, n_init=10, max_iter=100, algorithm="elkan")
kmedias.fit(datos_escalados)


grupos = kmedias.labels_
grupos

datos_cluster = datos.copy()
datos_cluster["cluster"] = grupos
datos_cluster


promedios = datos_cluster.groupby("cluster").mean()
promedios


promedios_total = datos.mean()

inercia_total = 0
for i in range(datos.shape[0]):
  inercia_total = inercia_total + np.sum((datos.iloc[i, :] - promedios_total) ** 2)

inercia_total


intra_clase = 0
for i in range(datos.shape[0]):
  intra_clase = intra_clase + np.sum((datos.iloc[i, :] - promedios.iloc[grupos[i], :]) ** 2)

intra_clase


promedios_total = datos.mean()

inter_clase = 0
for i in range(promedios.shape[0]):
  n = np.sum(grupos == i)
  inter_clase = inter_clase + (n * np.sum((promedios_total - promedios.iloc[i, :]) ** 2))

inter_clase

fig, ax = plt.subplots(1, 1, figsize = (12, 8))
no_print = promedios.plot(kind = 'bar', ax = ax)

no_print = ax.set_xlabel('Cl  ster', fontsize = 12)
no_print = ax.set_ylabel('Promedio', fontsize = 12)

no_print = ax.legend(title = "Materia", title_fontsize='13', fontsize='11', loc='lower center')

plt.show()

scaler = MinMaxScaler()
escalar_promedios = pd.DataFrame(scaler.fit_transform(promedios), columns=promedios.columns, index=promedios.index)
escalar_promedios = escalar_promedios * 100

columnas = escalar_promedios.columns

N = len(columnas)
theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
theta = np.concatenate([theta, [theta[0]]])

fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={"projection": "polar"})

no_print = ax.set_theta_zero_location("N")
no_print = ax.set_theta_direction(-1)
no_print = ax.set_rlabel_position(90)
no_print = ax.spines["polar"].set_zorder(1)
no_print = ax.spines["polar"].set_color("lightgrey")

for fila in escalar_promedios.index:
  valores = escalar_promedios.loc[fila, columnas].values.tolist()
  valores = valores + [valores[0]]
  
  no_print = ax.plot(theta, valores, linewidth=1.75, linestyle="solid", label = "Cluster " + str(fila), marker="o", markersize=10)
  no_print = ax.fill(theta, valores, alpha=0.50)

no_print = plt.yticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100"], color="black", size=12)
no_print = plt.ylim(-10, 100)
no_print = plt.xticks(theta, columnas.to_list() + [columnas[0]], color="black", size=12)
no_print = plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1)) 

plt.show()


kmedoids = KMedoids(n_clusters=3, max_iter=100, metric="euclidean", method='pam')
kmedoids.fit(datos_escalados)

grupos = kmedoids.labels_
grupos

datos_cluster = datos.copy()
datos_cluster["cluster"] = grupos
datos_cluster
promedios = datos_cluster.groupby("cluster").mean()
promedios

promedios_total = datos.mean()

inercia_total = 0
for i in range(datos.shape[0]):
  inercia_total = inercia_total + np.sum((datos.iloc[i, :] - promedios_total) ** 2)

inercia_total


fig, ax = plt.subplots(1, 1, figsize = (12, 8))
no_print = promedios.plot(kind = 'bar', ax = ax)

no_print = ax.set_xlabel('Cl  ster', fontsize = 12)
no_print = ax.set_ylabel('Promedio', fontsize = 12)

no_print = ax.legend(title = "Materia", title_fontsize='13', fontsize='11', loc='lower center')

plt.show()

scaler = MinMaxScaler()
escalar_promedios = pd.DataFrame(scaler.fit_transform(promedios), columns=promedios.columns, index=promedios.index)
escalar_promedios = escalar_promedios * 100

columnas = escalar_promedios.columns

N = len(columnas)
theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
theta = np.concatenate([theta, [theta[0]]])

fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={"projection": "polar"})

no_print = ax.set_theta_zero_location("N")
no_print = ax.set_theta_direction(-1)
no_print = ax.set_rlabel_position(90)
no_print = ax.spines["polar"].set_zorder(1)
no_print = ax.spines["polar"].set_color("lightgrey")

for fila in escalar_promedios.index:
  valores = escalar_promedios.loc[fila, columnas].values.tolist()
  valores = valores + [valores[0]]
  
  no_print = ax.plot(theta, valores, linewidth=1.75, linestyle="solid", label = "Cluster " + str(fila), marker="o", markersize=10)
  no_print = ax.fill(theta, valores, alpha=0.50)

no_print = plt.yticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100"], color="black", size=12)
no_print = plt.ylim(-10, 100)
no_print = plt.xticks(theta, columnas.to_list() + [columnas[0]], color="black", size=12)
no_print = plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1)) 

plt.show()


cantidad_clusters = range(2, 10)
lista_kmedias = [KMeans(n_clusters=i).fit(datos_escalados) for i in cantidad_clusters]
varianza = [lista_kmedias[i].inertia_ for i in range(len(lista_kmedias))]

#Gr  fico
fig, ax = plt.subplots(figsize=(15, 8))
no_print = ax.plot(cantidad_clusters, varianza, 'o-')
no_print = ax.set_xlabel('N  mero de cl  steres')
no_print = ax.set_ylabel('Varianza explicada por cada cluster (Inercia Intraclases)')
no_print = ax.set_title('Codo de Jambu')
plt.show()


from sklearn.metrics import silhouette_score

cantidad_clusters = range(2, 10)
lista_kmedias = [KMeans(n_clusters=i).fit(datos_escalados) for i in cantidad_clusters]
silhouette = [silhouette_score(datos_escalados, lista_kmedias[i].labels_) for i in range(len(lista_kmedias))]

fig, ax = plt.subplots(figsize = (15,8))
no_print = ax.plot(cantidad_clusters, silhouette, 'o-')
no_print = ax.set_xlabel("N  mero de cl  steres")
no_print = ax.set_ylabel("Silhouette")
no_print = ax.set_title('M  todo de Silhouette')
plt.show()
