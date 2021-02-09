import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from scipy.sparse import csr_matrix
import math

class Matrix_AnimeRecommendations(object):
    def create_matrix(self, df):
        print("Creating Anime Recommendations matrix...")
        list_cat = []
        cat = df.category_item.unique()

        print("\tFiltering unique categories...")
        for cate in cat:
            if cate is not np.nan:
                s = str(cate).split(", ")
                for i in s:
                    list_cat.append(i)
                
        df_aux = pd.DataFrame(list_cat)
        df_aux.columns = ["category_item"]
        cat = df_aux.category_item.unique()
        ids = df.id_item.values
        le_cat = LabelEncoder()
        le_cat.fit(cat)
        le_id = LabelEncoder()
        le_id.fit(ids)

        list_interacts = list()
        
        print("\tRecovering ids and categories...")
        for i, j in df.iterrows():
            print("\t\t\tLinha " + str(i+1) + " de " + str(len(df)), end = '\r', flush = True)
            s = str(j.values[2]).split(", ")
            for x in s:
                if not x == 'nan':
                    list_interacts.append([le_id.transform([j.values[0]])[0], le_cat.transform([x])[0], 1])

        print("")
        print("\tCreating sparse matrix...")
        matriz_de_coordenadas = np.array(list_interacts)
        linhas = matriz_de_coordenadas[:, 0]
        colunas = matriz_de_coordenadas[:, 1]
        valores = matriz_de_coordenadas[:, 2]
        matriz_esparsa = csr_matrix((valores, (linhas, colunas)), shape = (len(le_id.classes_), len(le_cat.classes_)))
        
        return matriz_esparsa, le_id

        