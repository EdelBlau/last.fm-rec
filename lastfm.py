# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.sparse import csr_matrix

# display results to 3 decimal points, not in scientific notation
pd.set_option('display.float_format', lambda x: '%.3f' % x)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

#https://beckernick.github.io/music_recommender/
#El dataset de Last.fm se divide en dos, los datos de actividad y los de perfil
#los datos de actividad son 360k registros de información sobre usuarios individuales
# detalla cuantas veces ha escuchado un usuario una canción
# los datos de perfil indican dónde reside el usuario
#importante los datos en tsv, nunca hacer transformación de tsv a csv porque pandas no detecta columnas
user_data = pd.read_table('../input/usersha1-artmbid-artname-plays.tsv',
                          header = None, nrows = 2e7,
                          names = ['users', 'musicbrainz-artist-id', 'artist-name', 'plays'],
                          usecols = ['users', 'artist-name', 'plays'])
user_profiles = pd.read_table('../input/usersha1-profile.tsv',
                          header = None,
                          names = ['users', 'gender', 'age', 'country', 'signup'],
                          usecols = ['users', 'country'])

#revisamos el contenido de los data sets
user_data.head()
user_profiles.head()
#como se va a realizar un filtrado colaborativo basado en elementos, las recomendaciones se basarán en los patrones de escucha de artistas
#los artistas menos conocidos tendrán escuchas de menos usuarios, lo que añadirá ruido al patrón.
# esto haría que las predicciones sean muy sensibles con usuarios concretos, por lo que sólo se valorarán artistas populares

#para buscar los artistas populares calculamos las escuchas totales de cada artista
#como nuestro archivo de escuchas tiene una fila por usuario y artista, hay que agregarlo al nivel del artista
#con pandas podemos agrupar por nombre de artista y calcular el número de escuchas
#si el nombre de artista no existe no funcionará, por lo que primero se eliminan esas filas

if user_data['artist-name'].isnull().sum() > 0:
    user_data = user_data.dropna(axis = 0, subset = ['artist-name'])
    
artist_plays = (user_data.
     groupby(by = ['artist-name'])['plays'].
     sum().
     reset_index().
     rename(columns = {'plays': 'total_artist_plays'})
     [['artist-name', 'total_artist_plays']]
    )
artist_plays.head()
# ahora podemos agrupar el total de escuchas en la actividad de usuario
#
user_data_with_artist_plays = user_data.merge(artist_plays, left_on = 'artist-name', right_on = 'artist-name', how = 'left')
user_data_with_artist_plays.head()

#al tener tantos artistas es altamente probable que muchos artistas se hayan escuchado pocas veces
#estadisticas descriptivas de los datos

artist_plays['total_artist_plays'].describe()

#de media un artista sólo ha sido escuchado unas 200 veces
#vamos a mirar los puntos altos

artist_plays['total_artist_plays'].quantile(np.arange(.9, 1, .01))

#Un 1% de los artistas apenas tienen 200k escuchas o más, 2% tienen 80k o más y 3% tienen 40k o más
#como hay tantos artistas, nos limitamos al top 3%
# es una medida arbitraria, pero nos deja con 9k artistas
# nos apañamos con estos para evitar el problema del ruido

popularity_threshold = 40000
user_data_popular_artists = user_data_with_artist_plays.query('total_artist_plays >= @popularity_threshold')
user_data_popular_artists.head()

#filtrando sólo los usuarios de US
# así se filtrará solo entre los artistas y usuarios americanos

#primero se junta la información del perfil de usuario con la info del país 
# después se filtra para obtener usuarios solo de estados unidos

combined = user_data_popular_artists.merge(user_profiles, left_on = 'users', right_on = 'users', how = 'left')
usa_data = combined.query('country == \'United States\'')
usa_data.head()


#antes de comenzar el analisis comprobamos que los datos sean consistentes internamente.
#cada usuario debe tener un conteo de escuchas por cada artista
# así que buscamos por instancias donde las filas tengan los mismos valores de users y artist-name

if not usa_data[usa_data.duplicated(['users', 'artist-name'])].empty:
    initial_rows = usa_data.shape[0]

    print('Initial dataframe shape {0}'.format(usa_data.shape)) 
    usa_data = usa_data.drop_duplicates(['users', 'artist-name'])
    current_rows = usa_data.shape[0]
    print('New dataframe shape {0}'.format(usa_data.shape)) 
    print('Removed {0} rows'.format(initial_rows - current_rows)) 


#implementar el modelo de vecinos cercanos
# queremos que los datos estén en un array m x n, donde m es el número de artistas y n el número de usuarios.
# para ello pivotamos el dataset para tener a artistas como filas y usuarios como columnas
# los nulos se rellenarán con 0 ya que se realizarán operaciones de algebra lineal (distancia entre vectore)
# por último transformamos los valores del dataframe en una matriz dispersa de scipy para calculos más eficientes

wide_artist_data = usa_data.pivot(index = 'artist-name', columns = 'users', values = 'plays').fillna(0)
wide_artist_data_sparse = csr_matrix(wide_artist_data.values)

#ajustando el modelo
#se inicializa la clase NearestNeighbors como model_knn
#especificando metric = cosine, el modelo medirá la similitud entre los vectores de los artistas usando la similitud del coseno
from sklearn.neighbors import NearestNeighbors

model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model_knn.fit(wide_artist_data_sparse)

#hacer recomendaciones
query_index = np.random.choice(wide_artist_data.shape[0])
distances, indices = model_knn.kneighbors(wide_artist_data.iloc[query_index, :].values.reshape(1, -1), n_neighbors = 6)

for i in range(0, len(distances.flatten())):
    if i == 0:
        print ('Recommendations for {0}:\n'.format(wide_artist_data.index[query_index]))
    else:
        print ('{0}: {1}, with distance of {2}:'.format(i, wide_artist_data.index[indices.flatten()[i]], distances.flatten()[i]))

#datos de repreducciones en forma binaria
#en el anterior ejemplo se utilizan las reproducciones reales como 
#valores en los vectores de artistas

# otra opción es convertir cada vector en un binario (1 o 0, ha escuchado la canción o no) 
# con numpy se haría aplicando la función sign a cada columna del dataframe

wide_artist_data_zero_one = wide_artist_data.apply(np.sign)
wide_artist_data_zero_one_sparse = csr_matrix(wide_artist_data_zero_one.values)

#save_sparse_csr('./lastfm_sparse_artist_matrix_binary.npz', wide_artist_data_zero_one_sparse)
model_nn_binary = NearestNeighbors(metric='cosine', algorithm='brute')
model_nn_binary.fit(wide_artist_data_zero_one_sparse)

#comparativa entre ambas

distances, indices = model_nn_binary.kneighbors(wide_artist_data_zero_one.iloc[query_index, :].values.reshape(1, -1), n_neighbors = 6)

for i in range(0, len(distances.flatten())):
    if i == 0:
        print('Recommendations with binary play data for {0}:\n'.format(wide_artist_data_zero_one.index[query_index])) 
    else:
        print('{0}: {1}, with distance of {2}:'.format(i, wide_artist_data_zero_one.index[indices.flatten()[i]], distances.flatten()[i]))

# recomendador con fuzzy matching
# para cubrir búsquedas con nombres ambiguos o con typos
from fuzzywuzzy import fuzz

def print_artist_recommendations(query_artist, artist_plays_matrix, knn_model, k):
    """
    Inputs:
    query_artist: query artist name
    artist_plays_matrix: artist play count dataframe (not the sparse one, the pandas dataframe)
    knn_model: our previously fitted sklearn knn model
    k: the number of nearest neighbors.
    
    Prints: Artist recommendations for the query artist
    Returns: None
    """
    query_index = None
    ratio_tuples = []
    
    for i in artist_plays_matrix.index:
        ratio = fuzz.ratio(i.lower(), query_artist.lower())
        if ratio >= 75:
            current_query_index = artist_plays_matrix.index.tolist().index(i)
            ratio_tuples.append((i, ratio, current_query_index))
    
    print('Possible matches: {0}\n'.format([(x[0], x[1]) for x in ratio_tuples])) 
    
    try:
        query_index = max(ratio_tuples, key = lambda x: x[1])[2] # get the index of the best artist match in the data
    except:
        print('Your artist didn\'t match any artists in the data. Try again') 
        return None
    
    distances, indices = knn_model.kneighbors(artist_plays_matrix.iloc[query_index, :].values.reshape(1, -1), n_neighbors = k + 1)

    for i in range(0, len(distances.flatten())):
        if i == 0:
            print('Recommendations for {0}:\n'.format(artist_plays_matrix.index[query_index])) 
        else:
            print('{0}: {1}, with distance of {2}:'.format(i, artist_plays_matrix.index[indices.flatten()[i]], distances.flatten()[i])) 

    return None


#ejemplos

print_artist_recommendations('red hot chili peppers', wide_artist_data_zero_one, model_nn_binary, k = 10)
print_artist_recommendations('arctic monkeys', wide_artist_data_zero_one, model_nn_binary, k = 10)
print_artist_recommendations('u2', wide_artist_data_zero_one, model_nn_binary, k = 10)
print_artist_recommendations('dispatch', wide_artist_data_zero_one, model_nn_binary, k = 10)