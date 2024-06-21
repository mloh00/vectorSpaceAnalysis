import os

from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
from chromadb.config import Settings

import json
import matplotlib
import matplotlib.collections
import matplotlib.pyplot as plt
import mplcursors
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import textwrap
import InstructorEmbedding
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import pdist, squareform

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"

embeddings = None
the_table = None
#----- Function for getting embedded query vectors -----#
def getQueryVector(query):
    global embeddings
    if not embeddings:
        embeddings = HuggingFaceInstructEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            embed_instruction="Represent the document for retrieval:",
            query_instruction="Represent the question for retrieving supporting documents:",
        )

    query_vector = embeddings.embed_query(query)
    return query, query_vector

#----- Function for getting similiarities and distances between vectors -----#
def getVectorSimiliarities(query_vectors):
    similiarities = pdist(query_vectors, 'cosine')
    
    return squareform(similiarities) 

def getVectorDistances(query_vectors):
    distances = pdist(query_vectors, 'euclidean')
    
    return squareform(distances) 

def getProjectedVectorSimiliarities(projected_query_vectors):
    similiarities = pdist(projected_query_vectors, 'cosine')

    return squareform(similiarities) 

def getProjectedVectorDistances(projected_query_vectors):   
    distances = pdist(projected_query_vectors, 'euclidean')

    return squareform(distances) 

#----- Function for creating the vector space -----#
def createVectorSpace(path):
    db = Chroma(persist_directory=path,
                client_settings=Settings(anonymized_telemetry=False, is_persistent=True))

    data = db.get(include=['embeddings', 'documents', 'metadatas'], where={'lang': 'en'})
    links = []
    for k in range(len(data['metadatas'])):
        if 'link' in data['metadatas'][k]:
            links.append(data['metadatas'][k]['link'][35:])

    # Get only docs with lang=en in the metadata
    # data = db.get(include=['embeddings', 'documents', 'metadatas'], where={'metadatas.lang': 'en'})

    return data, links

def displayVectorSpace(data, query_pairs, links):
    queries = []
    query_vectors = []
    for query_pair in query_pairs:
        queries.append(query_pair[0][0])
        query_vectors.append(query_pair[0][1])

    vectors = np.array(data["embeddings"])
    vectors = np.append(vectors, query_vectors, axis=0)
    texts = data["documents"] + queries
    links = links + queries
    # origins = data["metadatas"]

    if 1:
        tsne = TSNE(n_components=2, random_state=0)
        projected = tsne.fit_transform(vectors)
    else:
        projected = vectors[:, :2]

    print('Finished projecting...')
    # projected, query_vector = projected[:-1], projected[-1]

    if 0:
        colors = KMeans(n_clusters=5, random_state=0).fit_predict(projected)
    else:
        colors = np.zeros(len(projected))
        colors[-len(queries):] = 1

    fig, ax = plt.subplots()
    print(len(projected))
    print(len(links))
    print(len(vectors))

    #----- Vector similiarities and distances -----#
    distances = getVectorDistances(query_vectors)
    similiarities = getVectorSimiliarities(query_vectors)
    projected_query_vectors = projected[-len(queries):]
    projected_similiarities = getProjectedVectorSimiliarities(projected_query_vectors)
    lines_similiarities_distances = {}
    projected_distances = getProjectedVectorDistances(projected_query_vectors)

    #----- Draw lines between query vectors -----#
    QUERY_LINES = False
    if QUERY_LINES:
        for i in range(len(query_vectors)):
            for j in range(i+1, len(query_vectors)):
                line, = plt.plot([projected_query_vectors[i][0], projected_query_vectors[j][0]], [projected_query_vectors[i][1], projected_query_vectors[j][1]], 'k--', alpha=0.5)
                lines_similiarities_distances[line] = (similiarities[i][j], projected_similiarities[i][j], distances[i][j], projected_distances[i][j], queries[i], queries[j])
        
        def update_annotation(sel):
            if isinstance(sel.artist, matplotlib.lines.Line2D):
                line = sel.artist
                real_similiarity, projected_similiarity, real_distance, projected_distance, query_i, query_j = lines_similiarities_distances[line]

                sel.annotation.set_text(f'Queries: {query_i}, {query_j}\nProjected distance: {projected_distance:.2f}\nProjected similiarity: {projected_similiarity:.2f}\nReal distance: {real_distance:.2f}\nReal similiarity: {real_similiarity:.2f}')

        cursor.connect("add", update_annotation)
    #----- Draw projected points -----#
    sizes = [100 if i > len(projected) - len(queries) else 30 for i in range(len(projected))]
    scatter = ax.scatter(projected[:, 0], projected[:, 1], c=colors, alpha=0.5, s=sizes)
    cursor = mplcursors.cursor(scatter)

    #----- Display Properties of projected and real points -----#
    points = []
    textbox = ax.text(0.01, 0.01, '', transform=ax.transAxes)
    textbox2 = ax.text(0.01, 0.960, '', transform=ax.transAxes)

    def on_select(sel):
        global the_table
        if isinstance(sel.artist, matplotlib.collections.PathCollection):
            sel.annotation.set_text(textwrap.fill(texts[sel.target.index], 60))
            #----- Find nearby projected points within radius -----#
            selected_point = sel.target
            radius_p = 1
            nearby_projected_points = []
            similiarites_ = []

            for j, point in enumerate(projected):
                if selected_point[0] == point[0] and selected_point[1] == point[1]:
                    selected_real_point = vectors[j]

            for i, point in enumerate(projected):
                distance_p = round(np.sqrt((point[0] - selected_point[0])**2 + (point[1] - selected_point[1])**2), 4)
                if distance_p < radius_p:
                    nearby_projected_points.append((i, distance_p))
                    for r, point_ in enumerate(projected):
                        if point[0] == point_[0] and point[1] == point_[1]:
                            real_point = vectors[r]                    
                    similiarites_.append((links[i], round(pdist([selected_real_point, real_point], 'cosine')[0], 2)))

            #----- Find nearby real points within radius -----#
            radius_r = 0.001
            nearby_real_points = []
            similiarities_of_nearby_real_points = []
            real_point = None
            for j, point in enumerate(projected):
                if selected_point[0] == point[0] and selected_point[1] == point[1]:
                    real_point = vectors[j]

            for i, point in enumerate(vectors):
                distance_r = round(np.sqrt((point[0] - real_point[0])**2 + (point[1] - real_point[1])**2), 4)
                similiarities_r = round(pdist([point, real_point], 'cosine')[0], 2)
                if distance_r < radius_r:
                    nearby_real_points.append((i, distance_r))
                    similiarities_of_nearby_real_points.append((links[i], similiarities_r))
            
            sorted_nearby_projected_points = sorted(nearby_projected_points, key=lambda x: x[1], reverse=False)
            sorted_nearby_real_points = sorted(nearby_real_points, key=lambda x: x[1], reverse=False)
            similiarities_of_nearby_real_points = sorted(similiarities_of_nearby_real_points, key=lambda x: x[1], reverse=False)
            textbox2.set_text(f"Nearby projected points (radius={radius_p}): {sorted_nearby_projected_points}\nNearby real points (radius={radius_r}): {sorted_nearby_real_points}")

            #----- Similiarity and distance between selected points -----#
            points.append(sel.target)
            if len(points) == 2:
                distance_matrix = getProjectedVectorDistances(points)
                similiarty_matrix = getProjectedVectorSimiliarities(points)
                similiarity = similiarty_matrix[0][1]
                distance = distance_matrix[0][1]
                projected_list = np.ndarray.tolist(projected)

                for i in range(len(projected)):
                    if points[0][0] == projected_list[i][0] and points[0][1] == projected_list[i][1]:
                        real_point1 = vectors[i]
                    if points[1][0] == projected_list[i][0] and points[1][1] == projected_list[i][1]:
                        real_point2 = vectors[i]

                real_points = [real_point1, real_point2]
                real_distance_matrix = getVectorDistances(real_points)
                real_similiarty_matrix = getVectorSimiliarities(real_points)

                textbox.set_text(f"Projected distance: {distance:.2f}\nProjected similiarity: {similiarity:.2f}\nReal distance: {real_distance_matrix[0][1]:.2f}\nReal similiarity: {real_similiarty_matrix[0][1]:.2f}")

                points.clear()

            #----- Display nearby projected points in a table -----#
            if the_table is not None:
                the_table.remove()

            col_labels = ['Doc of nearby points', 'Real similiarity of points']
            table_vals = sorted(similiarites_, key=lambda x: x[1], reverse=False)

            the_table = plt.table(cellText=table_vals,
                              colLabels=col_labels,
                              loc='lower right')
            the_table.auto_set_font_size(False)
            the_table.set_fontsize(10)
            the_table.scale(0.40, 1)

            plt.draw()

    cursor.connect("add", on_select)


    plt.show()

if __name__ == '__main__':
    
    with open("queries.json", "r", encoding="utf-8") as f:
        chunkQueries = json.load(f)

    query_pairs = []
    for chunkq in chunkQueries:
        if chunkq["lang"] == "en":
            query_pairs.append([getQueryVector(chunkq["text"])])

    data, links = createVectorSpace("DB")

    displayVectorSpace(data, query_pairs, links)
