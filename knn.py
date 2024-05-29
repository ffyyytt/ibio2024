import pickle
import hashlib

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans

(q_feature, q_id), (g_feature, g_id) = pickle.load(open('data', 'rb'))
kmeans = KMeans(n_clusters=1000, random_state=0, n_init="auto").fit(g_feature)

g_label = kmeans.predict(g_feature)
q_label = kmeans.predict(q_feature)

g_label = [bin(int(hashlib.md5(str(i).encode()).hexdigest(), 16))[16:16+48] for i in g_label]
q_label = [bin(int(hashlib.md5(str(i).encode()).hexdigest(), 16))[16:16+48] for i in q_label]

dfg = pd.DataFrame()
dfg["image_id"] = [i+".jpg" for i in g_id]
dfg["hashcode"] = g_label

dfg.to_csv("G.csv", index = False)

dfq = pd.DataFrame()
dfq["image_id"] = [i+".jpg" for i in q_id]
dfq["hashcode"] = q_label

dfq.to_csv("Q.csv", index = False)