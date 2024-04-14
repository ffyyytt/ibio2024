import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--query', type=str, default='submit_query.csv', help='48-bit hash code file for query set.')
parser.add_argument('--gallery', type=str, default='submit_gallery.csv', help='48-bit hash code file for gallery set.')
parser.add_argument('--submit', type=str, default='submit.csv', help='Final submit csv file.')
parser.add_argument('--k', type=int, default=100, help='Topk == 100.')
args = parser.parse_args()

k = args.k
query_code_path = args.query
gallery_code_path = args.gallery
submit_path = args.submit

query_code = []
query_images = []
with open(query_code_path, 'r') as f:
    f.readline()
    for line in f:
        image, code = line[:-1].split(',')
        code = [-1. if i == 0 else 1. for i in list(map(float, list(code[1:-1])))]
        query_code.append(code)
        query_images.append(image)
query_code = np.array(query_code)

gallery_code = []
gallery_images = []
with open(gallery_code_path, 'r') as f:
    f.readline()
    for line in f:
        image, code = line[:-1].split(',')
        code = [-1. if i == 0 else 1. for i in list(map(float, list(code[1:-1])))]
        gallery_code.append(code)
        gallery_images.append(image[:-4])
gallery_code = np.array(gallery_code)

with open(submit_path, 'w') as f:
    f.write('Id,Predicted\n')
    for i, q in enumerate(query_code):
        hamming_dist = q.shape[0] - np.dot(q, gallery_code.T)
        index = np.argsort(hamming_dist)[:k]
        retrieval_images = [gallery_images[j] for j in index]
        f.write(query_images[i] + ',')
        f.write(' '.join(retrieval_images) + '\n')
        print(f'\r writing {i + 1}/{query_code.shape[0]}', end='')