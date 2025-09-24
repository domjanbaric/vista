import numpy as np
import json
import time
from joblib import Parallel, delayed
from joblib import parallel_backend
from tqdm import tqdm
# -- Optimized low-level operations ----------------------------------------

def vector_clearness_fast(v1: np.ndarray, v2: np.ndarray, n: int) -> float:
    """Compute cosine similarity of the top-n most "noisy" dimensions."""
    # Fast elementwise difference and alternative metric
    diff = np.abs(v1 - v2)
    alt  = 2.0 - diff
    d    = np.maximum(diff, alt)
    # Select top-n dimensions without full sort
    idx  = np.argpartition(d, -n)[-n:]
    v1n, v2n = v1[idx], v2[idx]
    # Compute cosine
    dot   = np.dot(v1n, v2n)
    norm1 = np.linalg.norm(v1n)
    norm2 = np.linalg.norm(v2n)
    return dot / (norm1 * norm2)

# -- Similarity between groups ----------------------------------------------

def compare_groups(group1, group2, distance_matrix):
    # average pairwise distance (already in matrix)
    total = 0.0
    for i in group1:
        total += distance_matrix[i, group2].sum()
    return total / (len(group1) * len(group2))

# -- Compare two lists of vectors -------------------------------------------

def compare_lists(vecs1, vecs2, n=250):
    sims = []
    # flatten single vectors
    if not hasattr(vecs1[0], '__len__'):
        vecs1 = [vecs1]
    if not hasattr(vecs2[0], '__len__'):
        vecs2 = [vecs2]
    for v1 in vecs1:
        for v2 in vecs2:
            sims.append(vector_clearness_fast(v1, v2, n))
    return float(np.median(sims))

# -- High-level text comparison ---------------------------------------------

def text_comparison(a_ids, b_ids, vec_dict):
    total_score = 0.0
    count = 0
    for i in a_ids:
        for j in b_ids:
            di = vec_dict[i]
            dj = vec_dict[j]
            # Tweet vs tweet or mixed
            if di['is tweet'] and dj['is tweet']:
                sims = [compare_lists(di[f'tweet vector {o}'], dj[f'tweet vector {o}']) for o in range(1,5)]
                score = sum(sims) / 4.0
            elif di['is tweet']:
                sims = [compare_lists(di[f'tweet vector {o}'], dj['tweet vector']) for o in range(1,5)]
                score = sum(sims) / 4.0
            elif dj['is tweet']:
                sims = [compare_lists(di['tweet vector'], dj[f'tweet vector {o}']) for o in range(1,5)]
                score = sum(sims) / 4.0
            else:
                # summary stop and noun vectors
                stop_min = min(len(di['summary stop vector']), len(dj['summary stop vector']))
                noun_min = min(len(di['summary noun vector']), len(dj['summary noun vector']))
                stop_sims = [compare_lists(di['summary stop vector'][k], dj['summary stop vector'][k]) for k in range(stop_min)]
                noun_sims = [compare_lists(di['summary noun vector'][k], dj['summary noun vector'][k]) for k in range(noun_min)]
                stop_score = sum(stop_sims) / 4.0
                noun_score = sum(noun_sims) / 4.0
                text_score = compare_lists(di['text vector'], dj['text vector'])
                score = (1.8 * text_score + 0.4 * stop_score + 0.8 * noun_score) / 3.0
            total_score += score
            count += 1
    return total_score / count

# -- Parallel distance matrix construction ---------------------------------

def make_distance_matrix_parallel(vec_dict, n_jobs=-1, batch_size=1):
    """
    Build the full N×N distance matrix over `vec_dict` using a threading backend
    to avoid per‐process memory duplication.
    """
    keys = sorted(vec_dict.keys())
    size = len(keys)
    dist = np.zeros((size, size), dtype=float)

    def work(pair):
        i, j = pair
        id_i, id_j = keys[i], keys[j]
        val = text_comparison([id_i], [id_j], vec_dict)
        return i, j, val

    # prepare all upper‐triangle index pairs (i ≤ j)
    pairs = [(i, j) for i in range(size) for j in range(i, size)]
    total = len(pairs)

    # run on threads (NumPy releases the GIL) with small batches
    with parallel_backend('threading'):
        results = Parallel(n_jobs=n_jobs, batch_size=batch_size)(
            delayed(work)(p) for p in tqdm(pairs, desc="Building dist matrix", total=total)
        )

    # fill both halves of the symmetric matrix
    for i, j, v in results:
        dist[i, j] = v
        dist[j, i] = v

    return dist, keys

# -- Hierarchical clustering (unchanged) ------------------------------------

def connect_texts(keys, dist, threshold=0.7):
    curr = [[k] for k in keys]
    iterations = [list(curr)]
    last_len = -1
    while len(curr) != last_len:
        last_len = len(curr)
        connections, both = [], []
        for grp in curr:
            sims = []
            for other in curr:
                if other is grp: continue
                ixs = [keys.index(x) for x in grp]
                jxs = [keys.index(x) for x in other]
                sims.append((compare_groups(ixs, jxs, dist), other))
            try:
                best, closest = max(sims, key=lambda x: x[0])
            except:
                best=-10
            if best < threshold:
                connections.append(list(grp))
            else:
                merged = sorted(grp + closest)
                if merged not in connections:
                    connections.append(merged)
                else:
                    both.append(merged)
        next_iter = both.copy()
        for g in curr:
            if not any(set(g) <= set(h) for h in next_iter):
                next_iter.append(g)
        curr = next_iter
        iterations.append(list(curr))
    return iterations[:-1]

# -- IO and orchestration ---------------------------------------------------

def input_texts(file_path):
    with open(file_path) as f:
        data = json.load(f)
    vdict, tdict = {}, {}
    for item in data:
        idx = int(item['id'])
        tdict[idx] = item['text']
        if not item['is tweet']:
            d = {
                'text vector'           : np.array(item['text vector']),
                'summary stop vector'   : [np.array(v) for v in item['summary stop vector']],
                'summary noun vector'   : [np.array(v) for v in item['summary noun vector']],
                'is tweet'              : item.get('is tweet', False)
            }
        else:
            d={
                'is tweet': item.get('is tweet', True)
               }
        if d['is tweet']:
            for o in range(1,5):
                d[f'tweet vector {o}'] = np.array(item[f'tweet vector {o}'])
        else:
            d['tweet vector'] = np.array(item.get('tweet vector', np.zeros_like(d['text vector'])))
        vdict[idx] = d
    return vdict, tdict

def making_base(iterations_list):  #Out of iteration_list returns list with elements that connected together, out of that list we make a forest
    not_connected = []
    trees_base = []
    is_last_iteration = 1
    for i in reversed(iterations_list):
        for j in i:
            if is_last_iteration == 1 and len(j) == 1:
                not_connected.append(j[0])
            elif is_last_iteration == 1:
                trees_base.append([j])
            elif (j[0] not in not_connected):
                for q in trees_base:
                    if (set(j) & set(q[0]) and not any(j == sublist for sublist in q)):
                        q.append(j)
        is_last_iteration = 0
    return trees_base,not_connected


def cluster(input_file: str, output_file: str):
    vec_dict, text_dict = input_texts(input_file)
    print("Vectorised texts:", len(vec_dict))
    dist_matrix, keys = make_distance_matrix_parallel(vec_dict)
    print("Distance matrix done.")
    iterations = connect_texts(keys, dist_matrix)
    print("Clustering done.")
    trees_base, not_connected = making_base(iterations)
    base_dict = {'trees_base': trees_base, 'not_connected': not_connected, 'text_dict': text_dict}
    with open(output_file, 'w') as f:
        json.dump(base_dict, f)
    return

# --------------------------------- CLI --------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="VISTA: build similarity matrix, link texts, and emit trees_base."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input JSON (see input_texts format).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to write output JSON (trees_base, not_connected, text_dict).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Merge threshold for linking (default: 0.7).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Joblib parallelism for distance matrix (-1 = all cores).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Joblib batch size (default: 1).",
    )

    args = parser.parse_args()
    cluster(
        input_file=args.input,
        output_file=args.output,
        threshold=args.threshold,
        n_jobs=args.n_jobs,
        batch_size=args.batch_size,
    )