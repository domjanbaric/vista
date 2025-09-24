import argparse

from openai import AsyncOpenAI
import numpy as np
from clustering import compare
from vectorisation import openai_response_async,stopwords,norm,get_embedding
import asyncio
import json
import time
import copy
import tqdm
client = AsyncOpenAI(api_key=API_KEY)







async def topic_list(text_list, mandatory_topics= None):
    """Ask GPT for ≤10 common topics (≤3 words each) covering *all* texts."""
    mandatory_topics = mandatory_topics or []

    total_length = sum(len(text) for text in text_list)

    if total_length / 4 > 120_000:
        # Truncate each text proportionally so total stays under 400,000
        max_total = 400_000
        text_in = [
            text[: max_total * len(text) // total_length] for text in text_list
        ]
    else:
        text_in=text_list
    n = min(len(text_list), 10)

    prompt = f"""
Give me no more than {n} common topics for all the text given in the list. If you think that some of these topics: {mandatory_topics} are common topics for all the text given in list be sure to include them.
The list is given below and is delimited by triple quotes. Don't explain why they are common and make those topics a maximum of 3 words.
Just return the topics, nothing else, separated by &.
'''{text_in}'''
"""
    response = await openai_response_async(prompt)
    return [item.strip() for item in response.split("&") if item.strip()]


# ------------- main score‑based aggregator -----------------------------------

async def generate_topics(nodes, text_dict, given_topics = None):
    """Return a list of candidate topics for the node that best agrees across samples."""
    given_topics = list(given_topics or [])
    # gather node texts, truncate early
    node_texts = [text_dict[str(i)] if str(i) in text_dict else text_dict[i] for i in nodes]

    # collect candidate lists in parallel
    candidate_lists = await asyncio.gather(*(topic_list(node_texts, given_topics) for _ in range(2)))

    unique_topics = {t for lst in candidate_lists for t in lst}
    unique_topics = list(unique_topics)  # keep order
    vectors = await asyncio.gather(*(get_embedding(t) for t in unique_topics))
    topic_vectors = dict(zip(unique_topics, vectors))

    def embed(lst):
        return [topic_vectors[t] for t in lst]

    best_score = float("-inf")
    best_topics= None

    vecs_cache= {}
    for i, lst in enumerate(candidate_lists):
        vecs_i = vecs_cache.setdefault(i, embed(lst))
        score = 0.0
        for j in range(len(candidate_lists)):
            if i == j:
                continue
            vecs_j = vecs_cache.setdefault(j, embed(candidate_lists[j]))
            score += compare(vecs_i, vecs_j, 250)
        if score > best_score:
            best_score, best_topics = score, lst

    best_topics = list(set(best_topics or []) | set(given_topics))
    return best_topics

async def topic_with_child(child_id, parent_id, tree_base, text_dict, stored_topics):
    child_vecs = norm([await get_embedding(t) for t in stored_topics[child_id]])

    # Generate new topics using child's existing set as mandatory
    topics = await generate_topics(tree_base[parent_id], text_dict, stored_topics[child_id])

    # Separate those already present in child vs. brand‑new
    inherited = [t for t in topics if t in stored_topics[child_id]]
    fresh = [t for t in topics if t not in inherited]

    # If there are no fresh topics we just keep the first inherited one
    if not fresh and inherited:
        stored_topics[parent_id] = [inherited[0]]
        return stored_topics

    try:
        fresh_vecs = norm([await get_embedding(t) for t in fresh])
    except Exception:
        fresh_vecs = []  # fallback – if embedding fails skip similarity check

    best_pair: list[str] = []
    best_score = float("-inf")

    for i in range(len(fresh_vecs)):
        for j in range(i + 1, len(fresh_vecs)):
            score = compare(child_vecs, [fresh_vecs[i], fresh_vecs[j]], 250)
            if score > best_score:
                best_score, best_pair = score, [fresh[i], fresh[j]]

    chosen: list[str] = [inherited[0]] if inherited else []
    for t in best_pair:
        if chosen:  # if we already chose one topic, keep only it (rule from original code)
            break
        chosen.append(t)

    if not chosen:  # fallback – take first fresh topic
        chosen = [fresh[0]]

    stored_topics[parent_id] = chosen
    return stored_topics


# -----------------------------------------------------------------------------
# Tree / forest orchestrators
# -----------------------------------------------------------------------------

async def topics_to_tree(tree_base, text_dict,forest):
    try:
        """Assign topics bottom‑up within a *single* tree (in place)."""
        stored_topics = [
            elem[-1] if isinstance(elem, list) and elem and isinstance(elem[-1], list) else []
            for elem in tree_base
        ]
        with open('./greska_save.json','r') as f:
            stored_topics = json.load(f)['stored_topics']
        for idx in tqdm.tqdm(range(len(tree_base) - 1, -1, -1)):
            node = tree_base[idx]
            if stored_topics[idx] != []:
                continue
            if len(node) == 2:
                top = await generate_topics(node, text_dict)
                stored_topics[idx] = [top[0]]
                continue

            if len(node) <= 2:
                continue  # leaf

            # find immediate children
            child1_id = child2_id = None
            for j in range(idx + 1, len(tree_base)):
                for k in range(j + 1, len(tree_base)):
                    if sorted(tree_base[j] + tree_base[k]) == node:
                        child1_id, child2_id = j, k
                        break
                if child1_id is not None:
                    break

            if child1_id is None or child2_id is None:
                raise RuntimeError("Could not locate children for node", node)

            # If any child is a singleton, we can reuse the specialised helper
            if len(tree_base[child1_id]) == 1:
                stored_topics = await topic_with_child(child2_id, idx, tree_base, text_dict, stored_topics)
                continue
            if len(tree_base[child2_id]) == 1:
                stored_topics = await topic_with_child(child1_id, idx, tree_base, text_dict, stored_topics)
                continue

            # General case – both children are larger groups
            child1_vecs = norm([await get_embedding(t) for t in stored_topics[child1_id]])
            child2_vecs = norm([await get_embedding(t) for t in stored_topics[child2_id]])

            topics = await generate_topics(node, text_dict, stored_topics[child1_id] + stored_topics[child2_id])

            inherited = [t for t in topics if t in stored_topics[child1_id] or t in stored_topics[child2_id]]
            fresh = [t for t in topics if t not in inherited]

            try:
                fresh_vecs = norm([await get_embedding(t) for t in fresh])
            except Exception:
                fresh_vecs = []

            best_pair: list[str] = []
            best_score = float("-inf")

            for i in range(len(fresh_vecs)):
                for j in range(i + 1, len(fresh_vecs)):
                    score = compare(child1_vecs, [fresh_vecs[i], fresh_vecs[j]], 250) + compare(child2_vecs, [fresh_vecs[i], fresh_vecs[j]], 250)
                    if score > best_score:
                        best_score, best_pair = score, [fresh[i], fresh[j]]

            chosen: list[str] = [inherited[0]] if inherited else []
            for t in best_pair:
                if chosen:
                    break
                chosen.append(t)
            if not chosen:
                chosen = [fresh[0]]

            stored_topics[idx] = chosen

        # finally attach topics to *non‑leaf* nodes
        for idx, node in enumerate(tree_base):
            if len(node) != 1:  # non‑leaf
                node.append(stored_topics[idx])
        return tree_base
    except BaseException as e:
        print(e)
        dict_save={'forest':forest,'current':tree_base,'stored_topics':stored_topics}
        with open('./greska_save.json', 'w') as f:
            f.write(json.dumps(dict_save))
        return tree_base



async def topics_to_forest(forest, text_dict):
    """Sequentially process a list of trees. Wrap calls with semaphore if you want parallelism."""
    result = []
    for i in range(len(forest)):
        forest[i]=await topics_to_tree(forest[i], text_dict,forest)
    return forest


def input(path):
    with open(path, 'r') as f:
        data = json.load(f)
    trees_base=data['trees_base']
    text_dict=data['text_dict']
    return trees_base, text_dict

async def same_topics(topics):
    prompt=f'''
    You will be given a list of topics for newspapers, your task is to find topics in that list that mean the same thing or a very similar thing.
    Your output must be if you determine that topic1 an topic2 are very similar output topic1==topic2, output all the pairs that you would say are very similar
    and between pairs put &&&. Dont write anything else in output except pairs of very similar topics. If there are more than two topics that mean very similar put the all like this: topic1==topic2==...==topicn.
    If there are no similar topics output NO. 
    list of topics:{topics}
    '''
    x= await openai_response_async(prompt)
    return x

def split_evenly(lst: list[str], n_chunks: int) -> list[list[str]]:
    """Split lst into <= n_chunks chunks with sizes that differ by ≤1."""
    n_chunks = min(n_chunks, len(lst))          # never produce empty chunks
    k, m = divmod(len(lst), n_chunks)           # k = base size, m = long chunks
    chunks, start = [], 0
    for i in range(n_chunks):
        end = start + k + (1 if i < m else 0)
        chunks.append(lst[start:end])
        start = end
    return chunks

def groups_from_llm(raw: str) -> list[list[str]]:
    """Parse  'a==b==c &&& x==y'  →  [['a','b','c'], ['x','y']] ."""
    if raw==None:
        return []
    raw = raw.strip()
    if not raw or raw.upper() == "NO":
        return []
    return [[t.strip() for t in grp.split("==") if t.strip()]
            for grp in raw.split("&&&")]
# ---------- main function -------------------------------------------

async def filter_tree(tree):
    # 1) Collect *unique* leaf-level topics
    print(tree)
    remaining = {t for node in tree if len(node) > 1 for t in node[-1]}
    remaining = list(remaining)
    print('topica',remaining)# deterministic order is helpful

    conversions = {}        # original(lower) -> canonical (original spelling)

    # 2) Iterate over decreasing chunk counts
    for n_chunks in [100,80,70,50,30,20,10, 6,3,2,1]:
        print(n_chunks)
        if len(remaining) <= 1:
            break

        chunks = split_evenly(remaining, n_chunks)
        # skip singleton chunks – they cannot yield duplicates
        tasks  = [same_topics(chunk) for chunk in chunks if len(chunk) > 1]
        raw_replies = await asyncio.gather(*tasks)
        print(raw_replies)
        # 3) Extract groups and update the conversion map
        for reply in raw_replies:
            for group in groups_from_llm(reply):
                # choose a canonical string for the whole group
                # ─ heuristics:  (a) reuse earlier canonical if any member is known
                canonical = next((conversions[t.lower()]
                                  for t in group if t.lower() in conversions),
                                 group[-1])    # (b) otherwise use the last token
                # register every synonym -> canonical
                for t in group:
                    conversions[t.lower()] = canonical

        # drop every topic that is now mapped to something else
        remaining = [t for t in remaining if t.lower() not in conversions]

    # 4) Rewrite the tree with canonical names
    for node in tree:
        if len(node) > 1:
            node[-1] = list({conversions.get(t.lower(), t) for t in node[-1]})

    # 5) return both the cleaned tree AND the “what became what” map
    return tree, conversions

async def filter_forest(forest):
    results  = await asyncio.gather(*[filter_tree(t) for t in forest])
    new_forest = [r[0] for r in results]           # cleaned trees
    mapping     = {}
    for _, conv in results:                       # merge all conversions
        mapping.update(conv)
    return new_forest, mapping


async def topic_element(element,list):
    dict = {}
    for i in list:
        if element in i and len(i) > 1:
            for j in i[-1]:
                if j not in dict:
                    dict[j] = 2 ** (1 / len(i))
                else:
                    dict[j] += 2 ** (1 / len(i))
    sorted_items = sorted(dict.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_items) > 2:
        return sorted_items[2][0]
    if len(sorted_items) < 2:
        return sorted_items[0][0]
    return sorted_items[1][0]

async def topic_tree(tree):
    keys = tree[0][:-1]  # all except the last element
    results = await asyncio.gather(*[topic_element(key, tree) for key in keys])
    return dict(zip(keys, results))


async def topic_forest(forest):
    dicts = await asyncio.gather(*[topic_tree(tree) for tree in forest])
    merged = {}
    for d in dicts:
        merged.update(d)
    return merged


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run VISTA topic assignment and filtering pipeline."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="./cluster_bbc.json",
        help="Path to input JSON file (default: ./cluster_bbc.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./topic_nase_2.json",
        help="Path to output JSON file (default: ./topic_nase_2.json)",
    )

    args = parser.parse_args()
    asyncio.run(main(input_path=args.input, output_path=args.output))