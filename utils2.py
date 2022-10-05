import numba as nb
import numpy as np
import os
import time
import multiprocessing
import tensorflow as tf
from collections import defaultdict
import numpy as np
import torch
import math


def get_feature_matrix(filename, i: int, shift_id=0, tf_idf=False, time_id=None):
    count = 0
    time_dict = dict()
    TS = 1000000
    num_triple = 0
    time_set = set()
    entity_set = set()
    th = defaultdict(lambda: defaultdict(int))
    t_e = defaultdict(set)
    e_t = defaultdict(int)
    for line in open(filename + 'triples_' + str(i), 'r'):
        num_triple += 1
        words = line.split()
        if len(words) == 4:
            head, r, tail, t = [int(item) for item in words]
            t_encode = 1
        else:
            head, r, tail, t1, t2 = [int(item) for item in words]
            t_encode = t1 * 1000 + t2
            t = time_id[t_encode]
        time_set.add(t)
        head, tail = head - shift_id, tail - shift_id
        entity_set.add(head)
        entity_set.add(tail)
        if t_encode > 0:
            th[head][t] += 1
            th[tail][t + TS] += 1
            t_e[t].add(head)
            t_e[t + TS].add(tail)
            e_t[head] += 1
            e_t[tail] += 1

    index, value = [], []
    # num_ent = len(entity_set)
    from wl_test import getLast
    num_ent = getLast(filename + 'ent_ids_' + str(i))
    if i == 2:
        num_ent -= shift_id

    if time_id is not None:
        num_time = len(time_id.keys())
    else:
        num_time = len(time_set)
    for ent, dic in th.items():
        for time, cnt in dic.items():
            t = time if time < TS else time + num_time - TS  # different id for head and tail
            index.append((ent, t))
            if tf_idf:
                tf = cnt / e_t[ent]
                idf = math.log(num_ent / (len(t_e[time]) + 1))
                value.append(tf * idf)
            else:
                value.append(cnt)

    index = torch.LongTensor(index)
    print(num_ent, num_time)
    matrix = torch.sparse_coo_tensor(torch.transpose(index, 0, 1), torch.Tensor(value),
                                     (num_ent, 2 * num_time))
    return matrix


def get_link(filename, shift_id=0):
    links = []
    for line in open(filename + 'sup_pairs', 'r'):
        e1, e2 = line.split()
        links.append((int(e1), int(e2) - shift_id))
    for line in open(filename + 'ref_pairs', 'r'):
        e1, e2 = line.split()
        links.append((int(e1), int(e2) - shift_id))
    return links

def get_time(filename):
    time_dict = dict()
    count = 0
    for i in [0, 1]:
        for line in open(filename + 'triples_' + str(i + 1), 'r'):
            words = line.split()
            head, r, tail, t1, t2 = [int(item) for item in words]
            t = t1 * 1000 + t2
            if t not in time_dict.keys():
                time_dict[t] = count
                count += 1
    return time_dict


def load_triples(file_path, reverse=True):
    @nb.njit
    def reverse_triples(triples):
        reversed_triples = np.zeros_like(triples)
        for i in range(len(triples)):
            reversed_triples[i,0] = triples[i,2]
            reversed_triples[i,2] = triples[i,0]
            if reverse:
                reversed_triples[i, 1] = triples[i, 1] + rel_size
            else:
                reversed_triples[i, 1] = triples[i, 1]
        return reversed_triples
    
    with open(file_path + "triples_1") as f:
        triples1 = f.readlines()
        
    with open(file_path + "triples_2") as f:
        triples2 = f.readlines()
        
    triples = np.array([line.replace("\n","").split("\t")[0:3] for line in triples1 + triples2]).astype(np.int64)
    node_size = max([np.max(triples[:, 0]), np.max(triples[:,2])]) + 1
    rel_size = np.max(triples[:, 1]) + 1
    
    all_triples = np.concatenate([triples, reverse_triples(triples)], axis=0)
    all_triples = np.unique(all_triples, axis=0)
    
    return all_triples, node_size, rel_size*2 if reverse else rel_size

def load_aligned_pair(file_path,ratio = 0.3):
    if "sup_ent_ids" not in os.listdir(file_path):
        with open(file_path + "ref_ent_ids") as f:
            aligned = f.readlines()
    else:
        with open(file_path + "ref_ent_ids") as f:
            ref = f.readlines()
        with open(file_path + "sup_ent_ids") as f:
            sup = f.readlines()
        aligned = ref + sup
        
    aligned = np.array([line.replace("\n", "").split("\t") for line in aligned]).astype(np.int64)
    np.random.shuffle(aligned)
    return aligned[:int(len(aligned) * ratio)], aligned[int(len(aligned) * ratio):]

def test(sims,mode = "sinkhorn", batch_size = 1024):
    if mode == "sinkhorn":
        results = []
        for epoch in range(len(sims) // batch_size + 1):
            sim = sims[epoch*batch_size:(epoch+1)*batch_size]
            rank = tf.argsort(-sim, axis=-1)
            ans_rank = np.array([i for i in range(epoch * batch_size, min((epoch+1) * batch_size, len(sims)))])
            x = np.expand_dims(ans_rank, axis=1)
            y = tf.tile(x, [1, len(sims)])
            results.append(tf.where(tf.equal(tf.cast(rank, ans_rank.dtype), tf.tile(np.expand_dims(ans_rank, axis=1), [1, len(sims)]))).numpy())
        results = np.concatenate(results, axis=0)
        
        @nb.jit(nopython=True)
        def cal(results):
            hits1, hits10, mrr = 0, 0, 0
            for x in results[:, 1]:
                if x < 1:
                    hits1 += 1
                if x < 10:
                    hits10 += 1
                mrr += 1/(x + 1)
            return hits1, hits10, mrr
        hits1, hits10, mrr = cal(results)
        print("hits@1 : %.2f%% hits@10 : %.2f%% MRR : %.2f%%" % (hits1/len(sims)*100, hits10/len(sims)*100, mrr/len(sims)*100))
        return hits1/len(sims), hits10/len(sims), mrr/len(sims)
    else:
        c = 0
        for i, j in enumerate(sims[1]):
            if i == j:
                c += 1
        print("hits@1 : %.2f%%" %(100 * c/len(sims[0])))
        return c/len(sims[0])


def find_pairs(test_pair, sim1, sim2):
    rank = tf.argmax(sim1, axis=-1)
    left_id = test_pair[:, 0]
    right_id = test_pair[:, 1]
    cnt = 0
    link_1, link_2 = set(), set()
    for x in range(len(rank)):
        link_1.add((left_id[x], right_id[rank[x]]))
    rank2 = tf.argmax(sim2, axis=-1)
    for x in range(len(rank2)):
        link_2.add((left_id[rank2[x]], right_id[x]))
    overall = link_1.intersection(link_2)
    print(len(overall))
    test_set = set(zip(left_id, right_id))

    print(len(overall.intersection(test_set)))
    y = len(overall.intersection(test_set))
    print(y/len(overall))
    links = list(overall)

    with open('unsup_link', 'w') as f:
        for link in links:
            f.write(str(link[0]) + '\t' + str(link[1]) + '\n')
    return links