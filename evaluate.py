#!/usr/bin/env python3
#
# Copyright (c) 2017-present, All rights reserved.
# Written by Julien Tissier <30314448+tca19@users.noreply.github.com>
#
# This file is part of Dict2vec.
#
# Dict2vec is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Dict2vec is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License at the root of this repository for
# more details.
#
# You should have received a copy of the GNU General Public License
# along with Dict2vec.  If not, see <http://www.gnu.org/licenses/>.

import os
import sys
import math
import argparse
import numpy as np
import scipy.stats as st
import re
import pdb
from tqdm import tqdm

FILE_DIR = "data/eval/"
results = dict()
oov     = dict()

def getSubwords(word, n):
    #n==1 is a special case
    word = "<{}>".format(word)
    subwords = []
    if n > 1:
        for i in range((len(word)-n)+1):
            subwords.append(word[i:i+n])
    elif n==1:
        subwords.append(word[:2])
        for i in range(2, len(word)-n-1):
            subwords.append(word[i])
        subwords.append(word[-2:])
    return subwords

def tanimotoSim(v1, v2):
    """Return the Tanimoto similarity between v1 and v2 (numpy arrays)"""
    dotProd = np.dot(v1, v2)
    return dotProd / (np.linalg.norm(v1)**2 + np.linalg.norm(v2)**2 - dotProd)


def cosineSim(v1, v2):
    """Return the cosine similarity between v1 and v2 (numpy arrays)"""
    dotProd = np.dot(v1, v2)
    return dotProd / (np.linalg.norm(v1) * np.linalg.norm(v2))

def CosAdd(a1, a2, b1, b2):
    return cosineSim(b2, a2 - a1 + b1)

def CosMul(a1, a2, b1, b2):
    return (cosineSim(b2, b1)*cosineSim(b2, a2))/(cosineSim(b2, a1)+0.001)

def solveAnalogy(embedding, analogy, analogy_solution):
    a1, a2, b1, b2 = analogy
    sol_score_ca = CosAdd(a1, a2, b1, b2)
    sol_score_cm = CosMul(a1, a2, b1, b2)
    solution_ca = analogy_solution; solved_ca = False
    solution_cm = analogy_solution; solved_cm = False
    for word, vector in embedding.items():
        if not solved_ca:
            score_ca = CosAdd(a1, a2, b1, vector)
            if score_ca > sol_score_ca:
                solution_ca = word
                solved_ca = True
        if not solved_cm:
            score_cm = CosMul(a1, a2, b1, vector)
            if score_cm > sol_score_cm:
                solution_cm = word
                solved_cm = True
        if solved_ca and solved_cm:
            break
    return {'ca': solution_ca == analogy_solution, 'cm': solution_cm == analogy_solution}

def init_results():
    """Read the filename for each file in the evaluation directory"""
    for filename in os.listdir(FILE_DIR):
        if not filename in results:
            results[filename] = []

def create_embedding(filename):
    """Creates a dictionary with the vocabulary and vectors. Assumes .vec format"""
    embedding = {}
    with open(filename) as of:
        vocab_size, vector_size = [int(val) for val in of.readline().strip().split()]
        # for line in of:
        for _ in tqdm(range(vocab_size)):
            line = of.readline().strip().split()
            word, vector = line[0], np.array([float(val) for val in line[1:]])
            embedding[word] = vector
    return embedding

def evaluate(filename, args):
    if args.evaluation_task == 'similarity':
        evaluateSimilarity(filename, args)
    else:
        evaluateAnalogy(filename, args)
        # evaluateAnalogyPar(filename, args)

# def processAnalogy(embedding, analogy_words, args):
def processAnalogy(arguments):
    embedding, analogy_words, args = arguments
    stats = {'cos_add': 0, 'cos_mul': 0, 'found': 0, 'not_found': 0}
    wa1, wa2, wb1, wb2 = analogy_words
    if not args.subwords:
        if not (wa1 in embedding and wa2 in embedding and\
                wb1 in embedding and wb2 in embedding):
            stats['not_found'] += 1
            return stats
        else:
            stats['found'] += 1
            a1, a2, b1, b2 = embedding[wa1], embedding[wa2], embedding[wb1], embedding[wb2]
    else:
        if not args.fasttext or (args.fasttext and wa1 not in embedding):
            subwords_word_a1 = ['sw_{}'.format(sw)
                                for sw in getSubwords(regex.sub('', wa1), args.subwords)]
        else:
            subwords_word_a1 = [wa1]

        if not args.fasttext or (args.fasttext and wa2 not in embedding):
            subwords_word_a2 = ['sw_{}'.format(sw)
                                for sw in getSubwords(regex.sub('', wa2), args.subwords)]
        else:
            subwords_word_a2 = [wa2]

        if not args.fasttext or (args.fasttext and wb1 not in embedding):
            subwords_word_b1 = ['sw_{}'.format(sw)
                                for sw in getSubwords(regex.sub('', wb1), args.subwords)]
        else:
            subwords_word_b1 = [wb1]

        if not args.fasttext or (args.fasttext and wb2 not in embedding):
            subwords_word_b2 = ['sw_{}'.format(sw)
                                for sw in getSubwords(regex.sub('', wb2), args.subwords)]
        else:
            subwords_word_b2 = [wb2]

        nf = False

        a1 = np.zeros(next(iter(embedding.values())).size)
        for word in subwords_word_a1:
            if word not in embedding:
                nf = True
                break
            a1 += embedding[word]
        a1 /= len(subwords_word_a1)

        a2 = np.zeros(next(iter(embedding.values())).size)
        for word in subwords_word_a2:
            if word not in embedding:
                nf = True
                break
            a2 += embedding[word]
        a2 /= len(subwords_word_a2)

        b1 = np.zeros(next(iter(embedding.values())).size)
        for word in subwords_word_b1:
            if word not in embedding:
                nf = True
                break
            b1 += embedding[word]
        b1 /= len(subwords_word_b1)

        b2 = np.zeros(next(iter(embedding.values())).size)
        for word in subwords_word_b2:
            if word not in embedding:
                nf = True
                break
            b2 += embedding[word]
        b2 /= len(subwords_word_b2)

        if nf:
            stats['not_found'] += 1
            return stats
        else:
            stats['found'] += 1

    solution = solveAnalogy(embedding, (a1, a2, b1, b2), wb2)

    stats['cos_add'] = int(solution['ca'])
    stats['cos_mul'] = int(solution['cm'])
    return stats

def evaluateAnalogyPar(filename, args):
    embedding = create_embedding(filename)
    analogies = {}
    for filename in results:
        print('Reading {} file'.format(filename))
        with open(os.path.join(FILE_DIR, filename)) as analogy_file:
            line = analogy_file.readline().strip()
            while line:
                if not line.startswith(':'):
                    line = analogy_file.readline().strip()
                    continue
                kind = line.replace(': ', '')
                analogies[kind] = []
                line = analogy_file.readline().strip()
                while line and not line.startswith(':'):
                    words = line.split()
                    analogies[kind].append(tuple(words if args.cased else
                                                 [w.lower() for w in words]))
                    line = analogy_file.readline().strip()
            print('Done.')

        results[filename] = {}
        for kind, awords in tqdm(analogies.items()):
            stats = {
                'cos_add': 0, 'cos_mul': 0,
                'found': 0, 'samples': len(awords), 'not_found': 0
            }
            #pool
            # print('Multiprocess start')
            # result_analogies = [processAnalogy(embedding, words, args) for words in tqdm(awords)]
            with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            #     result_analogies = pool.starmap(processAnalogy, [(embedding, words, args)
            #                                                      for words in tqdm(awords)])
                result_analogies = pool.map(processAnalogy, [(embedding, words, args)
                                                             for words in tqdm(awords)])

            for result in result_analogies:
                for stat in result:
                    stats[stat] += result[stat]

            results[filename][kind] = stats

def evaluateAnalogy(filename, args):
    #using unofficial version of gensim for modified analogy evaluation
    sys.path.insert(1, '<gensim_dir>')
    from gensim.models import KeyedVectors

    embedding = KeyedVectors.load_word2vec_format(filename)
    print(f"Solving analogies with k={args.k}")
    if args.search_sol:
        print("Looking for solution in topk results.")
    for eval_file in results:
        #3CosAdd
        scores_ca = embedding.evaluate_word_analogies(os.path.join(FILE_DIR, eval_file),
                                                      case_insensitive=(not args.cased),
                                                      topk=args.k,
                                                      search_solution=args.search_sol)
        scores_cm = embedding.evaluate_word_analogies(os.path.join(FILE_DIR, eval_file),
                                                      case_insensitive=(not args.cased),
                                                      topk=args.k, method='3CosMul',
                                                      search_solution=args.search_sol)
        results[eval_file] = {
            'cosAdd': scores_ca[1][:-1],
            'cosMul': scores_cm[1][:-1],
            'total': {
                'cosAdd': scores_ca[0],
                'cosMul': scores_cm[0],
                'ca_error': scores_ca[1][-1]['error'],
                'cm_error': scores_cm[1][-1]['error'],
                'samples': scores_ca[1][-1]['total_samples'],
                'oov_ratio': scores_ca[1][-1]['oov_ratio']
            }
        }

def evaluateSimilarity(filename, args):
    """Compute Spearman rank coefficient for each evaluation file"""

    # step 0 : create the word embedding
    embedding = create_embedding(filename)

    # step 1 : iterate over each evaluation data file and compute spearman
    for filename in results:
        found, not_found = 0, 0
        with open(os.path.join(FILE_DIR, filename)) as f:
            file_similarity = []
            embedding_similarity = []
            for line in f:
                w1, w2, val = line.split()
                if not args.cased:
                    w1, w2, val = w1.lower(), w2.lower(), float(val)
                else:
                    val = float(val)
                #TODO: reduce this
                if not args.subwords:
                    if not w1 in embedding or not w2 in embedding:
                        not_found += 1
                    else:
                        found += 1
                        v1, v2 = embedding[w1], embedding[w2]
                        cosine = cosineSim(v1, v2)
                        file_similarity.append(val)
                        embedding_similarity.append(cosine)

                        #tanimoto = tanimotoSim(v1, v2)
                        #file_similarity.append(val)
                        #embedding_similarity.append(tanimoto)
                else:
                    #TODO: check for normalized word beforehand?
                    #some words have dashes (-)
                    if not args.fasttext or (args.fasttext and w1 not in embedding):
                        subwords_word1 = ['sw_{}'.format(sw)
                                          for sw in getSubwords(regex.sub('', w1), args.subwords)]
                    else:
                        subwords_word1 = [w1]

                    if not args.fasttext or (args.fasttext and w2 not in embedding):
                        subwords_word2 = ['sw_{}'.format(sw)
                                          for sw in getSubwords(regex.sub('', w2), args.subwords)]
                    else:
                        subwords_word2 = [w2]

                    nf = False

                    v1 = np.zeros(next(iter(embedding.values())).size)
                    for word in subwords_word1:
                        if word not in embedding:
                            nf = True
                            break
                        v1 += embedding[word]
                    v1 /= len(subwords_word1)

                    v2 = np.zeros(next(iter(embedding.values())).size)
                    for word in subwords_word2:
                        if word not in embedding:
                            nf = True
                            break
                        v2 += embedding[word]
                    v2 /= len(subwords_word2)

                    if nf:
                        not_found += 1
                    else:
                        found += 1
                        cosine = cosineSim(v1, v2)
                        file_similarity.append(val)
                        embedding_similarity.append(cosine)
            try:
                rho, p_val = st.spearmanr(file_similarity, embedding_similarity)

            except Exception as e:
                pdb.set_trace()
                raise e
            results[filename].append(rho)
            oov[filename] = (found, found+not_found)

def stats(args):
    if args.evaluation_task == 'similarity':
        """Compute statistics on results"""
        title = "{}| {}| {}| {}| {}| {}".format("Filename".ljust(16),
                                  "AVG".ljust(5), "MIN".ljust(5), "MAX".ljust(5),
                                  "STD".ljust(5), "oov".ljust(5))
        print(title)
        print("="*len(title))

        weighted_avg = 0
        total_found  = 0
        total_total = 0

        prnt_str = ''

        for filename in sorted(results.keys()):
            average = sum(results[filename]) / float(len(results[filename]))
            minimum = min(results[filename])
            maximum = max(results[filename])
            std = sum([(results[filename][i] - average)**2 for i in
                       range(len(results[filename]))])
            std /= float(len(results[filename]))
            std = math.sqrt(std)

            weighted_avg += oov[filename][0] * average
            total_found  += oov[filename][0]
            total_total += oov[filename][1]

            ratio_oov = 100 - (oov[filename][0] /  oov[filename][1]) * 100

            print("{0}| {1:.3f}| {2:.3f}| {3:.3f}| {4:.3f}|  {5}%".format(
                  filename.ljust(16),
                  average, minimum, maximum, std, int(ratio_oov)))

        prnt_str += 'Total found: {}'.format(total_found)
        print("-"*len(title))
        print("{0}| {1:.3f}".format("W.Average".ljust(16),
                                    weighted_avg / total_found))
        print(prnt_str)
        print("Total words: {}".format(total_total))
    else:
        for filename, scores in results.items():
            print('Analogy task: {}'.format(filename))
            print('{:-^99}'.format(''))
            print('| {:<27} | 3CosAdd | 3CosMul | Num Samples | Not Found | CA Error | CM Error |'.format('Type'))
            for i in range(len(scores['cosAdd'])):
                print('| {:<27} | {:>7.2%} | {:>7.2%} | {:^11} | {:>9.2%} | {:>8.2f} | {:>8.2f} |'.format(
                        scores['cosAdd'][i]['section'],
                        len(scores['cosAdd'][i]['correct'])/scores['cosAdd'][i]['samples'],
                        len(scores['cosMul'][i]['correct'])/scores['cosMul'][i]['samples'],
                        scores['cosAdd'][i]['samples'],
                        scores['cosAdd'][i]['oov']/scores['cosAdd'][i]['samples'],
                        scores['cosAdd'][i]['error'],
                        scores['cosMul'][i]['error']
                    ))
            print('{:-^99}'.format(''))
            print('| {:<27} | {:>7.2%} | {:>7.2%} | {:^11} | {:>8.2f}% | {:>8.2f} | {:>8.2f} |'.format(
                    'Total:',
                    scores['total']['cosAdd'],
                    scores['total']['cosMul'],
                    scores['total']['samples'],
                    scores['total']['oov_ratio'],
                    scores['total']['ca_error'],
                    scores['total']['cm_error']
                ))
            print('{:-^99}'.format(''))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
             description="Evaluate semantic similarities or analogies of word embeddings.",
             )

    parser.add_argument('filenames', metavar='FILE', nargs='+',
                        help='Filename of word embedding(s) to evaluate.')
    parser.add_argument('-e', choices=['similarity', 'analogy'], dest='evaluation_task',
                        default='similarity', help='Evaluate word similarity or word analogies.')
    parser.add_argument('-k', type=int, dest='k', default=5,
                        help='Top k results when looking for solutions in word analogy.')
    parser.add_argument('-m', action='store_true', dest='search_sol', help='Looks for solution in top k results.')
    parser.add_argument('-s', action='store_true', dest='subwords', help='Use subword embeddings.')
    parser.add_argument('-f', action='store_true', dest='fasttext', help='Use fastText embeddings.')
    parser.add_argument('-c', action='store_true', dest='cased',
                        help='Cased words.')

    args = parser.parse_args()

    if args.fasttext:
        args.subwords = args.fasttext

    FILE_DIR = os.path.join(FILE_DIR, args.evaluation_task)
    init_results()
    for f in args.filenames:
        evaluate(f, args)
    stats(args)
