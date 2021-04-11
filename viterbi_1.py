from math import log
from collections import Counter
import numpy as np


def viterbi_1(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''

    """
    initial p: how often does each tag occur at start of sentence
    transition p: how often does tag t_b follow tag t_a
    emission p: how often does tag t yield word w
    """
    emission_alpha = 0.00001
    transition_alpha = 0.00001
    cnt_tag = Counter()
    cnt_word = Counter()
    cnt_start = Counter()
    tag_index = {}
    emission_count = {}
    transition_count = {}
    i = 0
    prev_tag = 'START'
    for sentence in train:
        cnt_start[sentence[0][1]] += 1
        for pair in sentence:
            if pair in emission_count:
                emission_count[pair] += 1
            if pair not in emission_count:
                emission_count[pair] = 1
            if (prev_tag, pair[1]) in transition_count:
                transition_count[(prev_tag, pair[1])] += 1
            if (prev_tag, pair[1]) not in transition_count:
                transition_count[(prev_tag, pair[1])] = 1
            if pair[1] not in cnt_tag:
                tag_index[pair[1]] = i
                i += 1
            cnt_tag[pair[1]] += 1
            cnt_word[pair[0]] += 1
            prev_tag = pair[1]
    transition_count.pop(('START', 'START'))

    num_tag = len(cnt_tag)
    num_word = len(cnt_word)
    vocab = {}
    wordcount = 0
    for i, word in enumerate(cnt_word):
        wordcount += 1
        vocab[word] = i
    vocab['UNK'] = wordcount
    # print(len(vocab))
    tags = list(cnt_tag.keys())
    B = np.zeros((num_tag, len(vocab)))
    B_keys = set(list(emission_count.keys()))
    # print(num_word)
    # print(len(emission_count))
    for i in range(num_tag):
        for j, word in enumerate(vocab):
            count = 0
            if (word, tags[i]) in B_keys:
                count = emission_count[(word, tags[i])]
            count_tag = cnt_tag[tags[i]]
            B[i, j] = log((count+emission_alpha)/(count_tag+emission_alpha*(len(cnt_tag)+1)))

    A = np.zeros((num_tag, num_tag))
    A_keys = set(transition_count.keys())
    for i in range(num_tag):
        for j in range(num_tag):
            count = 0
            if (tags[i], tags[j]) in A_keys:
                count = transition_count[(tags[i], tags[j])]
            count_prev_tag = cnt_tag[tags[i]]
            A[i, j] = log((count + transition_alpha)/(count_prev_tag+transition_alpha*(len(cnt_tag)+1)))

    Pi = []
    for i in range(0, num_tag):
        if i == 0:
            Pi.append(log((emission_alpha+cnt_start['START'])/(sum(cnt_start.values())+emission_alpha*(len(cnt_start)+1))))
        else:
            Pi.append(log(emission_alpha/(sum(cnt_start.values())+emission_alpha*(len(cnt_start)+1))))

    tag_list = list(cnt_tag.keys())
    result = []
    # test data on trellis
    for sentence in test:
        temp = []
        wrapper = []
        trellis = np.zeros((num_tag, len(sentence)))
        back_ptr = np.zeros((num_tag, len(sentence)))
        path = np.empty(len(sentence))
        # initialization
        trellis[:, 0] = Pi * B[:, vocab[sentence[0]]]
        back_ptr[:, 0] = 0
        for t in range(1, len(sentence)):
            for i in range(len(tag_list)):
                if sentence[t] in vocab:
                    trellis[i, t] = np.max(trellis[:, t - 1] + A[:, i] + B[i, vocab[sentence[t]]])
                    back_ptr[i, t] = np.argmax(trellis[:, t - 1] + A[:, i] + B[i, vocab[sentence[t]]])
                else:
                    trellis[i, t] = np.max(trellis[:, t - 1] + A[:, i] + B[i, vocab['UNK']])
                    back_ptr[i, t] = np.argmax(trellis[:, t - 1] + A[:, i] + B[i, vocab['UNK']])

        path[len(sentence)-1] = np.argmax(trellis[:, len(sentence)-1])
        for t in range(len(sentence)-2, -1, -1):
            path[t] = back_ptr[int(path[t+1]), t+1]
        temp = path.tolist()
        for i in range(len(sentence)):
            wrapper.append((sentence[i], tag_list[int(temp[i])]))
        result.append(wrapper)

    return result
