
def emission(gene_count):
    '''
    find number of times a tag appear
    345128 1-GRAM O
    41072 1-GRAM I-GENE
    '''
    with open(gene_count,'r') as f:
        tags_count = dict()
        tags = list()
        for line in f:
            line_list = line.strip().split(" ")
            if(line_list[1] == '1-GRAM'):
                tags.append(line_list[2])
                tags_count[line_list[2]] = int(line_list[0])
            else:
                continue
    return tags_count, tags

def base_tagger(gene_count, gene_dev, tags_count):
    '''
    check tag emission
    ex:
    345128 1-GRAM O
    41072 1-GRAM I-GENE
    reverse tag O 3 times
            tag I-GENE 2 times
    -> emission O = 3 / 345128
       emission I-GENE = 2 / 41072
    '''
    tag_emiss = dict()
    best_tag = dict()
    with open(gene_count,'r') as f:
        for line in f:
            line_list = line.strip().split(" ")
            if(line_list[1] != "WORDTAG"):
                        break
            tag = line_list[2]

            emiss = int(line_list[0]) / tags_count[tag]
            word = line_list[3]

            if line_list[3] in best_tag and tag_emiss[word] > emiss:
                continue
            else:
                best_tag[word] = tag
                tag_emiss[word] = emiss

    fd = open(gene_dev,'r')
    fp = open('gene_dev.p1.out_t','w')
    for word in fd:
        word = word.strip()
        if word == '':
            fp.write('\n')
            continue
        if word in best_tag:
            fp.write(word + ' ' + best_tag[word] + '\n')
        else:
            fp.write(word + ' ' + best_tag["__RARE__"] + '\n')
    fd.close()
    fp.close()

def base_tagger_inform(gene_count, gene_dev, tags_count):
    '''
    check tag emission
    ex:
    345128 1-GRAM O
    41072 1-GRAM I-GENE
    reverse tag O 3 times
            tag I-GENE 2 times
    -> emission O = 3 / 345128
       emission I-GENE = 2 / 41072
    '''
    tag_emiss = dict()
    best_tag = dict()
    with open(gene_count,'r') as f:
        for line in f:
            line_list = line.strip().split(" ")
            if(line_list[1] != "WORDTAG"):
                        break
            tag = line_list[2]

            emiss = int(line_list[0]) / tags_count[tag]
            word = line_list[3]

            if line_list[3] in best_tag and tag_emiss[word] > emiss:
                continue
            else:
                best_tag[word] = tag
                tag_emiss[word] = emiss

    fd = open(gene_dev,'r')
    fp = open('gene_dev.p1.out.inform_t','w')
    for word in fd:
        word = word.strip()
        if word == '':
            fp.write('\n')
            continue
        if word in best_tag:
            fp.write(word + ' ' + best_tag[word] + '\n')
        else:
            digit_tag = False
            alpha_tag = False
            for c in word:
                if c.isdigit():
                    digit_tag = True
                if c.isalpha():
                    alpha_tag = True

            if alpha_tag and digit_tag:
                fp.write(word + ' ' + best_tag["__DigitAndAlpha__"] + '\n')

            # elif digit_tag and not alpha_tag:
            #     fp.write(word + ' ' + best_tag["__othernum__"] + '\n')

            # elif word.isupper():
            #     fp.write(word + ' ' + best_tag["__allCaps__"] + '\n')
            #
            # elif word[0].isupper():
            #     fp.write(word + ' ' + best_tag["__initCap__"] + '\n')

            else:
                fp.write(word + ' ' + best_tag["__RARE__"] + '\n')
    fd.close()
    fp.close()


def find_uni_tag(gene_count,tags_count):
    '''
    input:
    tags_count: dict
        O: 345128
        I-GENE: 41072
    output:
    uni_tag: dict
        ('reverse','O'): 3/345128
        ('reverse','I-GENE'): 2/41072
    '''
    uni_tag = dict()
    with open(gene_count,'r') as f:
        for line in f:
            line_list = line.strip().split(" ")
            if(line_list[1] != "WORDTAG"):
                        break
            tag = line_list[2]
            emiss = int(line_list[0]) / tags_count[tag]
            word = line_list[3]
            uni_tag[(word, tag)] = emiss
    return uni_tag


def find_tri_tag(gene_count):
    '''
    output:
        tri_tag: dict()
        (*,*,I-GENE): count(*,*,I-GENE) / count(*,*)
    '''
    with open(gene_count,'r') as f:
        bi_count = dict()
        tri_count = dict()
        tri_tag = dict()

        for line in f:
            line_list = line.strip().split(" ")
            if(line_list[1] == '2-GRAM'):
                bi_count[(line_list[2],line_list[3])] = int(line_list[0])
            elif(line_list[1] == '3-GRAM'):
                tri_count[(line_list[2], line_list[3], line_list[4])] = int(line_list[0])
            else:
                continue
        for k in tri_count:
            tri_tag[k] = tri_count[k] / bi_count[(k[0],k[1])]
    return tri_tag



def viterbi(sentence_orig, uni_tag, tri_tag, tags):
    '''
    sentence: word list
    uni_tag: e(x|y) = count(y->x)/count(y) dict
    tri_tag: q(yi | yi-2, yi-1) = count(yi-2,yi-1,yi) / count(yi-2,yi-1) dict
    tags: list of tags
    '''
    import math
    sentence = sentence_orig.copy()
    n = len(sentence)

    # rare viterbi
    #===========================================================================
    # for i in range(n):
    #     not_in = True
    #     for t in tags:
    #         if (sentence[i],t) in uni_tag:
    #             not_in = False
    #     if not_in:
    #         sentence[i] = '__RARE__'

    #===========================================================================

    # rare viterbi with informative classes
    #===========================================================================
    for i in range(n):
        not_in = True
        word = sentence[i]
        for t in tags:
            if (word,t) in uni_tag:
                not_in = False
        if not_in:
            digit_tag = False
            alpha_tag = False
            for c in word:
                if c.isdigit():
                    digit_tag = True
                if c.isalpha():
                    alpha_tag = True

            if alpha_tag and digit_tag:
                sentence[i] = '__DigitAndAlpha__'

            # elif digit_tag and not alpha_tag:
            #     sentence[i] = '__othernum__'

            # elif word.isupper():
            #     sentence[i] = '__allCaps__'
            #
            # elif word[0].isupper():
            #     sentence[i] = '__initCap__'
            else:
                sentence[i] = '__RARE__'
    #===========================================================================

    pi = [{} for i in range(n+1)]
    bp = [{} for i in range(n+1)]

    # k=0
    u_k0 = tags.copy()
    u_k0.append('*')
    v_k0 = tags.copy()
    v_k0.append('*')
    for u in u_k0:
        for v in v_k0:
            if(u=='*' and v=='*'):
                pi[0][(u,v)] = 1
            else:
                pi[0][(u,v)] = 0
    # k=1
    u_k1 = tags.copy()
    u_k1.append('*')
    v_k1 = tags.copy()
    # w_k1 = tags.copy()
    # w_k1.append('*')
    w_k1 = ['*']

    for u in u_k1:
        for v in v_k1:
            p_max = -math.inf
            w_max = ''
            for w in w_k1:
                if (sentence[0],v) in uni_tag:
                    e = uni_tag[(sentence[0],v)]
                else:
                    e = 0
                if (w,u,v) in tri_tag:
                    q = tri_tag[(w,u,v)]
                else:
                    q = 0

                p = pi[0][(w,u)]*q*e
                if p>p_max:
                    p_max = p
                    w_max = w
            pi[1][(u,v)] = p_max
            bp[1][(u,v)] = w_max


    # k=2
    u_k2 = tags.copy()
    v_k2 = tags.copy()
    # w_k2 = tags.copy()
    # w_k2.append('*')
    w_k2 = ['*']

    for u in u_k2:
        for v in v_k2:
            p_max = -math.inf
            w_max = ''
            for w in w_k2:
                if (sentence[1],v) in uni_tag:
                    e = uni_tag[(sentence[1],v)]
                else:
                    e = 0
                if (w,u,v) in tri_tag:
                    q = tri_tag[(w,u,v)]
                else:
                    q = 0

                p = pi[1][(w,u)]*q*e
                if p>p_max:
                    p_max = p
                    w_max = w
            pi[2][(u,v)] = p_max
            bp[2][(u,v)] = w_max

    # k=3 ~ k=n
    u_k3 = tags.copy()
    v_k3 = tags.copy()
    w_k3 = tags.copy()
    for k in range(3,n+1):
        for u in u_k3:
            for v in v_k3:
                p_max = -math.inf
                w_max = ''
                for w in w_k3:
                    if (sentence[k-1],v) in uni_tag:
                        e = uni_tag[(sentence[k-1],v)]
                    else:
                        e = 0
                    if (w,u,v) in tri_tag:
                        q = tri_tag[(w,u,v)]
                    else:
                        q = 0

                    p = pi[k-1][(w,u)]*q*e
                    #print('pi: {}'.format(pi[k-1][(w,u)]) + ' q: {}'.format(q) + ' e: {}'.format(e))
                    if p>p_max:
                        p_max = p
                        w_max = w
                pi[k][(u,v)] = p_max
                bp[k][(u,v)] = w_max

    p_max = -math.inf
    y_n = ""
    y_n_1 = ""
    for u in u_k2:
        for v in v_k2:
            p = pi[n][(u,v)]*tri_tag[(u,v,'STOP')]
            if p>p_max:
                p_max = p
                y_n_1 = u
                y_n = v

    sentence_tag_r = list()
    sentence_tag_r.append(y_n)
    sentence_tag_r.append(y_n_1)
    for k in range(n-2,0,-1):
        sentence_tag_r.append(bp[k+2][(sentence_tag_r[-1],sentence_tag_r[-2])])
    sentence_tag = list(reversed(sentence_tag_r))
    return sentence_tag

def viterbi_tagger(gene_dev, uni_tag, tri_tag, tags):
    fd = open(gene_dev,'r')
    # fp = open('gene_dev.p1.out.viterbi','w')
    fp = open('gene_dev.p1.out.viterbi.inform','w')
    sentence = list()
    for word in fd:
        word = word.strip()
        if word != '':
            sentence.append(word)
        else:
            if len(sentence) == 1:
                sentence_tag = ['O']
            else:
                sentence_tag = viterbi(sentence, uni_tag, tri_tag, tags)
            for i,tag in enumerate(sentence_tag):
                fp.write(sentence[i] + " " + tag + '\n')
            fp.write('\n')
            sentence = list()

    fd.close()
    fp.close()

if __name__ == '__main__':
    # tags_count, tags = emission('gene.counts.rare')
    tags_count, tags = emission('gene.counts.rare.inform')

    # base_tagger('gene.counts.rare', 'gene.dev',tags_count)
    # base_tagger('gene.counts.rare', 'gene.train.unlabel',tags_count)
    # base_tagger_inform('gene.counts.rare.inform','gene.dev', tags_count)
    base_tagger_inform('gene.counts.rare.inform','gene.train.unlabel',tags_count)

    # uni_tag = find_uni_tag('gene.counts.rare',tags_count)
    # tri_tag = find_tri_tag('gene.counts.rare')

    # uni_tag = find_uni_tag('gene.counts.rare.inform',tags_count)
    # tri_tag = find_tri_tag('gene.counts.rare.inform')
    # viterbi_tagger('gene.dev', uni_tag, tri_tag, tags)
    # viterbi_tagger('gene.train.unlabel', uni_tag, tri_tag, tags)
