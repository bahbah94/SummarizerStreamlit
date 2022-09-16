nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx


docu = '''
In an attempt to build an AI-ready workforce, Microsoft announced Intelligent Cloud Hub which has been launched to empower the next generation of students with AI-ready skills. Envisioned as a three-year collaborative program, Intelligent Cloud Hub will support around 100 institutions with AI infrastructure, course content and curriculum, developer support, development tools and give students access to cloud and AI services. As part of the program, the Redmond giant which wants to expand its reach and is planning to build a strong developer ecosystem in India with the program will set up the core AI infrastructure and IoT Hub for the selected campuses. The company will provide AI development tools and Azure AI services such as Microsoft Cognitive Services, Bot Services and Azure Machine Learning.According to Manish Prakash, Country General Manager-PS, Health and Education, Microsoft India, said, "With AI being the defining technology of our time, it is transforming lives and industry and the jobs of tomorrow will require a different skillset. This will require more collaborations and training and working with AI. That’s why it has become more critical than ever for educational institutions to integrate new cloud and AI technologies. The program is an attempt to ramp up the institutional set-up and build capabilities among the educators to educate the workforce of tomorrow." The program aims to build up the cognitive skills and in-depth understanding of developing intelligent cloud connected solutions for applications across industry. Earlier in April this year, the company announced Microsoft Professional Program In AI as a learning track open to the public. The program was developed to provide job ready skills to programmers who wanted to hone their skills in AI and data science with a series of online courses which featured hands-on labs and expert instructors as well. This program also included developer-focused AI school that provided a bunch of assets to help build AI skills.
'''

def clean_sentence(document):
    sentences = document.split(".")
    sentences = [s.strip() for s in sentences]
    sentences = [' '.join(s.lower() for s in sentence.split()) for sentence in sentences]
    #print(sentences[0])
    #return 0
    sentences = [sentence.replace("[^a-zA-Z]", " ").split(" ") for sentence in sentences]
    return sentences
def similarity(sent1,sent2,stopwords=None):
    if stopwords is None:
        stopwords = []
    all_words = list(set(sent1+sent2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1

    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1

    return 1 - cosine_distance(vector1, vector2)

def build_similarity_matrix(sentences, stopwords):
    similarity_matrix = np.zeros((len(sentences),len(sentences)))

    for i in range(len(sentences)):
        for j in range(i,len(sentences)):
            if i == j:
                continue
            similarity_matrix[i][j] = similarity(sentences[i],sentences[j],stopwords)
            similarity_matrix[j][i] = similarity_matrix[i][j]
    return similarity_matrix
def generate_summary(doc,top_n=5):
    stop_words = stopwords.words('english')
    summarized = []

    sentences = clean_sentence(doc)

    sentence_sim_matrix = build_similarity_matrix(sentences,stop_words)

    sentence_sim_graph = nx.from_numpy_array(sentence_sim_matrix)
    scores = nx.pagerank(sentence_sim_graph)

    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)

    for i in range(top_n):
        summarized.append(" ".join(ranked_sentence[i][1]))

    #print("scores are ",ranked_sentence)

    return ". ".join(summarized)

print(generate_summary(docu))
