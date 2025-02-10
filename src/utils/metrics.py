from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from bert_score import score
from bleurt import score as bleurt_score
import torch
import numpy as np
import tensorflow
import evaluate

# Load the BLEURT metric
bleurt = evaluate.load("bleurt", config_name="bleurt-20",device='cuda')
bertscore = evaluate.load("bertscore")
checkpoint = "C:\\Users\\charl\\Research\\forseer\\src\\run-scripts\\BLEURT-20"
scorer = bleurt_score.BleurtScorer(checkpoint)
def bleu(a,b):
    #print([a.split()])
    return sentence_bleu([a.split()],b.split())

def rouge(a,b):
    a = a.lower()
    b = b.lower()
    r = Rouge()
    scores = r.get_scores(a,b)
    return scores

def bert_score(a,b):
    results = bertscore.compute(predictions=[b], references=[a], lang="en")
    return results['f1']

def bleu_rt(a,b):
    #results = bleurt.compute(predictions=[b], references=[a])
    scores = scorer.score(references=[a], candidates=[b])
    return scores



def cosine_similarity(vector_a, vector_b):
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    return dot_product / (norm_a * norm_b)