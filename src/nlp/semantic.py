from sentence_transformers import SentenceTransformer, util

class SemanticScorer:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def get_similarity(self, student_text, model_text):
        student_emb = self.model.encode(student_text, convert_to_tensor=True)
        model_emb = self.model.encode(model_text, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(student_emb, model_emb)
        return similarity.item()
