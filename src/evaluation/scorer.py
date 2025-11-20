class Evaluator:
    def __init__(self, rubric):
        self.rubric = rubric  # dict with criteria and marks

    def assign_score(self, similarity_score):
        # Example: simple linear scaling
        total_marks = sum(self.rubric.values())
        return similarity_score * total_marks
