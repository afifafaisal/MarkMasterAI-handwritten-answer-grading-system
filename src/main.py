from ocr.ocr import OCRProcessor
from nlp.semantic import SemanticScorer
from evaluation.scorer import Evaluator

rubric = {"definition":2, "explanation":3, "example":2, "clarity":1, "extra":2}

ocr = OCRProcessor()
scorer = SemanticScorer()
evaluator = Evaluator(rubric)

# Step 1: OCR
student_text = ocr.extract_text('data/raw/student_answer.jpg')
print("Student Answer:", student_text)

# Step 2: Semantic similarity
model_answer = "Mitochondria is the powerhouse of the cell."
similarity = scorer.get_similarity(student_text, model_answer)
print("Similarity Score:", similarity)

# Step 3: Evaluate marks
marks = evaluator.assign_score(similarity)
print("Marks Awarded:", marks)
