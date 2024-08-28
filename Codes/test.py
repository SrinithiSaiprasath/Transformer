import PyMuPDF as fitz
import torch
from transformers import BertTokenizer, BertForQuestionAnswering

# Load the model and tokenizer
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"  # Example model
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

# Function to chunk text into manageable sizes
def chunk_text(text, max_tokens=512):
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        if len(tokenizer.encode(" ".join(current_chunk + [word]), add_special_tokens=False)) <= max_tokens:
            current_chunk.append(word)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# Function to get answer from the model
def get_answer(question, context):
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits

    # Get the most likely beginning and end of the answer
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1
    
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
    )

    return answer

# Function to answer a question using the PDF content
def answer_question_with_transformer(question, text_chunks):
    answers = []
    for chunk in text_chunks:
        answer = get_answer(question, chunk)
        answers.append(answer)
    return answers

# Function to aggregate the answers and pick the most frequent one
def aggregate_answers(answers):
    return Counter(answers).most_common(1)[0][0]

# Chatbot response function
def chatbot_respond(pdf_path, user_question):
    # Extract and chunk the PDF text
    pdf_text = extract_text_from_pdf(pdf_path)
    text_chunks = chunk_text(pdf_text, max_tokens=512)
    
    # Get answers using the transformer model
    answers = answer_question_with_transformer(user_question, text_chunks)
    
    # Aggregate and return the final answer
    return aggregate_answers(answers)

# Example usage
response = chatbot_respond("your_pdf_file.pdf", "What are the key points?")
print(response)
