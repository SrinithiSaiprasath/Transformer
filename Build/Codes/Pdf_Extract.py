from PyPDF2 import PdfReader
import pdfplumber
import re
import csv 
from Build.Codes.Transformer import *
# from transformers import AutoTokenizer
# from Transformer import * 

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def categorize_text_as_heading_content(text):
    # Pattern for headings (all-caps assumed as headings)
    heading_pattern = re.compile(r'^[A-Z\s]{2,}$')

    lines = text.splitlines()
    categorized_content = []
    current_heading = None
    current_content = []

    for line in lines:
        stripped_line = line.strip()
        if heading_pattern.match(stripped_line):
            if current_heading:
                # Finalize the previous heading and its content
                categorized_content.append((current_heading, " ".join(current_content)))
                current_content = []
            current_heading = stripped_line
        else:
            current_content.append(stripped_line)

    # Add any remaining content under the last heading
    if current_heading:
        categorized_content.append((current_heading, " ".join(current_content)))

    return categorized_content

def process_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        full_text = ""
        for page in pdf.pages:
            full_text += page.extract_text() + "\n"
        
        categorized_content = categorize_text_as_heading_content(full_text)
        return categorized_content


# output_file = "/workspaces/transformer/Dataset/output_file.csv"
def convert_to_csv(pdf_path,output_file):
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['question', 'answer'])  # Write the header
        content = process_pdf(pdf_path)
        for heading, contents in content:
            writer.writerow([heading, contents]) 

# convert_to_csv(pdf_path,output_file)

def get_vocab_size(questions,answers):
    tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        questions + answers, target_vocab_size=2**13)

    # questions , answers , tokenizer = content_split(pdf_path)

    # Define start and end token to indicate the start and end of a sentence
    START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]

    # Vocabulary size plus start and end token
    VOCAB_SIZE = tokenizer.vocab_size + 2

    return START_TOKEN, END_TOKEN , VOCAB_SIZE , tokenizer
def model_fit(pdf_path ):
    pdf_name = os.path.basename(pdf_path)
    pdf_name = os.path.splitext(pdf_name)[0]
    output_file = f"Dataset/CSV_Dataset/{pdf_name}.csv"
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['question', 'answer'])  # Write the header
        content = process_pdf(pdf_path)
        for heading, contents in content:
            writer.writerow([heading, contents]) 


    df = pd.read_csv(output_file)
    print(df.columns)
    df['question'] = df["question"].apply(lambda x: textPreprocess(str(x)))
    df['answer'] = df["answer"].apply(lambda x: textPreprocess(str(x)))
    questions, answers = df['question'].tolist(), df['answer'].tolist()
    START_TOKEN,END_TOKEN,VOCAB_SIZE,tokenizer =get_vocab_size(questions,answers)

    
    questions, answers = tokenize_and_filter(questions, answers,START_TOKEN , END_TOKEN,tokenizer)
    print('Vocab size: {}'.format(VOCAB_SIZE))
    print('Number of samples: {}'.format(len(questions)))

    dataset = tf.data.Dataset.from_tensor_slices((
        {
            'inputs': questions,
            'dec_inputs': answers[:, :-1]
        },
        {
            'outputs': answers[:, 1:]
        },
    ))
    dataset = dataset.cache()
    dataset = dataset.shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        
    tf.keras.backend.clear_session()
    # learning_rate = CustomSchedule(D_MODEL)
    # optimizer = tf.keras.optimizers.Adam()
        
    with strategy.scope():
        model = transformer(
            vocab_size=VOCAB_SIZE,
            num_layers=NUM_LAYERS,
            units=UNITS,
            d_model=D_MODEL,
            num_heads=NUM_HEADS,
            dropout=DROPOUT)

        model.compile(optimizer='adam', loss=loss_function)
    model.summary()

    import datetime
    logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
    model.fit(dataset, epochs = 150, callbacks = [tensorboard_callback])
    model.save_weights(f'Weights/{pdf_name}.weights.h5')
    
    return model
def load_model(pdf_path):
    pdf_name = os.path.basename(pdf_path)
    pdf_name = os.path.splitext(pdf_name)[0]
    choice = pdf_path
    output_file = f"Dataset/CSV_Dataset/{pdf_name}.csv"

    df = pd.read_csv(output_file)
    print(df.columns)
    df['question'] = df["question"].apply(lambda x: textPreprocess(str(x)))
    df['answer'] = df["answer"].apply(lambda x: textPreprocess(str(x)))
    questions, answers = df['question'].tolist(), df['answer'].tolist()

    START_TOKEN,END_TOKEN,VOCAB_SIZE,tokenizer =get_vocab_size(questions,answers)
    questions, answers = tokenize_and_filter(questions, answers,START_TOKEN , END_TOKEN , tokenizer)

    tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(questions + answers, target_vocab_size=2**13)
    dataset = tf.data.Dataset.from_tensor_slices((
        {
            'inputs': questions,
            'dec_inputs': answers[:, :-1]
        },
        {
            'outputs': answers[:, 1:]
        },
    ))
    dataset = dataset.cache()
    dataset = dataset.shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        
    tf.keras.backend.clear_session()
    learning_rate = CustomSchedule(D_MODEL)
    optimizer = tf.keras.optimizers.Adam()
        
    with strategy.scope():
        model = transformer(
            vocab_size=VOCAB_SIZE,
            num_layers=NUM_LAYERS,
            units=UNITS,
            d_model=D_MODEL,
            num_heads=NUM_HEADS,
            dropout=DROPOUT)

        model.compile(optimizer='adam', loss=loss_function)
    loaded_model = transformer(
        vocab_size=VOCAB_SIZE,
        num_layers=NUM_LAYERS,
        units=UNITS,
        d_model=D_MODEL,  
        num_heads=NUM_HEADS,
        dropout=DROPOUT)
    loaded_model.compile(optimizer=optimizer, loss=loss_function)
    loaded_model.load(f"{choice}.weights.h5")
    return loaded_model
