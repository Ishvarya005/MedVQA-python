import nlpaug.augmenter.word as naw
import pandas as pd
import tensorflow as tf
from transformers import TFAutoModelForQuestionAnswering, AutoTokenizer

# Load pre-trained model
model_name = 'deepset/roberta-base-squad2'
model = TFAutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Fine-tuning data (replace this with your own dataset)
fine_tuning_data = [{'question': 'Where is HD caused?',
     'context': 'The most vulnerable part of the brain in HD is called the striatum which is prone to HD. The striatum is a relatively small structure, deep underneath the wrinkly outside part of the brain, which is called the cortex.During the course of HD, cells in both the striatum and cortex shrink, dysfunction and eventually die. Study after study, investigating many hundreds of volunteers, has found that the striatum is the first location in the brain that shrinks in people carrying the HD mutation'},
    
    {'question': 'What causes HD?',
     'context': 'HD results from genetically programmed degeneration of brain cells, called neurons, in certain areas of the brain.'},

    {'question': 'What is Huntington’s disease?',
     'context': 'Huntington‘s disease (HD), also known as Huntington’s chorea, is a rare degenerative inherited genetic disorder of the brain.'},

    {'question': 'Why is it called the Huntington’s disease?',
     'context': 'HD is named after George Huntington, an American medical doctor who described the disease accurately in 1872. His description was based on observations of HD affected families from the village of East Hampton, Long Island, New York (USA), where Huntington lived and worked as a physician. He was the first person to identify the pattern of inheritance of HD.'},


    {'question': 'What are the symptoms when HD starts late in life?',
     'context': 'When HD starts late in life, chorea tends to be stronger, whereas slowness and stiffness are less prominent. If HD occurs late in life, it is likely to be more difficult to establish a family history because the individual’s parents may have already died, perhaps before they themselves showed signs of the disease.'}
         ]

# Augment data with paraphrased questions
augmented_data = []
aug = naw.ParaphraseAug()

for item in fine_tuning_data:
    augmented_question = aug.augment(item['question'])
    augmented_data.append({'question': augmented_question, 'context': item['context'], 'answer': item['answer']})

# Combine original and augmented data
combined_data = fine_tuning_data + augmented_data

# Tokenize and preprocess the combined data
tokenized_data = tokenizer([item['question'] for item in combined_data],
                           [item['context'] for item in combined_data],
                           padding=True,
                           truncation=True,
                           return_tensors='tf')

# Prepare labels for the model
labels = {
    'start_positions': tf.convert_to_tensor([tokenizer.encode(item['answer'], add_special_tokens=False).index(1) for item in combined_data]),
    'end_positions': tf.convert_to_tensor([tokenizer.encode(item['answer'], add_special_tokens=False).index(1) + len(tokenizer.encode(item['answer'], add_special_tokens=False)) - 1 for item in combined_data]),
}

# Fine-tune the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5), loss=model.compute_loss, metrics=['accuracy'])
model.fit(tokenized_data, labels, epochs=3)
