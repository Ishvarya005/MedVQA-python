# import tensorflow as tf
# from transformers import TFAutoModelForQuestionAnswering, AutoTokenizer, pipeline

# # Define questions and context
# QA_input = [
#     {'question': 'Where is HD caused?',
#      'context': 'The most vulnerable part of the brain in HD is called the striatum which is prone to HD. The striatum is a relatively small structure, deep underneath the wrinkly outside part of the brain, which is called the cortex.During the course of HD, cells in both the striatum and cortex shrink, dysfunction and eventually die. Study after study, investigating many hundreds of volunteers, has found that the striatum is the first location in the brain that shrinks in people carrying the HD mutation'},
    
#     {'question': 'What causes HD?',
#      'context': 'HD results from genetically programmed degeneration of brain cells, called neurons, in certain areas of the brain.'}
# ]

# # Select from the hub of pretrained models
# model_name = 'deepset/roberta-base-squad2'

# # Every model uses its own specific tokenizer, the text provided is converted into token IDs that are further passed to the model
# model = TFAutoModelForQuestionAnswering.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# # Process the first input
# input0 = tokenizer(QA_input[0]['question'], QA_input[0]['context'], return_tensors="tf")
# output0 = model(**input0)
# #we get the startlogits(starting tokenID for the ans),loss, endlogits,hiddenstate,attention (if you want you can specify as true) etc 

# # Extract answer using TensorFlow
# #extracting the starting and ending ids and are passed as input to the token ids that were already converted based on the text
# answer_start_idx = tf.argmax(output0['start_logits'], axis=1).numpy()[0]
# answer_end_idx = tf.argmax(output0['end_logits'], axis=1).numpy()[0]
# answer_tokens = input0['input_ids'][0, answer_start_idx: answer_end_idx + 1]
# answer = tokenizer.decode(answer_tokens.numpy())

# print("Q: {}\nAnswer: {}".format(QA_input[0]['question'], answer))

# # Process the second input
# input1 = tokenizer(QA_input[1]['question'], QA_input[1]['context'], return_tensors="tf")
# output1 = model(**input1)

# # Extract answer using TensorFlow
# answer_start_idx = tf.argmax(output1['start_logits'], axis=1).numpy()[0]
# answer_end_idx = tf.argmax(output1['end_logits'], axis=1).numpy()[0]
# answer_tokens = input1['input_ids'][0, answer_start_idx: answer_end_idx + 1]
# answer = tokenizer.decode(answer_tokens.numpy())

# print("Q: {}\nAnswer: {}".format(QA_input[1]['question'], answer))
import tensorflow as tf
from transformers import TFAutoModelForQuestionAnswering, AutoTokenizer, pipeline

# Define questions and context
QA_input = [
    {'question': 'Where is HD caused?',
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

# Select from the hub of pretrained models
model_name = 'deepset/roberta-base-squad2'

# Every model uses its own specific tokenizer, the text provided is converted into token IDs that are further passed to the model
model = TFAutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

for i, qa_pair in enumerate(QA_input):
    # Process the input
    input_data = tokenizer(qa_pair['question'], qa_pair['context'], return_tensors="tf")
    output = model(**input_data)

    # Extract answer using TensorFlow
    answer_start_idx = tf.argmax(output['start_logits'], axis=1).numpy()[0]
    answer_end_idx = tf.argmax(output['end_logits'], axis=1).numpy()[0]
    answer_tokens = input_data['input_ids'][0, answer_start_idx: answer_end_idx + 1]
    answer = tokenizer.decode(answer_tokens.numpy())

    print("Q{}: {}\nAnswer: {}".format(i + 1, qa_pair['question'], answer))
