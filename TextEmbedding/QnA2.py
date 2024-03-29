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
     'context': 'When HD starts late in life, chorea tends to be stronger, whereas slowness and stiffness are less prominent. If HD occurs late in life, it is likely to be more difficult to establish a family history because the individual’s parents may have already died, perhaps before they themselves showed signs of the disease.'},
<<<<<<< HEAD
    
    {'question': 'When do HD symptoms appear?',
      'context':'Most individuals develop the disease during mid-adult life, i.e. between 35 and 55 years of age. Approximately 10% of people develop symptoms prior to the age of 20 (juvenile HD) and another 10% after the age of 55. More rarely, symptoms appear before the age of 10 years (infantile HD).'},
    { 'question':'How long does HD last?',
     'context':'HD is a fatal illness, developing at a gradual and relentless rate. The average duration of the disease is 15-20 years, but this varies between individuals.'},
    {'question':'How do I know if I have HD?',
      'context':'If you suspect that you have HD, you should consult an HD specialist (usually a neurologist) for diagnostic testing.'},
    {'question':'How common is HD?'},
    { 'context':'HD is a rare disease which affects up to approximately 1 in 10,000 people in most European countries. In Germany for instance, about 10,000 people have HD and a further 50,000 are considered at risk for inheriting the HD gene because they have (or had) a parent with HD. Men and women are equally likely to inherit the gene and develop the disease.'},
     {'question':'How does HD start?'},
    { 'context':'The first subtle signs may be slight personality or mood changes. Forgetfulness, clumsiness and random, brief, “fidgeting” movements of the fingers ortoes might also be a hint. Often, medical advice is not sought during these very early stages of the disease, and some years may pass by before the disorder is medically diagnosed. Hence, the onset of HD is described as “insidious”, as the disease emerges very slowly.'},
    {'question':'What are the symptoms of HD?'},
    { 'context':'The first subtle signs may be slight personality or mood changes. Forgetfulness, clumsiness and random, brief, “fidgeting” movements of the fingers ortoes might also be a hint. Often, medical advice is not sought during these very early stages of the disease, and some years may pass by before the disorder is medically diagnosed. Hence, the onset of HD is described as “insidious”, as the disease emerges very slowly.'},
    {'question':'How does HD progress?'},
    { 'context':'HD can be divided into five stages: • Early Stage: The person is diagnosed as having HD and can function fully both at home and work. Early Intermediate Stage: The person remains employable but at a lower capacity. He/she is still able to manage daily affairs despite some difficulties.  Late Intermediate Stage: The person can no longer work and manage household responsibilities. He/she needs considerable help or supervision to handle daily financial affairs. Other daily activities may be slightly difficult but usually only require minor help. Early Advanced Stage: The person is no longer independent in daily activities but is still able to live at home supported by the family or professional carers. Advanced Stage: The person requires complete support in daily activities and professional nursing care is usually needed.'},    
    {'question':'Do the symptoms of juvenile HD differ from those of the adult form?'},
    { 'context':'When HD starts early in life (i.e. under the age of 20), chorea is less prominent whereas slowness of movement (bradykinesia) and stiffness become more prevalent. In most cases, the rate of progression of juvenile HD tends to be faster than in the adult form. Early features of juvenile HD include strong behavioural changes, learning problems, decline at school and speech problems. Epileptic seizures occasionally occur in HD, being more common among young patients.'},
     {'question':'What are the symptoms when HD starts late in life?'},
    {'context':'When HD starts late in life, chorea tends to be stronger, whereas slowness and stiffness are less prominent. If HD occurs late in life, it is likely to be more difficult to establish a family history because the individual’s parent  may have already died, perhaps before they themselves showed signs of the disease.'},
    {'question':'What type of disease is HD'},
    {'context':'HD is an autosomal genetic disease. This means that it may affect both men and women equally because the abnormal gene is located on a chromosome which is the same in both sexes (autosome or non-sex chromosome).'},
    {'question':'Can HD skip a generation?'},
    {'context':'If a person does not inherit the HD gene, he/she will not develop the disease and will not pass HD on to the next generation. The HD gene cannot skip a generation, but the symptoms can. This situation may occur if the gene carrier dies before the symptoms appear, so that it becomes more difficult to establish a family history.'},
     {'question': 'Is juvenile HD always inherited from the father?',
=======

    {'question': 'Is juvenile HD always inherited from the father?',
>>>>>>> origin/master
     'context': 'Juvenile Huntington’s disease (HD) is primarily inherited from fathers (75%) or mothers (25%). When the gene has 36 or more CAG units, the repeats are more likely to change in size, particularly when inherited from the father, leading to earlier symptom onset due to a phenomenon known as anticipation.'},

    {'question': 'If a man carries the HD gene, does this mean that his children will develop juvenile HD?',
     'context' : 'Juvenile onset is rare. If a man is affected, it does not follow that his children will necessarily have juvenile HD.'},

    {'question': 'Can HD strike without a family history of the disease?',
     'context': 'Yes, but this is very rare. “De-novo” HD mutation refers to the situation where HD appears in a family without a history of the disease. This means that a new, spontaneous mutation occurred which was not inherited from either parent.'},
   
    {'question': 'What happens if both parents carry the HD gene?',
     'context': 'This is an extremely rare situation. If both of your parents carry an abnormal copy of the gene, your overall risk of inheriting the HD gene increases to 75%. '},
    
    {'question': 'Are there other diseases like HD?',
     'context': 'Yes, a few HD-like diseases (HDLD) have been described, although the genes responsible for these disorders are different from the one that causes HD. Moreover, the nature of these diseases and their symptoms are slightly different'},

    {'question': 'How is HD diagnosed?',
     'context': 'If you suspect that you have HD, you should consult an HD specialist (usually a neurologist) for diagnostic clinical and genetic testing. If you already show symptoms of HD, your doctor will make a diagnosis on the basis of your medical history and clinical findings. The results of this diagnosis are then checked by genetic tests (confirmatory testing). '},

    {'question': 'What is a predictive test?',
     'context': 'A predictive test is a genetic test to determine whether a person will develop a certain genetic disease. It is by definition performed in a pre-symptomatic stage, i.e. before any signs or symptoms of the disease appear.'},
         
    {'question': 'One of my parents was recently diagnosed with HD. Should I undergo predictive testing?',
     'context': 'Living with the knowledge that you are at risk can be very worrying. You may feel that you would prefer to know for certain whether or not you have the abnormal HD gene. At this stage, genetic counselling can be very helpful'},


     {'question': 'Where can I take the test?',
     'context': 'Genetic testing is only provided by genetics specialists or genetics clinics. You can ask your general practitioner to arrange an appointment for you.'},
    
    {'question': 'How is the genetic test performed?',
     'context': 'The DNA is extracted from blood cells and analysed in a specialised laboratory. Your affected parent’s blood may also be tested to check the original diagnosis of HD.'},
    
    {'question': 'What does the genetic test detect?',
     'context': 'The genetic test is a DNA test which determines the length of the CAG repeat in the HD gene and thus detects the mutation. The test can tell whether you carry the HD mutation, but it cannot tell you when the disease itself will start to develop.'},
    
    {'question': 'How are the genetic results interpreted?',
     'context': 'There are four different types of results: A result under 27 CAG repeats is unequivocally normal. A repeat length between 27-35 repeats is normal, but there is a small risk that the repeat may increase in future generations. Between 36-39 repeats the result is abnormal, but there is a chance that the person may be affected very late in life or even not at all. Over 40 repeats the gene is unequivocally abnormal.'},
    
    {'question': 'How reliable is the genetic test?',
     'context': 'HD was one of the first inherited genetic disorders for which an accurate genetic test could be performed. The results from the DNA analysis are usually double checked using two separate blood samples.'},
    
    {'question': 'Are the test results confidential?',
     'context': 'Yes, the test results are kept confidential and are only disclosed to another person with your written permission.'},

    {'question': 'Will my health insurance pay for predictive testing?',
     'context': 'You need to check with your insurance provider if they cover pre-symptomatic testing. However, before doing so, you should weigh the risks and benefits carefully. It might happen that an insurance company deny health coverage or cancel an existing policy when a person is tested HD positive.'},
    
    {'question': 'What is the prognosis if I am diagnosed with HD?',
     'context': 'In the long term, the diagnosis of HD is fatal. The average duration of HD from symptom onset until death is 15-20 years. However, this varies greatly among different individuals and can range from 2 to 43 years.'},
    
    {'question': 'Should I tell my children about HD in our family?',
     'context': 'Yes, but you should do it in an age-appropriate manner and in a language that the child can understand. Children need to hear about HD from their parents and not from someone else. Otherwise, they might think that the affected parent’s behaviour is due to alcoholism or drug use, or that the parent does not love them.'},
    
    {'question': 'When should I talk to my children about HD?',
     'context': 'As a rule, it is important to tell children about HD if a person in the family is showing symptoms. This prevents children from drawing wrong conclusions about the person’s behaviour.'},
    
    {'question': 'May minor children undergo genetic testing?',
     'context': 'In general, a minimum age of 18 is recommended, as it is hoped that at this age a person has the maturity needed to deal with the awareness of carrying the HD gene. However, in exceptional cases, it may be reasonable to perform the genetic test in children, for example if they show signs of juvenile HD or in pregnant women under the age of 18.'},
    
    {'question': 'One of my husband’s parents has HD and we are thinking about having children. What should we do?',
     'context': 'In this case, you should consider genetic counselling before starting a family. Your husband may undergo genetic testing to see if he carries the HD gene. If he does not carry the mutant gene, your children will not inherit the disease. If he does carry the HD gene, then each of your children will have a 50% risk of inheriting the HD gene.'},
    
    {'question': 'If I carry the HD gene, does this mean that I should not have children?',
     'context': 'Deciding whether to have children or not despite the risk of HD is a personal decision which only you and your spouse can make. We recommend that you do it with the guidance of a genetic counsellor. There are currently special genetic procedures available in some countries to minimise the risk. You should also consider that by the time your children grow up there might be a cure for HD.'},
    
    {'question': 'May I test my unborn child?',
     'context': 'The genetic techniques currently available allow testing of unborn children, known as prenatal (before birth) diagnosis. However, testing of unborn children needs to fulfil certain medical and legal criteria, which may be country-specific.'},
    
    {'question': 'How is prenatal diagnosis performed?',
     'context': 'There are two classical procedures of prenatal diagnosis: Amniocentesis (also called amniotic fluid test) is a procedure in which amniotic fluid containing cells of the unborn child is collected with a needle, usually after the 14th week of pregnancy. Collection of the sample from the chorionic villi (tissue of the placenta) can be done earlier (between 9 and 12 weeks of pregnancy), but is more risky to the unborn child.'},

    {'question': 'Can I test my unborn child without disclosing my own genetic status?',
     'context': 'Yes. The “exclusion test” compares the genetic pattern of the unborn child with the genetic pattern of the grandparents.'},
    
    {'question': 'Is it possible to conceive a child who does not carry the HD gene?',
     'context': 'Yes, preimplantation genetic diagnosis (PGD), also known as embryo screening, is a modern diagnostic procedure combined with in vitro fertilisation (IVF). The embryos are screened prior to implantation. Using this technique, only embryos inheriting normal copies of the gene are implanted into the womb.'},
    
    {'question': 'Is there a cure for HD?',
     'context': 'Unfortunately, there are currently no medications that are proven to effectively treat the underlying causes of HD. However, basic and clinical research has dramatically increased our knowledge of HD in the last years.'},
    
    {'question': 'Are there any treatments for HD?',
     'context': 'Although there is no cure for HD at the moment, some treatments do control the symptoms of the disease (symptomatic treatments) and improve quality of life. These are divided into pharmacological (drug) and non-pharmacological (non-drug) treatments.'},
    
    {'question': 'What are the most important treatable symptoms of HD?',
     'context': 'Chorea, bradykinesia, irritability, apathy, depression, anxiety and sleep disturbances have been reported as the most distressing problems of HD. There are different options for the pharmacological treatment of these symptoms.'},
    
    {'question': 'What medicines are used to treat the symptoms of HD?',
     'context': 'Certain antipsychotics (neuroleptics) for chorea and hyperkinesias; antidepressants for depression, apathy and other mood disorders; anxiolytic drugs for anxiety; and hypnotic drugs for sleep disturbances.'},
    
    {'question': 'Is there a special diet for HD?',
     'context': 'The benefit of a special diet rich in vitamins, coenzymes and other compounds (e.g. creatine, coenzyme Q10 and ethyl-EPA) for HD is much discussed but not clinically proven. In the later stages of the disease, weight loss can be a problem and a high-calorie diet may become necessary. Referral to a dietitian may be helpful.'},
    
    {'question': 'What does a positive HD result mean?',
     'context': 'A positive HD result may change your life in many different aspects, e.g. deciding whether to have children, planning for the future, rethinking priorities, negotiating appropriate housing, etc. It may also make mortgages, health and life insurances difficult.'},
    
    {'question': 'How will HD affect my day-to-day life?',
     'context': 'HD will gradually affect your ability to live an independent life. Working, social activities, and general daily activities will become increasingly difficult to perform. As the disease progresses, you may become more dependent on help and support from relatives, health and social care professionals.'},
    
    {'question': 'May I drive if I am an HD gene carrier?',
     'context': 'This can be a very sensitive issue. In some countries, you may have to inform the driver licensing authority if you have a medical condition that affects your ability to drive. People who are in the early stages of the disease are sometimes given licenses which can be reviewed on a regular basis.'},
    
    {'question': 'What are the most relevant impairments in daily life?',
     'context': 'Most HD patients and their carers perceive behavioural symptoms as more distressing than motor and cognitive impairments. These include depression, apathy, anxiety, irritability, and obsessive-compulsive behaviors.'},
    
    {'question': 'Are there any strategies on how to better cope with HD?',
     'context': 'A better understanding of the behavioural and cognitive impairments may help develop strategies to accommodate these changes and to maintain a warm relationship with people suffering from HD. You can also get important information and valuable advice from both HD specialists and lay organizations in your respective country.'}
<<<<<<< HEAD
   ]
=======
         ]
>>>>>>> origin/master

# Augment data with paraphrased questions
augmented_data = []
aug = naw.SynonymAug()

for item in fine_tuning_data:
    augmented_question = aug.augment(item['question'])
    augmented_data.append({'question': augmented_question, 'context': item['context'], 'answer': ''})

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
