# Summarization_Preprocessing
##Software Requirements

BERT(pip install bert-serving-client   and pip install bert-serving-server)
For further query visit https://github.com/hanxiao/bert-as-service

Spacy(pip install spacy)

Gensim(pip install gensim)

Download GoogleNewPretrained weight https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing unzip and put it in the same folder where you are putting this code file.



## Functions Description

* make_folder(i) :- This will create the folder with the document name as the folder name.

* original_file_copy(i) :- This will copy the original file to the folder created.

* sentence_length(i) :- This will make a csv file which has 3 columns as SentenceID , Sentence , Word Length.

* bert_embedding_csvfile(i) :- This will create a csv file of sentence embedding using BERT model. Size of the csv is fixed for everry sentence (n,92499). This file will have 92500 columns SentenceId and then embedding. Here n, is the number of sentences in the document file.

* bert_embedding_for_tokenized_sentence(i) :- This will create a csv file of average of all word embedding present in the sentence using BERT model using . Size of the csv is fixed for everry sentence (n,768). This file will have 769 columns SentenceId and then embedding where 768 depends on the bert model used.In this we are basically finding the BERT embedding for each and every word in the sentence . Here n, is the number of sentences in the document file.

* googlenewspretrainmodel_embedding(i):- This will create a csv file of average of all word embedding present in the sentence using Googlenews Pretrainned model. 

* cosine_similarity_bert(i) :-This will create a csv file of cosine similarity of each pair of sentence present in the document using bert_embedding . Size of the csv is fixed for everry sentence (n,n).
Here n, is the number of sentences in the document file.

* cosine_similarity_bert_for_tokenized_sentence(i) :- This will create a csv file of cosine similarity of each pair of sentence present in the document using bert_embedding_for_tokenized_sentence . Size of the csv is fixed for everry sentence (n,n). Here n, is the number of sentences in the document file.

* cosine_similarity_googlepretrainedmodel(i) :-  This will create a csv file of cosine similarity of each pair of sentence present in the document using        . Size of the csv is fixed for everry sentence (n,n). Here n, is the number of sentences in the document file.

* document_vector_bert(i) :- This will create a csv file of average of all sentence embedding present in the original csv of BERT embedding to create a document vector.

* document_vector_bert_tokenized(i):- This will create a csv file of average of all sentence embedding present in the original csv of Tokenized BERT embedding to create a document vector.

* document_vector_googlepretrainned(i):- This will create a csv file of average of all sentence embedding present in the original csv of Googlenews Pretrained model embedding to create a document vector.

* Bert_document_to_sentence_cosine_similarity(i) :- This will create a csv file of cosine similarity of each pair of sentence present in the document with the document vector formed by BERT embedding. Size of the csv is  (1,n). Here n, is the number of sentences in the document file.

* Tokenized_Bert_document_to_sentence_cosine_similarity(i) :- This will create a csv file of cosine similarity of each pair of sentence present in the document with the document vector formed by tokenized BERT embedding. Size of the csv is  (1,n). Here n, is the number of sentences in the document file. 

* wmd_distance(i):- This will create a csv file of Word Mover Distance of each pair of sentence present in the document with each sentence . Dimensions of the csv is  (n,n). Here n, is the number of sentences in the document file. 


## RUN 
python/python3 Preprocessing.py








