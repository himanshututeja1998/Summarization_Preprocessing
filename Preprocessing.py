
# coding: utf-8

# In[20]:


import numpy as np
import pandas as pd
import os


# In[21]:


folders = os.listdir()
documents=[x for x in folders if x.split(".")[-1]=='csv']
print("NO. OF FILES   :  "+str(len(documents)))


# In[22]:


def make_folder(i):
    mypath =i.split(".")[0].split("_")[0]
    print("FOLDER NAME        "+mypath)
    if not os.path.isdir(mypath):
        os.makedirs(mypath)


# In[23]:


from shutil import copyfile
def original_file_copy(i):
    dst=i.split(".")[0].split("_")[0]+"/"+i
    copyfile(i, dst)


# In[24]:


def sentence_length(i):
    df=pd.read_csv(i, encoding='latin-1')
    df=df.values
    f=[]
    for ii in range(len(df)):
        #print(df[ii][1])
        #print(ii)
        f.append(len(df[ii][1].split(" ")))
        #print(len(df[ii][1].split(" ")))
    f1=np.asarray(f)
    f_new=f1.reshape((len(df),1))
    f_append = pd.DataFrame(f_new.tolist())
    df_new=pd.DataFrame(df)
    final=pd.concat([df_new, f_append],axis=1)    
    mypath=i.split(".")[0].split("_")[0]
    csv_name=str(mypath)+"/"+str(mypath)+"_Senetence_length_file.csv"
    final.to_csv(csv_name, index=False)
    


# In[ ]:


from bert_serving.client import BertClient
bert=BertClient(check_version=False)
def bert_embedding(sentence):
    embedding=[]
    filtered_sentence=''
    for i in sentence.split(" "):
        i.lower()
        i=i.replace(':','')
        i=i.replace(';','')
        i=i.replace(')','')
        i=i.replace('(','')
        i=i.replace('-','')
        i=i.replace('@','')
        i=i.replace('$','')
        i=i.replace('\n','')
        i=i.replace('\r','')
        i=i.replace('.','')
        i=i.replace('  ','')
        i=i.replace('','')
        
        if i is not "":
            filtered_sentence+=i+" "
    #print(filtered_sentence)
    try:
        embed=bert.encode([filtered_sentence])
    except:
        embed=[]
        pass
    return embed

######################################################For TOKENIZED SENTENCE###################################################


def bert_embedding_tokenized(sentence):
    embedding=[]
    filtered_sentence=''
    for i in sentence.split(" "):
        i.lower()
        i=i.replace(':','')
        i=i.replace(';','')
        i=i.replace(')','')
        i=i.replace('(','')
        i=i.replace('-','')
        i=i.replace('@','')
        i=i.replace('$','')
        i=i.replace('\n','')
        i=i.replace('\r','')
        i=i.replace('.','')
        i=i.replace('  ','')
        i=i.replace('','')
        
        if i is not "":
            filtered_sentence+=i+" "
    #print(filtered_sentence)
    try:
        filtered_sentence=filtered_sentence.split()
        embed=bert.encode([filtered_sentence],is_tokenized=True)
    except:
        embed=[]
        pass
    return embed
    


# In[ ]:


def bert_embedding_csvfile(i):
    data=pd.read_csv(i, encoding='latin-1')
    mypath=i.split(".")[0].split("_")[0]
    data=data.values
    data=np.asarray(data)
    final_vec=[]
    count=1
    max1=0
    labels=[]
    for ii in data:
        lst=[]
        labels.append(ii[0])
        b=bert_embedding(ii[1])
        if len(b)!=0:
            #lst.append(ii[0])
            out=np.hstack(b[0])
            max1=max(max1,len(out))
            for i in range(len(out)):
                lst.append(out[i])
            #lst.append(out[:])
            #lst.append(out)
            final_vec.append(lst)
            #print(count)
            count+=1
    #print(max1) 
    #print(len(data))
    #print(len(final_vec))
    equal_vec=np.zeros((len(data),92499))
    #f=final_vec[:][0]
    #print(f)
    f=np.asarray(labels)
    f.reshape((len(data),1))
    f=f.tolist()
    ff=pd.DataFrame(f)
    #print(final_vec)
    #final_vec=np.asarray(final_vec)
   # print(final_vec.shape)
    for kk in range(len(final_vec)):
        #print(final_vec[kk])
        for ll in range(len(final_vec[kk])):
            #print(final_vec[kk][ll])
            #break
            equal_vec[kk][ll]=final_vec[kk][ll]
        #break
    df = pd.DataFrame(equal_vec)
    final=pd.concat([ff, df],axis=1)
    
    csv_name=str(mypath)+"/"+str(mypath)+"_BERT_vector_Representation.csv"
    final.to_csv(csv_name, index=False)
    #with open("haha.txt","w") as f:
    	#f.write(str(final_vec))

        
#####################################################For Tokenized SENTENCE####################################################




def bert_embedding_for_tokenized_sentence(i):
    data=pd.read_csv(i, encoding='latin-1')
    mypath=i.split(".")[0].split("_")[0]
    data=data.values
    #print(len(data))
    data=np.asarray(data)
    final_vec=[]
    count=1
    max1=0
    labels=[]
    for ii in data:
        lst=[]
        labels.append(ii[0])
        b=bert_embedding_tokenized(ii[1])
        a=np.asarray(b)
        bert_representation=np.mean(a[0], axis=0)
        final_vec.append(bert_representation.tolist()[:])
        
        #print(a.shape)
        #max1=max(max1,len(b))
    #print(max1) 
    #print(len(data))
    #print(len(final_vec))
#     equal_vec=np.zeros((len(data),92499))
#     #f=final_vec[:][0]
#     #print(f)
    f=np.asarray(labels)
    f.reshape((len(data),1))
    f=f.tolist()
    ff=pd.DataFrame(f)
#     for kk in range(len(final_vec)):
#         for ll in range(len(final_vec[kk])):
#             equal_vec[kk][ll]=final_vec[kk][ll]
    df = pd.DataFrame(final_vec)
    final=pd.concat([ff, df],axis=1)
    
    csv_name=str(mypath)+"/"+str(mypath)+"_Tokenized_BERT_vector_Representation.csv"
    final.to_csv(csv_name, index=False)
#     #with open("haha.txt","w") as f:
#     	#f.write(str(final_vec))


# In[ ]:


import scipy.spatial.distance as distance
def cosine_similarity_bert(i):
    mypath=i.split(".")[0].split("_")[0]
    bert_csv=str(mypath)+"/"+str(mypath)+"_BERT_vector_Representation.csv"
    data=pd.read_csv(bert_csv, encoding='latin-1')
    data=pd.DataFrame(data)
    data1=data.values
    #print(type(data1))
    data1=data1.tolist()
    #print(type(data1))
    cosine=np.zeros((len(data1),len(data1)))
    for kk in range(len(data1)):
        for ll in range(kk,len(data1)):
            a=np.asarray(data1[kk][1:])
            b=np.asarray(data1[ll][1:])
#             print(len(a))
#             print(len(b))
#             print(type(a))
#             print(type(b))
#             print(a[1])
#             print(b[1])
#             print(type(a[1]))
#             print(type(b[1]))
            cosine[kk][ll] = 1 - distance.cosine(a, b)
            cosine[ll][kk] = cosine[kk][ll]
    cos=str(mypath)+"/"+str(mypath)+"_BERT_cosine.csv"
    cosine=cosine.tolist()
    df = pd.DataFrame(cosine)
    
    df.to_csv(cos, index=False)
    
    
##########################################TOKENIZED COSINE SIMILARITY##########################################################


def cosine_similarity_bert_for_tokenized_sentence(i):
    mypath=i.split(".")[0].split("_")[0]
    bert_csv=str(mypath)+"/"+str(mypath)+"_Tokenized_BERT_vector_Representation.csv"
    data=pd.read_csv(bert_csv, encoding='latin-1')
    data=pd.DataFrame(data)
    data1=data.values
    #print(type(data1))
    data1=data1.tolist()
    #print(type(data1))
    cosine=np.zeros((len(data1),len(data1)))
    for kk in range(len(data1)):
        for ll in range(kk,len(data1)):
            a=np.asarray(data1[kk][1:])
            b=np.asarray(data1[ll][1:])
#             print(len(a))
#             print(len(b))
#             print(type(a))
#             print(type(b))
#             print(a[1])
#             print(b[1])
#             print(type(a[1]))
#             print(type(b[1]))
            cosine[kk][ll] = 1 - distance.cosine(a, b)
            cosine[ll][kk] = cosine[kk][ll]
    cos=str(mypath)+"/"+str(mypath)+"_BERT_TOKENIZED_SENTENCE_cosine.csv"
    cosine=cosine.tolist()
    df = pd.DataFrame(cosine)
    
    df.to_csv(cos, index=False)
    
    
##########################################################GooglePretrained Model COSINE SIMILARITY##############################



def cosine_similarity_googlepretrainedmodel(i):
    mypath=i.split(".")[0].split("_")[0]
    bert_csv=str(mypath)+"/"+str(mypath)+"_GooglePreTrainedModel_vector_Representation.csv"
    data=pd.read_csv(bert_csv, encoding='latin-1')
    data=pd.DataFrame(data)
    data1=data.values
    #print(type(data1))
    data1=data1.tolist()
    #print(type(data1))
    cosine=np.zeros((len(data1),len(data1)))
    for kk in range(len(data1)):
        for ll in range(kk,len(data1)):
            a=np.asarray(data1[kk][1:])
            b=np.asarray(data1[ll][1:])
#             print(len(a))
#             print(len(b))
#             print(type(a))
#             print(type(b))
#             print(a[1])
#             print(b[1])
#             print(type(a[1]))
#             print(type(b[1]))
            cosine[kk][ll] = 1 - distance.cosine(a, b)
            cosine[ll][kk] = cosine[kk][ll]
    cos=str(mypath)+"/"+str(mypath)+"_GOOGLEPRETRAINED_SENTENCE_cosine.csv"
    cosine=cosine.tolist()
    df = pd.DataFrame(cosine)
    
    df.to_csv(cos, index=False)



# In[ ]:


def Bert_document_to_sentence_cosine_similarity(i):
    mypath=i.split(".")[0].split("_")[0]
    bert_csv=str(mypath)+"/"+str(mypath)+"_BERT_vector_Representation.csv"
    data=pd.read_csv(bert_csv, encoding='latin-1')
    data=pd.DataFrame(data)
    data1=data.values
    #print(type(data1))
    data1=data1.tolist()
    #print(type(data1))
    doc_csv=str(mypath)+"/"+str(mypath)+"_Document_vector_Bert_Representation.csv"
    doc_data=pd.read_csv(doc_csv, encoding='latin-1')
    doc_data=pd.DataFrame(doc_data)
    doc_data1=doc_data.values
    #print(type(data1))
    doc_data1=doc_data1.tolist()
    cosine=np.zeros((len(data1),1))
    for kk in range(len(data1)):
        a=np.asarray(data1[kk][1:])
        b=np.asarray(doc_data1[0][:])
#             print(len(a))
#             print(len(b))
#             print(type(a))
#             print(type(b))
#             print(a[1])
#             print(b[1])
#             print(type(a[1]))
#             print(type(b[1]))
        cosine[kk][0] = 1 - distance.cosine(a, b)
            
    cos=str(mypath)+"/"+str(mypath)+"_Document_To_sentence_BERT_cosine.csv"
    cosine=cosine.tolist()
    df = pd.DataFrame(cosine)
    
    df.to_csv(cos, index=False)
    
    
    
    
    
##############################################Tokenized_Bert_document_to_sentence_cosine_similarity#######################

def Tokenized_Bert_document_to_sentence_cosine_similarity(i):
    mypath=i.split(".")[0].split("_")[0]
    bert_csv=str(mypath)+"/"+str(mypath)+"_Tokenized_BERT_vector_Representation.csv"
    data=pd.read_csv(bert_csv, encoding='latin-1')
    data=pd.DataFrame(data)
    data1=data.values
    #print(type(data1))
    data1=data1.tolist()
    #print(type(data1))
    doc_csv=str(mypath)+"/"+str(mypath)+"_Document_vector_Tokenized_Bert_Representation.csv"
    doc_data=pd.read_csv(doc_csv, encoding='latin-1')
    doc_data=pd.DataFrame(doc_data)
    doc_data1=doc_data.values
    #print(type(data1))
    doc_data1=doc_data1.tolist()
    cosine=np.zeros((len(data1),1))
    for kk in range(len(data1)):
        a=np.asarray(data1[kk][1:])
        b=np.asarray(doc_data1[0][:])
#             print(len(a))
#             print(len(b))
#             print(type(a))
#             print(type(b))
#             print(a[1])
#             print(b[1])
#             print(type(a[1]))
#             print(type(b[1]))
        cosine[kk][0] = 1 - distance.cosine(a, b)
            
    cos=str(mypath)+"/"+str(mypath)+"_Document_To_sentence_Tokenized_BERT_cosine.csv"
    cosine=cosine.tolist()
    df = pd.DataFrame(cosine)
    
    df.to_csv(cos, index=False)
    
    
    


# In[13]:


import gensim
model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
import numpy as np
def word2vec_averaging(sentence):
    word2vec=[]
    length1=len(sentence.split(" "))
    for i in sentence.split(" "):
        i=i.lower()
        if i in model.vocab:
            
            word2vec.append(model[i])
            #print("present"+i)
        else:
            length1=length1-1
        
    word2vec=np.asarray(word2vec)
    #print("WORD2VEC : ")
    #print(word2vec)
    #print(length1)
    average=np.zeros((1,300))
    if length1>0:
        for j in range(300):
            summ=0.0
            for i in range(length1):
                summ+=word2vec[i][j]
            average[0][j]=summ/(float)(length1)
    #print("AVERAGE")
    average=np.array(average).tolist()
    return average


# ###################################################WMD DISTANCE#################################################################




from nltk import sent_tokenize, word_tokenize
import nltk
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
model.init_sims(replace=True)

def wmd_distance(i):
    data=pd.read_csv(i, encoding='latin-1')
    mypath=i.split(".")[0].split("_")[0]
    data=data.values
    #print(len(data))
    data=np.asarray(data)
    wmd=np.zeros((len(data),len(data)))
    for kk in range(len(data)):
        for ll in range(kk,len(data)):
            a=data[kk][1]
            b=data[ll][1]
            aa= a.lower().split()
            bb= b.lower().split()
            aaa= [w for w in aa if w not in stop_words]
            bbb= [w for w in bb if w not in stop_words]
            wmd[kk][ll] = model.wmdistance(aaa, bbb)
            wmd[ll][kk] = wmd[kk][ll]
    wmd_dis=str(mypath)+"/"+str(mypath)+"_WMD_DISTANCE.csv"
    wmd=wmd.tolist()
    df = pd.DataFrame(wmd)

    df.to_csv(wmd_dis, index=False)

    


# In[ ]:


def googlenewspretrainmodel_embedding(i):
    data=pd.read_csv(i, encoding='latin-1')
    mypath=i.split(".")[0].split("_")[0]
    data=data.values
    #print(len(data))
    data=np.asarray(data)
    final_vec=[]
    count=1
    max1=0
    labels=[]
    for ii in data:
        lst=[]
        labels.append(ii[0])
        b=word2vec_averaging(ii[1])
        a=np.asarray(b)
        google_representation=np.mean(a, axis=0)
        final_vec.append(google_representation.tolist()[:])
    f=np.asarray(labels)
    f.reshape((len(data),1))
    f=f.tolist()
    ff=pd.DataFrame(f)
    #     for kk in range(len(final_vec)):
    #         for ll in range(len(final_vec[kk])):
    #             equal_vec[kk][ll]=final_vec[kk][ll]
    df = pd.DataFrame(final_vec)
    final=pd.concat([ff, df],axis=1)

    csv_name=str(mypath)+"/"+str(mypath)+"_GooglePreTrainedModel_vector_Representation.csv"
    final.to_csv(csv_name, index=False)

        
    
    


# In[ ]:


import pandas as pd
import numpy as np
def document_vector_bert(i):
    
    mypath=i.split(".")[0].split("_")[0]
    bert_csv=str(mypath)+"/"+str(mypath)+"_BERT_vector_Representation.csv"
    data=pd.read_csv(bert_csv, encoding='latin-1')
    data1=pd.DataFrame(data)
    data=data1.values
    #print(len(data))
    document=[]
    for i in range(len(data)):
        document.append(data[i][1:])
    
    #print(document)
    document1=np.asarray(document)
    print(document1.shape)
    document_vector=np.mean(document1, axis=0)
    doc_vec=document_vector.reshape((1,92499))
    print(doc_vec.shape)
    #final=[]
    #for ii in range(92499):
    #    final.append(document_vector[ii])
        
    
    #print(document_vector.shape)
    document_name=str(mypath)+"/"+str(mypath)+"_Document_vector_Bert_Representation.csv"
    final=pd.DataFrame(doc_vec)
    final.to_csv(document_name, index=False)   
    
    
##########################################Document VECTOR for BERT TOKENIZED FILE###############################################



def document_vector_bert_tokenized(i):
    mypath=i.split(".")[0].split("_")[0]
    bert_csv=str(mypath)+"/"+str(mypath)+"_Tokenized_BERT_vector_Representation.csv"
    data=pd.read_csv(bert_csv, encoding='latin-1')
    data1=pd.DataFrame(data)
    data=data1.values
    #print(len(data))
    document=[]
    for i in range(len(data)):
        document.append(data[i][1:])
    
    #print(document)
    document1=np.asarray(document)
    print(document1.shape)
    document_vector=np.mean(document1, axis=0)
    doc_vec=document_vector.reshape((1,len(document_vector)))
    print(doc_vec.shape)
    #final=[]
    #for ii in range(92499):
    #    final.append(document_vector[ii])
        
    
    #print(document_vector.shape)
    document_name=str(mypath)+"/"+str(mypath)+"_Document_vector_Tokenized_Bert_Representation.csv"
    final=pd.DataFrame(doc_vec)
    final.to_csv(document_name, index=False)   
    
    
    
    
######################################DOCUMENT VECTOR FOR GOOGLEPRETRAINNED MODEL###############################################


def document_vector_googlepretrainned(i):
    mypath=i.split(".")[0].split("_")[0]
    bert_csv=str(mypath)+"/"+str(mypath)+"_GooglePreTrainedModel_vector_Representation.csv"
    data=pd.read_csv(bert_csv, encoding='latin-1')
    data1=pd.DataFrame(data)
    data=data1.values
    #print(len(data))
    document=[]
    for i in range(len(data)):
        document.append(data[i][1:])
    
    #print(document)
    document1=np.asarray(document)
    print(document1.shape)
    document_vector=np.mean(document1, axis=0)
    doc_vec=document_vector.reshape((1,len(document_vector)))
    print(doc_vec.shape)
    #final=[]
    #for ii in range(92499):
    #    final.append(document_vector[ii])
        
    
    #print(document_vector.shape)
    document_name=str(mypath)+"/"+str(mypath)+"_Document_vector_GooglePreTrainned_Representation.csv"
    final=pd.DataFrame(doc_vec)
    final.to_csv(document_name, index=False)   
    


# In[25]:


count=1
for i in documents:
    print(i)
    make_folder(i)
    original_file_copy(i)
    sentence_length(i)
    print(str(count)+"     Senetence Length File Creation++++++++++++++++++++++++++++++++++COMPLETED++++++++++++++++++++++++++++")
    




    bert_embedding_csvfile(i)
    print(str(count)+"      BERT EMBEDDING++++++++++++++++++++++++++++++++++COMPLETED++++++++++++++++++++++++++++")
    cosine_similarity_bert(i)
    print(str(count)+"      COSINE FOR SENTENCE BERT EMBEDDING++++++++++++++++++++++++++++++++++COMPLETED++++++++++++++++++++++++++++")
    document_vector_bert(i)
    print(str(count)+"      Document vector BERT EMBEDDING++++++++++++++++++++++++++++++++++COMPLETED++++++++++++++++++++++++++++")
    Bert_document_to_sentence_cosine_similarity(i)
    print(str(count)+"      Sentence to Document Cosine similarity BERT EMBEDDING++++++++++++++++++++++++++++++++++COMPLETED++++++++++++++++++++++++++++")
    




    bert_embedding_for_tokenized_sentence(i)
    print(str(count)+"         TOKENIZED BERT EMBEDDING++++++++++++++++++++++++++++++++++COMPLETED++++++++++++++++++++++++++++")
    cosine_similarity_bert_for_tokenized_sentence(i)
    print(str(count)+"        COSINE FOR TOKENIZED SENTENCE BERT EMBEDDING++++++++++++++++++++++++++++++++++COMPLETED++++++++++++++++++++++++++++")
    document_vector_bert_tokenized(i)
    print(str(count)+"      Document vector TOKENIZED BERT EMBEDDING++++++++++++++++++++++++++++++++++COMPLETED++++++++++++++++++++++++++++")
    Tokenized_Bert_document_to_sentence_cosine_similarity(i)
    print(str(count)+"      Sentence to Document Cosine similarity TOKENIZED BERT EMBEDDING++++++++++++++++++++++++++++++++++COMPLETED++++++++++++++++++++++++++++")
    
    
    
    googlenewspretrainmodel_embedding(i)
    print(str(count)+"      GOOGLE NEWS PRETRAINED MODEL EMBEDDING++++++++++++++++++++++++++++++++++COMPLETED++++++++++++++++++++++++++++")
    cosine_similarity_googlepretrainedmodel(i)
    print(str(count)+"        COSINE FOR GOOGLE NEWS PRETRAINED MODEL EMBEDDING++++++++++++++++++++++++++++++++++COMPLETED++++++++++++++++++++++++++++")
    document_vector_googlepretrainned(i)
    print(str(count)+"      Document vector FOR GooglePreTrained EMBEDDING++++++++++++++++++++++++++++++++++COMPLETED++++++++++++++++++++++++++++")
    
    wmd_distance(i)
    print(str(count)+"         WMD DISTANCE++++++++++++++++++++++++++++++++++COMPLETED++++++++++++++++++++++++++++")
    
    count+=1
    
    #break

