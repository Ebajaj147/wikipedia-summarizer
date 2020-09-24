import wikipedia
import nltk
import string
nltk.download('punkt')
import csv
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from itertools import count


def get_nth_index(xx, stt, nn):
    #cc = count()
    #return next(i for i, j in enumerate(xx) if j==stt and next(cc) == nn-1)
    c=0
    c1=0
    for aa in xx:
        c1=c1+1
        if aa==stt:
            c=c+1
            if c==nn:
                return c1-1






term=input("Enter the search term:")
#print(type(term))
term=term.lower()
st=wikipedia.page(term).content

#st=wikipedia.summary(term, sentences=1000)
st=st.lower()


print("\n\n Text Extracted from Wikipedia: \n\n")

print(st)
#st="Machine learning (ML) is the machine scientific study of algorithms and statistical models that computer systems use to perform a specific task without using explicit instructions, relying on patterns and inference instead. It is seen as a subset of artificial intelligence. Machine learning algorithms build a mathematical model based on sample data, known as training data, in order to make predictions or decisions without being explicitly programmed to perform the task. Machine learning algorithms are used in a wide variety of applications, such as email filtering and computer vision, where it is difficult or infeasible to develop a conventional algorithm for effectively performing the task."
#st=st.lower()
#st.encode('ascii','ignore')
#print(type(st))

sentence_data = st
sent = nltk.sent_tokenize(sentence_data)

print("\n\n Sentece Boundary Identification: \n\n")
print (sent)
sent.insert(0,'sentence')

#print (type(sent))

#sentences file(og)
with open("sentences.csv", 'w', newline='', encoding="utf-8") as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    for s in sent:
        ll=[]
        ll.append(s)
        wr.writerow(ll)

sent.remove('sentence')

l=[]

for i in sent:

    word_data = i
    nltk_tokens = nltk.word_tokenize(word_data)
    l.append(nltk_tokens)






with open("temp.csv", "w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(l)

with open('temp.csv', 'r', encoding="utf-8") as f:
    reader = csv.reader(f)
    nlist = list(reader)


stop_words = set(stopwords.words('english'))
n1list=[]
for x in nlist:
    filtered_sentence = []
    for w in x:
        if w not in stop_words:
            filtered_sentence.append(w)
    x=filtered_sentence
    str1=" "
    str1=str1.join(x)
    #print(str1)
    str1=str1.translate(str.maketrans('','',string.punctuation))
    #print(str1)

    n1list.append(str1.split())


#print("\n\n")
#print(n1list)

with open("no_punctuation.csv", "w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(n1list)

# joining search words

with open("tokenized.csv", "w", newline='', encoding="utf-8") as f:

    writer = csv.writer(f)
    writer.writerows(n1list)


#print("n1list\n\n", n1list)
#sentences file(post stop removal)
sent1=[]
for d in n1list:
    x=" ".join(d)
    sent1.append(x)

#print("sent1:\n\n", sent1)
sent1.insert(0,'sentence')

with open("sentences1.csv", 'w', newline='', encoding="utf-8") as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    for s in sent1:
        ll=[]
        ll.append(s)
        wr.writerow(ll)

sent1.remove('sentence')


search=term.split()
nll=[]
if len(search)>1:

    for x in n1list:
        #y=x
        y=x[:]

        there=1
        while there==1:
            for t in search:
                if t in x:
                    continue
                else:
                    there=0
                    break


            if there==1:
                cno=x.count(search[0])
                #print(cno)
                for d in range(1,cno+1):
                    #print("r")
                    s1index = get_nth_index(x, search[0], d)
                    #print(s1index)
                    there=1
                    for i in range(1,len(search)):
                        #print(x.index(search[i]))
                        #print(s1index+i)
                        if get_nth_index(x, search[i], d)==s1index+i:
                            continue
                        else:
                            there=0
                            break

                    if there==1:
                        #print(y)
                        sindex = y.index(search[0])
                        y[sindex]=term
                        for i in range(1, len(search)):
                            del y[sindex+i]
                        there=0
                        #print(y)
                        #print(x)
        nll.append(y)



    #print("\n\n")
    #print(n1list)
    with open("tokenized.csv", "w", newline='', encoding="utf-8") as f:

        writer = csv.writer(f)
        writer.writerows(nll)



print("\n\nTokenization (Removal of punctuation and stop words): \n\n")

if len(search)>1:
    print(nll)
else:
    print(n1list)


print("\n\n")




#BEGIN LEMMATIZATIOn


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

lemmatizer = WordNetLemmatizer()
#word = 'was'
#print(lemmatizer.lemmatize(word, get_wordnet_pos(word)))


with open('tokenized.csv', 'r', encoding="utf-8") as f:
    reader = csv.reader(f)
    leml = list(reader)

#print(leml)

nleml=[]
for z in leml:
    l=[]
    for j in z:
        word=j
        #print(j)
        j=lemmatizer.lemmatize(word, get_wordnet_pos(word))
        l.append(j)
    nleml.append(l)


#print(nleml)

with open("lemmatized.csv", "w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(nleml)

print("\n\nAfter Lemmatization: \n\n")
print(nleml)
print("\n\n")

print("\n\n After POS Tagging: \n\n")

#tagged = nltk.pos_tag(nleml[0])
#print type(tagged[0])
#print(tagged[0][1])

#POS TAGGING

posl=[]
for d in nleml:
    tagged = nltk.pos_tag(d)
    posl.append(tagged)


print(posl)

with open("postagged.csv", "w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(posl)


print("\n\nNOW RUN KG_Builder.py File to Obtain Knowledge Graph!")

