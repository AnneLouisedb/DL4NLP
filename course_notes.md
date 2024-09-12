## Exam Preparation

### Week 1: Introduction to Natural Language Processing (NLP)

#### Key NLP Tasks
- Text categorization
- Document summarization
- Machine translation (50% of web pages are in languages other than English)
- Question answering

#### Pre-Deep Learning Approaches
- Support Vector Machines (SVM)
- Decision Trees
- Generative Bayesian Approaches
- Maximum Entropy Approaches

#### Neural Networks in NLP
**Advantages:**
- No feature engineering required

**Disadvantages:**
- Large amounts of training data needed
- Difficulty in tracing errors

#### Word-Based Models
**Feed-Forward Language Model (FFNN)**
- Language modeling
- Word representation (embedding)

**Disadvantages:**
- Restricted to fixed sentence length

#### Sequence Classification
**Convolutional Neural Networks (CNN)**
- Language modeling leading to summarization and translation
- Sentence classification
- Summarizes vectors in the top layer by pooling

#### Sequence Labeling
**Recurrent Neural Networks (RNN) Language Models**
- Summarization
- Sequence labeling (Part-of-Speech tagging, Named Entity Recognition)
- Machine translation

#### Encoder-Decoder Architecture
- Can use CNN or RNN for encoding
- Uses RNN for decoding
- Applications: Summarization, image captioning

#### Sequence-to-Sequence Modeling
**Neural Machine Translation**
- Uses RNN and Transformer architectures
- Transformers can perform Q&A and machine translation

#### Large Language Models
- Fine-tuning techniques

### Week 1.b

#### Intro to Language Modeling
- The primary task of a language model is to estimate the probability of a word sequence $$p(w_{1} \ldots w_{n})$$.
- Alternatively, given a history of words $$h$$, the model predicts the probability of the next word $$w$$ occurring: $$p(w \mid h)$$.
- Language modeling is a crucial component in many NLP tasks, including machine translation and speech recognition.

#### N-Gram Language Models
- N-gram models compute probabilities based on consecutive word sequences, utilizing the chain rule to decompose joint probabilities into conditional probabilities.
- The Markov assumption simplifies the model by assuming that the probability of a word depends only on the last $$n-1$$ words.

Count-based language models form the foundation of traditional approaches to language modeling. These models rely on the Markov assumption, which posits that the probability of a word depends only on a limited number of preceding words, leading to the concept of n-grams. N-gram models compute probabilities based on the frequency of word sequences in a corpus, with higher-order n-grams (e.g., trigrams, 4-grams) potentially capturing more context but also suffering from data sparsity.

#### Challenges in (Count based) Language Modeling
- Zero probabilities can occur when certain n-grams are not present in the training data, necessitating smoothing techniques to adjust probabilities.
- The size and precision of count-based models can be significant challenges. As the order of n-grams increases, the model size grows exponentially, leading to storage and computational issues.
- Semantic relationships: These models struggle to capture the semantic similarity between words like 'green' and 'blue'. They treat these words as entirely distinct entities, even though they are semantically related as colors.
- Morphological relationships: Count-based models don't inherently recognize the connection between morphologically related words such as 'bought' and 'buy'. These are treated as separate tokens, despite their shared root and meaning.

#### Smoothing Techniques
Language model smoothing is a crucial technique to address the zero-probability problem that arises when certain n-grams are not observed in the training data.
- Various smoothing methods address the problem of zero probabilities:
  - **Jelinek-Mercer Smoothing**: Weighted interpolation of conditional probabilities.
  - **Katz Smoothing**: Back-off to lower-order probabilities.
  - **Witten-Bell Smoothing**: Linear interpolation weighted by the number of contexts.
  - **Kneser-Ney Smoothing**: Weights lower-order probabilities based on context occurrences.
 

#### Probabilistic Neural Language Models (PNLMs) 
Represent a more modern approach to language modeling. These models use neural network architectures to learn distributed representations of words and capture more complex dependencies in language. The PNLM architecture typically involves embedding layers, hidden layers (capture the contextual dependencies between words in an n-gram), and output layers that predict the probability distribution of the next word.

- Each word in the vocabulary is mapped to a m-dimensional real-valued
vector.
- Trained with cross-entropy loss

Advanages:
- Limited memory needed
- Better perplexity
  
Practical challenges in implementing PNLMs include:
- Computationally expensive due to large vocabulary
- handling out-of-vocabulary words
- optimizing model performance for specific tasks or domains.





