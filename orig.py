#!/usr/bin/env python
# coding: utf-8

# # Neural Dependency Parsing
# **Authors**: Francesco Cazzaro, Universidad Politécnica de Cataluña, and Giorgio Satta, University of Padua

# ## Introduction
# 
# Transition-based dependency parsing is one of the most popular methods for implementing a dependency parsers.  We use here the **arc-standard** model, that has been presented in class. We augment the parser with neural machinery for contextual word embeddings and for choosing the most appropriate parser actions.  
# 
# We implement the following features:
# * LSTM representation for stack tokens
# * MLP for next transition classification, based on two top-most stack tokens and first token in the buffer
# * training under static oracle
# In order to keep the presentation at a simple level, we disregard arc labels in dependency trees. 
# 
# The reference paper is: 
# > Kiperwasser and Goldberg, Simple and Accurate Dependency Parsing Using Bidirectional LSTM Feature Representations  
# *Transactions of the Association for Computational Linguistics*, Volume 4, 2016.
# 
# Differently from this notebook, the original paper uses a model called **arc-hybrid**. 

# In[ ]:


#!pip install datasets  # huggingface library with dataset
#!pip install conllu    # aux library for processing CoNLL-U format


# In[ ]:


import torch
import torch.nn as nn
from functools import partial
from datasets import load_dataset


# ## Arc-standard
# 
# Recall that a **configuration** of the arc-standard parser is a triple of the form $( \sigma, \beta, A)$
# where:
# 
# * $\sigma$ is the stack;
# * $\beta$ is the input buffer;
# * $A$ is a set of arcs constructed so far.
# 
# We write $\sigma_i$, $i \geq 1$, for the $i$-th token in the stack; we also write $\beta_i$, $i \geq 1$, for the $i$-th token in the buffer. 
# 
# The parser can perform three types of **actions** (transitions):
# 
# * **shift**, which removes $\beta_1$ from the buffer and pushes it into the stack;
# * **left-arc**, which creates the arc $(\sigma_1 \rightarrow \sigma_2)$, and removes $\sigma_2$ from the stack;
# * **right-arc**, which creates the arc $(\sigma_2 \rightarrow \sigma_1)$, and removes $\sigma_1$ from the stack.
# 
# Let $w = w_0 w_1 \cdots w_{n}$ be the input sentence, with $w_0$ the special symbol `<ROOT>`.
# Stack and buffer are implemented as lists of integers, where `j` represents word $w_j$.  Top-most stack token is at the right-end of the list; first buffer token is at the left-end of the list. 
# Set $A$ is implemented as an array `arcs` of size $n+1$ such that if arc $(w_i \rightarrow w_j)$ is in $A$ then `arcs[j]=i`, and if $w_j$ is still missing its head node in the tree under construction, then `arcs[j]=-1`. We always have `arcs[0]=-1`.  We use this representation also for complete dependency trees.
# 

# In[ ]:


class ArcStandard:
  def __init__(self, sentence):
    self.sentence = sentence
    self.buffer = [i for i in range(len(self.sentence))]
    self.stack = []
    self.arcs = [-1 for _ in range(len(self.sentence))]

    # three shift moves to initialize the stack
    self.shift()
    self.shift()
    if len(self.sentence) > 2:
      self.shift()

  def shift(self):
    b1 = self.buffer[0]
    self.buffer = self.buffer[1:]
    self.stack.append(b1)

  def left_arc(self):
    o1 = self.stack.pop()
    o2 = self.stack.pop()
    self.arcs[o2] = o1
    self.stack.append(o1)
    if len(self.stack) < 2 and len(self.buffer) > 0:
      self.shift()

  def right_arc(self):
    o1 = self.stack.pop()
    o2 = self.stack.pop()
    self.arcs[o1] = o2
    self.stack.append(o2)
    if len(self.stack) < 2 and len(self.buffer) > 0:
      self.shift()

  def is_tree_final(self):
    return len(self.stack) == 1 and len(self.buffer) == 0

  def print_configuration(self):
    s = [self.sentence[i] for i in self.stack]
    b = [self.sentence[i] for i in self.buffer]
    print(s, b)
    print(self.arcs)




# ## Oracle
# 
# Recall that a **static oracle** maps parser configurations $c$ into  actions, and it does so by looking into the gold (reference) tree for the sentence at hand.  If $c$ does not contain any mistake, then the action provided by the oracle for $c$ is guaranted to be correct.  Furthermore, in cases where there is more than one correct action for $c$, the oracle always chooses a single action, called the **canonical** action. 
# 
# We use here the static oracle for the arc-standard parser that has been presented in class.  The oracle is based on the following conditions:
# * set $A$ in configuration $c$ does not contain any wrong dependency
# * left-arc has precedence over other actions, and can be done only if it constructs a gold dependency
# * right-arc has precedence over shift, and can be done only if it constructs a gold dependency and $\sigma_0$ has already collected all of its dependents
# * shift transition has lowest precedence, and can be done if the buffer is not empty
# 
# It is not difficult to see that the three actions above are mutually exclusive.

# In[ ]:


class Oracle:
  def __init__(self, parser, gold_tree):
    self.parser = parser
    self.gold = gold_tree

  def is_left_arc_gold(self):
    o1 = self.parser.stack[len(self.parser.stack)-1]
    o2 = self.parser.stack[len(self.parser.stack)-2]

    if self.gold[o2] == o1:
      return True

    return False

  def is_right_arc_gold(self):
    o1 = self.parser.stack[len(self.parser.stack)-1]
    o2 = self.parser.stack[len(self.parser.stack)-2]

    if self.gold[o1] != o2:
      return False
    
    # we only check missing dependents for sigma_1 to the right, since we assume A is correct and thus all left dependents for sigma_1 have already been collected
    for i in self.parser.buffer:  
      if self.gold[i] == o1:  
        return False

    return True
  
  def is_shift_gold(self):
    # because of the way we use the oracle (see later) this test could also be omitted
    if len(self.parser.buffer) == 0:
      return False
    
    if (self.is_left_arc_gold() or self.is_right_arc_gold()):
      return False
    
    return True

#    if not (self.is_left_arc_gold() or self.is_right_arc_gold()): # neither left-arc nor right-arc are gold transitions
#      return True
#
#    o1 = self.parser.stack[len(self.parser.stack)-1] # check for elements in the buffer that are dependents of sigma_1
#    for i in self.parser.buffer:
#      if self.gold[i] == o1:
#        return True
#
#    return False

# **Important**: the key `head` selects an array representing the gold dependency tree for the sentence at hand, in the format discussed above.  However, the integers in this array are represented as strings and need to be **converted** into integers.  We will have to do this conversion at several places below. 

# We now test the parser in **oracle mode** on our dataset.  This means that we drive the parser using our oracle, which always predicts the gold actions.  We also need to exclude non-projective trees since the arc-standard cannot parse these structures. 

# In[ ]:


def is_projective(tree):
  for i in range(len(tree)):
    if tree[i] == -1:
      continue
    left = min(i, tree[i])
    right = max(i, tree[i])

    for j in range(0, left):
      if tree[j] > left and tree[j] < right:
        return False
    for j in range(left+1, right):
      if tree[j] < left or tree[j] > right:
        return False
    for j in range(right+1, len(tree)):
      if tree[j] > left and tree[j] < right:
        return False
  
  return True


# ## Create training data and iterable dataloaders
# 
# Recall that to run the arc-standard parser we need a **classifier** that looks at some of the content of the current parser configuration and selects an approapriate action.  In order to train the classifier, we need to convert the gold trees in our treebank into several pairs of the form configuration/gold action.  This is what we do in this section.  
# 
# First of all, we need to preprocess the training set. We remove non-projective trees.  We also create dictionary of word/index pairs, to be used later when creating word embeddings.  Words that have less than three occurrences are not encoded and will later be mapped to special token `<unk>`.

# In[ ]:


# threshold is the minimum number of appearance for a token to be included in the embedding list
def create_dict(dataset, threshold=3):
  dic = {}  # dictionary of word counts
  for sample in dataset:
    for word in sample['tokens']:
      if word in dic:
        dic[word] += 1
      else:
        dic[word] = 1 

  map = {}  # dictionary of word/index pairs
  map["<pad>"] = 0
  map["<ROOT>"] = 1
  map["<unk>"] = 2

  next_indx = 3
  for word in dic.keys():
    if dic[word] >= threshold:
      map[word] = next_indx
      next_indx += 1

  return map


# Next function is used to create training data. 
# 
# For each sentence in the dataset, we use our oracle to compute the canonical action sequence leading to the gold tree.  We then pair configurations and canonical actions.  Since our neural classifier will look only into $\sigma_1$, $\sigma_2$ and $\beta_1$, we do not have to record the full parser configuration.   

# In[ ]:


def process_sample(sample, get_gold_path = False):
  sentence = ["<ROOT>"] + sample["tokens"]
  gold = [-1] + [int(i) for i in sample["head"]]  # heads in the gold tree are strings, converting to int

  enc_sentence = [emb_dictionary[word] if word in emb_dictionary else emb_dictionary["<unk>"] for word in sentence]

  # gold_path and gold_moves are parallel arrays whose elements refer to parsing steps
  gold_path = []   # record two topmost stack token and first buffer token for current step
  gold_moves = []  # oracle (canonical) move for current step: 0 is left, 1 right, 2 shift

  if get_gold_path:  # only for training
    parser = ArcStandard(sentence)
    oracle = Oracle(parser, gold)

    while not parser.is_tree_final():

      configuration = [parser.stack[len(parser.stack)-2], parser.stack[len(parser.stack)-1]]
      if len(parser.buffer) == 0:
        configuration.append(-1)
      else:
        configuration.append(parser.buffer[0])  
      gold_path.append(configuration)

      if oracle.is_left_arc_gold():  
        gold_moves.append(0)
        parser.left_arc()
      elif oracle.is_right_arc_gold():
        parser.right_arc()
        gold_moves.append(1)
      elif oracle.is_shift_gold():
        parser.shift()
        gold_moves.append(2)

  return enc_sentence, gold_path, gold_moves, gold


# Next function used to batch the training data. 

# In[ ]:


def prepare_batch(batch_data, get_gold_path=False):
  data = [process_sample(s, get_gold_path=get_gold_path) for s in batch_data]
  # sentences, paths, moves, trees are parallel arrays, each element refers to a sentence
  sentences = [s[0] for s in data]
  paths = [s[1] for s in data]
  moves = [s[2] for s in data]
  trees = [s[3] for s in data]
  return sentences, paths, moves, trees


# ## Create neural network model  
# 
# The main differences between the training program presented below and  Kiperwasser and Goldberg, 2016 are as follows:
# 
# * original model uses PoS_tags
# * original model also considers third top-most element of the stack
# * original model uses hinge loss and dynamic oracle / training
# 
# We are now ready to train our parser on the dataset.  We start with the definition of some parameters.
# 

# In[ ]:


EMBEDDING_SIZE = 100
LSTM_SIZE = 100
LSTM_LAYERS = 1
MLP_SIZE = 300
DROPOUT = 0.2
EPOCHS = 30
LR = 0.001   # learning rate


# Next, we create ...

# In[ ]:


class Net(nn.Module):

  def __init__(self, device):
    super(Net, self).__init__()
    self.device = device
    self.embeddings = nn.Embedding(len(emb_dictionary), EMBEDDING_SIZE, padding_idx=emb_dictionary["<pad>"])
    self.lstm = nn.LSTM(EMBEDDING_SIZE, LSTM_SIZE, num_layers = LSTM_LAYERS, bidirectional=True, dropout=DROPOUT)

    self.w1 = torch.nn.Linear(6*LSTM_SIZE, MLP_SIZE, bias=True)
    self.activation = torch.nn.Tanh()
    self.w2 = torch.nn.Linear(MLP_SIZE, 3, bias=True)
    self.softmax = torch.nn.Softmax(dim=-1)

    self.dropout = torch.nn.Dropout(DROPOUT)
  
  def forward(self, x, paths):
    x = [self.dropout(self.embeddings(torch.tensor(i).to(self.device))) for i in x]

    h = self.lstm_pass(x)

    mlp_input = self.get_mlp_input(paths, h)

    out = self.mlp(mlp_input)

    return out

  def lstm_pass(self, x):
    x = torch.nn.utils.rnn.pack_sequence(x, enforce_sorted=False)
    h, (h_0, c_0) = self.lstm(x)
    h, h_sizes = torch.nn.utils.rnn.pad_packed_sequence(h) # size h: (length_sentences, batch, output_hidden_units)
    return h

  def get_mlp_input(self, configurations, h):
    mlp_input = []
    zero_tensor = torch.zeros(2*LSTM_SIZE, requires_grad=False).to(self.device)
    for i in range(len(configurations)): #THIS IS QUITE UGLY, SEE IF THERE IS BETTER (and more efficient?) WAY TO DO IT
      for j in configurations[i]:
        mlp_input.append(torch.cat([zero_tensor if j[0]==-1 else h[j[0]][i], zero_tensor if j[1]==-1 else h[j[1]][i], zero_tensor if j[2]==-1 else h[j[2]][i]]))
    mlp_input = torch.stack(mlp_input).to(self.device)
    return mlp_input

  def mlp(self, x):
    return self.softmax(self.w2(self.dropout(self.activation(self.w1(self.dropout(x))))))

  def infere(self, x):

    parsers = [ArcStandard(i) for i in x]

    x = [self.embeddings(torch.tensor(i).to(self.device)) for i in x]

    h = self.lstm_pass(x)

    while not self.parsed_all(parsers):
      configurations = self.get_configurations(parsers)
      mlp_input = self.get_mlp_input(configurations, h)
      mlp_out = self.mlp(mlp_input)
      self.parse_step(parsers, mlp_out)

    return [parser.arcs for parser in parsers]

  def get_configurations(self, parsers):
    configurations = []

    for parser in parsers:
      if parser.is_tree_final():
        conf = [-1, -1, -1]
      else:
        conf = [parser.stack[len(parser.stack)-2], parser.stack[len(parser.stack)-1]]
        if len(parser.buffer) == 0:
          conf.append(-1)
        else:
          conf.append(parser.buffer[0])  
      configurations.append([conf])

    return configurations

  
  def parsed_all(self, parsers):
    for parser in parsers:
      if not parser.is_tree_final():
        return False
    return True

  #This is intuitively correct and no black swan has been observed but some kind of automated check should (MUST!) be run to be sure it has no troubling edge cases
  def parse_step(self, parsers, moves): #has inference code ever been not ugly? But we should look if it can be more good looking here.
    moves_argm = moves.argmax(-1)       #we are modeling all constraints here instead of the parser object. In this way lesson is clearer when explaining the parser
    for i in range(len(parsers)):
      if parsers[i].is_tree_final():
        continue
      else:
        if moves_argm[i] == 0:
          if parsers[i].stack[len(parsers[i].stack)-2] != 0:
            parsers[i].left_arc()
          else: #ALL OF THIS ELSE CAN BE REDUCED AS IF BUFFER>0 SHIFT, ELSE RIGHT ARC (i think!?!?!?! pretty sure though)
            if moves[i][1] > moves[i][2]:
              if len(parsers[i].buffer) == 0:
                parsers[i].right_arc()
              else:
                parsers[i].shift()
            else:
              if len(parsers[i].buffer) > 0:
                parsers[i].shift()
              else:
                parsers[i].right_arc()
        elif moves_argm[i] == 1:
          if parsers[i].stack[len(parsers[i].stack)-2] == 0 and len(parsers[i].buffer)>0:
            parsers[i].shift()
          else:
            parsers[i].right_arc()
        elif moves_argm[i] == 2:
          if len(parsers[i].buffer) > 0:
            parsers[i].shift()
          else:
            if moves[i][0] > moves[i][1]:
              if parsers[i].stack[len(parsers[i].stack)-2] != 0:
                parsers[i].left_arc()
              else:
                parsers[i].right_arc()
            else:
              parsers[i].right_arc()


# Train 

# In[ ]:


def evaluate(gold, preds): 
  total = 0
  correct = 0

  for g, p in zip(gold, preds):
    for i in range(1,len(g)):
      total += 1
      if g[i] == p[i]:
        correct += 1


  return correct/total


# In[ ]:


def train(model, dataloader, criterion, optimizer):
  model.train()
  total_loss = 0
  count = 0

  for batch in dataloader:
    optimizer.zero_grad()
    sentences, paths, moves, trees = batch

    out = model(sentences, paths)
    labels = torch.tensor(sum(moves, [])).to(device) #sum(moves, []) flatten the array
    loss = criterion(out, labels)

    count +=1
    total_loss += loss.item()

    loss.backward()
    optimizer.step()
  
  return total_loss/count

def validation(model, dataloader):
  model.eval()

  gold = []
  preds = []

  for batch in dataloader:
    sentences, paths, moves, trees = batch
    with torch.no_grad():
      pred = model.infere(sentences)

      gold += trees
      preds += pred
  
  return evaluate(gold, preds)

