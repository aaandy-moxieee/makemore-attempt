# %% [markdown]
# Buidling MakeMore MLP

# %%
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# %%
#Loading our data
words = open(r'dataset\names.txt',mode='r').read().splitlines()
words[:10]

# %%
#charset
chars = sorted(list(set(''.join(words))))
S_to_I = {s:i+1 for i,s in enumerate(chars)}
S_to_I['.'] = 0
I_to_S = {i:s for s,i in S_to_I.items()}

print(I_to_S)

# %% [markdown]
# Building the Dataset, and context size (length) and storing our input/index char (X) & our probable/next char (Y) in tensors.

# %%
block_size = 3
X, Y = [],[]
for w in words:
    #print(w)
    context = [0] * block_size
    for ch in w +'.':
        ix = S_to_I[ch]
        X.append(context)
        Y.append(ix)
        #print(''.join(I_to_S[i] for i in context), '----->', I_to_S[ix])
        context = context[1:] + [ix]
        
X = torch.tensor(X)
Y = torch.tensor(Y)

# %%


# %%
def build_dataset(words):
    block_size = 3
    X, Y = [],[]
    for w in words:
        #print(w)
        context = [0] * block_size
        for ch in w +'.':
            ix = S_to_I[ch]
            X.append(context)
            Y.append(ix)
            #print(''.join(I_to_S[i] for i in context), '----->', I_to_S[ix])
            context = context[1:] + [ix]
            
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X,Y

import random
random.seed(42)
random.shuffle(words)
n1 = int(0.8 *len(words))
n2 = int(0.9*len(words))

X_train, y_train = build_dataset(words[n1:])
X_valid, y_valid = build_dataset(words[n1:n2])
X_test , y_test = build_dataset(words[n2:])

# %%
len(X_train), len(X_test), len(X_valid)

# %%
len(y_train), len(y_test), len(y_valid)

# %% [markdown]
# Implementing our lookup & Embedding table (lookup C, init to randn) ----------

# %%
C = torch.randn([27,2])

# %% [markdown]
# F.one_hot(torch.tensor(3), num_classes=27).float()
# Another way to index in our lookup table using F.one_hot, which stretches out the input tensor and switches only the index at 3, when applying matrix mult. (1,27) @ (27,2) = (1,2) tensor which is == indexing in the lookup table.

# %%
emb = C[X]
emb.shape

# %% [markdown]
# Bulilding the hidden layer (tanh activation with 100 nodes as output from it)

# %%
w1 = torch.randn([6,100]) #created weights1 init to randn tensor with shape (3*2 from our embedding, and 100 outputs as per our hyperparameter)
b1 = torch.randn((100))

# %% [markdown]
# N.B Always check if tensors are broadcastable, in this case because we are adding b1 to the product of emb.view() & w1. We need to check that the product & b1 are broadcastable

# %%
#Tensor Sizes
# PRODUCT - (   N  , 100)
# BIAS    - (  '1' , 100)

# %% [markdown]
# They are broadcastable

# %%
#x*w + b
h = torch.tanh((emb.view([-1,6]) @ w1) + b1)
h

# %% [markdown]
# Creating the softmax layer, outputting probability of our possible charset

# %%
w2 = torch.randn([100,27]) #created weights2 init to randn tensor with shape (100 as outputs from prev layer tanh, and 27 outputs as per our 'possible' charset)
b2 = torch.randn((27))

logits = h @ w2 +b2
logits.shape

# %% [markdown]
# Creating loss function (Mean Log Likelihood)

# %%
counts = logits.exp()
prob = counts / counts.sum(dim=1, keepdim=True)

loss = -prob[torch.arange(16),Y].log().mean()
loss

# %% [markdown]
# Optimization Again!!!

# %%
g =  torch.Generator().manual_seed(2147483647)
C = torch.randn([27,10], generator=g)
w1 = torch.randn([30,300]) #created weights1 init to randn tensor with shape (3*2 from our embedding, and 100 outputs as per our hyperparameter)
b1 = torch.randn((300))
w2 = torch.randn([300,27]) #created weights2 init to randn tensor with shape (100 as outputs from prev layer tanh, and 27 outputs as per our 'possible' charset)
b2 = torch.randn((27))
params = [C, w1, b1, w2, b2] #created new params variable to hold all Parameters as a list, so we can calculate total parameters in our network

# %%
print(sum(p.nelement() for p in params))
for p in params:
    p.requires_grad = True

# %%
learn_exp = torch.linspace(-3, 0,steps=1000)
learn_step = 10**learn_exp

# %%
learn_used = []
steps = []
loss_acq = []

# %%
for i in range(100000):
    #Minibatch from dataset
    batch_size = torch.randint(0, X_train.shape[0], (64,))
    
    #Forward pass
    emb = C[X_train[batch_size]]
    h = torch.tanh((emb.view([-1,30]) @ w1) + b1)
    logits = h @ w2 +b2
    #counts = logits.exp()
    #prob = counts / counts.sum(dim=1, keepdim=True)
    #loss = -prob[torch.arange(16),Y].log().mean()
    loss = F.cross_entropy(logits, y_train[batch_size])
    
    #Backward pass
    for p in params:
        p.grad =None
    loss.backward()
    #learning rate adjust
    #lr = learn_step[i]
    #Update
    lr= 0.1 if i < 50000 else 0.01
    for p in params:
        p.data += -lr * p.grad
    #learning rate tracking
    #learn_used.append(learn_exp[i])
    steps.append(i)
    loss_acq.append(loss.log10().item())
        
print(loss.item())

# %%
plt.plot(steps,loss_acq)

# %% [markdown]
# Loss on entire Training set (no batches)

# %%
emb = C[X_train]
h = torch.tanh((emb.view([-1,30]) @ w1) + b1)
logits = h @ w2 +b2
#counts = logits.exp()
#prob = counts / counts.sum(dim=1, keepdim=True)
#loss = -prob[torch.arange(16),Y].log().mean()
loss = F.cross_entropy(logits, y_train)
print(loss.item())


# %% [markdown]
# Loss on Validation set

# %%
emb = C[X_valid]
h = torch.tanh((emb.view([-1,30]) @ w1) + b1)
logits = h @ w2 +b2
#counts = logits.exp()
#prob = counts / counts.sum(dim=1, keepdim=True)
#loss = -prob[torch.arange(16),Y].log().mean()
loss = F.cross_entropy(logits, y_valid)
print(loss.item())


# %%
plt.figure(figsize=(8,8))
plt.scatter(C[:,0].data,C[:,1].data, s=200)
for i in range (C.shape[0]):
    plt.text(C[i,0].item(), C[i,1].item(), I_to_S[i], ha= 'center', va= 'center', color='white')
    plt.grid('minor')

# %% [markdown]
# Sampling from the model

# %%
g = torch.Generator().manual_seed(2147483647)


for _ in range(20):
    
    out = []
    context = [0] * block_size
    while True:
        emb = C[torch.tensor([context])]
        h = torch.tanh((emb.view(1,-1) @ w1) + b1)
        logits = h @ w2 + b2
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break
        
    print(''.join(I_to_S[i] for i in out))

# %%



