import numpy as np
import torch
from torch.utils import data
import matplotlib.pyplot as plt

def func(x):
    # result = (20*x+3*x**2+0.1*x**3)*np.sin(x)*np.exp(-0.01*x)
    result = x**3 - 2*x**2 - 26*x + 100
    return (result)

train_samples = np.linspace(-5, 6, 3000)
train_data = np.reshape(train_samples, (3000, 1))
train_labels = func(train_data)

train_min = np.min(train_data)
train_max = np.max(train_data)
train_mean = np.mean(train_data)
train_std = np.var(train_data)

train_labels_min = np.min(train_labels)
train_labels_max = np.max(train_labels)
train_labels_mean = np.mean(train_labels)
train_labels_std = np.mean(train_labels)
# train_labels = (train_labels-train_labels_min)/(train_labels_max-train_labels_min)
train_labels = (train_labels-train_labels_mean)/train_labels_std

train_data = (train_data - train_mean)/train_std
# train_data = (train_data - train_min)/(train_max - train_min)
train_features = torch.tensor(train_data, dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.float32)

val_samples = np.linspace(-10, 10, 3000)
val_data = np.reshape(val_samples, (3000,1))
val_labels = func(val_data)
# val_mean = np.mean(val_data)
# val_std = np.var(val_data)
# val_data = (val_data-train_mean)/train_std
# val_data = (val_data - train_min)/(train_max - train_min)

val_data = (val_data - train_mean)/train_std

val_features = torch.tensor(val_data, dtype=torch.float32)
val_labels = torch.tensor(val_labels, dtype=torch.float32)

def load_array(data_arrays, batch_size, is_train=True): 
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 256
train_iter = load_array((train_features, train_labels), batch_size)

mlp= torch.nn.Sequential(
            torch.nn.Linear(1, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 1)
        )

loss = torch.nn.MSELoss()

trainer = torch.optim.Adam(mlp.parameters(), lr=0.0001)
loss_list = []
num_epochs = 300
for epoch in range(num_epochs):
    for X, y in train_iter:
        l = loss(mlp(X) ,y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(mlp(train_features), train_labels)
    loss_list.append(l.item())
    print(f'epoch {epoch + 1}, loss {l:f}')

with torch.no_grad():
    # pred = mlp(val_features)
    pred = mlp(val_features)*train_labels_std+train_labels_mean
    # l_val = np.sqrt((pred*(train_labels_max-train_labels_min)+train_labels_min- val_labels)**2)
    # l_val = np.sqrt((pred*train_labels_std+train_labels_mean)**2)
    l_val = np.sqrt((pred-val_labels)**2)

tl = train_labels.numpy()
tl = tl.squeeze()
pl = pred.numpy()
vl = val_labels.numpy()
vl = vl.squeeze()
ll = np.array(loss_list)
xl = np.array(range(300))
plt.subplot(221)
plt.plot(xl, ll)

plt.subplot(222)
plt.plot(val_samples, pl, 'r')
plt.plot(val_samples, vl, 'b')
plt.show()
