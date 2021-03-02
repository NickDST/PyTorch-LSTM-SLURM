#Author : Nicholas Ho
## Date: June 24 2020
## Filename: LSTM RNN for Data Sequence

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import os
import numpy as np
import pandas as pd

from datetime import datetime



# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# Hyper-parameters
sequence_length = 9
input_size = 598984 ## inputs_i.size(2)
hidden_size = 300 ##unsure about this but ill keep it this way for now
num_layers = 2
num_classes = 598984 ## inputs_i.size(2)
batch_size = 300 ## this probably needs to change
num_epochs = 10
learning_rate = 0.01
embedding_dim = 128

## for loading the data sets
seeds = [21,7,8,9, 10, 15, 20, 30, 45, 60, 30, 11, 55, 39, 33, 24, 16]
data_root = '../multislice10.csv'
split = 0.98
max_one_hot = input_size



class newDataset(Dataset):
    def __init__(self, rootdir, partition, validation, split, max_one_hot, start_idx, random_seed, start_round):

        df = pd.read_csv(rootdir)

        ##I will shuffle it here because it makes it easier to resume checkpoint
#         df = pre_df.sample(frac=1, random_state = random_seed).reset_index(drop=True)
        cutoff = int(len(df) * (split))
        df = df.drop(df.columns[0], axis=1)


        if(partition == 'train'):
            self.df = df[:cutoff]

            if(validation == 'val'):

                self.df = self.df.sample(frac=1, random_state = random_seed).reset_index(drop=True) ## randomize the validation before
                print("=> Loading Validation Dataset : 1%")

                amount = len(self.df) - int(len(self.df) * 0.01)
#                 print(amount)
                self.df = self.df[amount:]
            else:
                print("=> Loading Normal Training Dataset")

        elif(partition == 'test'):
            self.df = df[cutoff:]
        else:
            print('no partition found')
            self.df = df

        ## shuffles after cutoff to ensure loading checkpoints ... I need to check this
        ## but here it shuffles if it isnt a validation set so that epochs can be consistent with the same training data (just shuffled)
        if(validation != 'val'):
            self.df = self.df.sample(frac=1, random_state = random_seed).reset_index(drop=True)

        if(start_round != 0):
            print("=> Starting at a new point...", start_round)
            self.df = self.df[start_round:]

        print("=> Start Index Found....", start_idx)

        self.df = self.df.iloc[start_idx:]

        self.max_one_hot = max_one_hot
        self.partition = partition

    def __len__(self):
        return len(self.df)

    def __getpartition__(self):
        return (self.partition)

    def __getitem__(self, idx):

        value = self.df.iloc[idx]
        x = value.to_numpy()
        a = torch.from_numpy(x)

        inputs = a[:9]
        features = a[9:]

#         one_hot_feature = torch.nn.functional.one_hot(features, self.max_one_hot)


        return inputs, features



def save_checkpoint(model, optimizer, epoch, curr_round, loss, checkpointFile = 'ckpt_model.csv'):

    ckpt_df = pd.read_csv(checkpointFile)
    ckpt_df = ckpt_df.drop(ckpt_df.columns[0], axis=1)

    if(len(ckpt_df) == 0):
        current_version = 0

    else:
        last_save = int(ckpt_df.iloc[-1]['version'])
        current_version = last_save + 1



    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

    model_name = "models/model_" + str(current_version) + ".pt"

    ## this is the row data saved for convenience
    new_row = {"version": current_version,
               "model_name" : model_name,
               "datetime" : dt_string,
               "round": curr_round,
               "loss": loss,
                "curr_epoch" : epoch}
    ckpt_df = ckpt_df.append(new_row, ignore_index=True)

    ## write to the file
    ckpt_df.to_csv(checkpointFile)

    ## this is the actual file we are saving
    state = {'epoch': epoch, 'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(), 'last_round' : curr_round + 1 }

    torch.save(state, model_name)




def load_checkpoint(model, optimizer, checkpointFile = 'ckpt_model.csv'):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    start_round = 0

    if os.path.isfile(checkpointFile):

        ##read in checkpoint csv
        ckpt_df = pd.read_csv(checkpointFile)

        ##checking if it is empty or not
        if(len(ckpt_df) > 0):

            filename = ckpt_df.iloc[-1]['model_name'] ##loading in the modelname

            if os.path.isfile(filename):
                print("=> loading checkpoint '{}'".format(filename))
                checkpoint = torch.load(filename)
#                 start_epoch = checkpoint['epoch']
#                 start_round = checkpoint['start_round']

                start_epoch = ckpt_df.iloc[-1]['curr_epoch']
                start_round = ckpt_df.iloc[-1]['round']

                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                model.eval()
                print("=> loaded checkpoint '{}' (epoch {} at round {})"
                          .format(filename, start_epoch, start_round))
            else:
                print("=> no checkpoint found at '{}'".format(filename))

        else:
            print("=> ckpt_model.csv is empty...continuing with new parameters")

    else:

        ckpt_df = pd.DataFrame(columns = ["version", "model_name", "datetime", "round", "loss", "curr_epoch"])
        ckpt_df.to_csv('ckpt_model.csv')


    return model, optimizer, start_epoch, start_round





# Recurrent neural network (many-to-one)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, embedding_dim):

        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        self.emb = nn.Embedding(input_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, num_classes) ##manual change
#         self.fc = nn.Linear(hidden_size, 100 * 9) ##manual change
#         self.fc2 = nn.Linear(100 * 9, num_classes) ## to allow for reshape ## batch size times sequence length

#         self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Set initial hidden and cell states
#         batch_size = x.size(0) ## the first value is the batch size


        embeds = self.emb(x) ## this converts it into the respective ## 2,9,150

#         print("embeds shape:", embeds.shape)


        h0 = torch.zeros(self.num_layers, embeds.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, embeds.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(embeds, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

#         out = out.reshape(out.size(0), out.size(1) * out.size(2))
#         out = self.fc(out)


        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])

        return out





print("=> Loading Model....")
model = RNN(input_size, hidden_size, num_layers, num_classes, embedding_dim).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.7)


print("=> Model Loaded ")





print("=> Checking Checkpoints....")

model, optimizer, start_epoch, start_round = load_checkpoint(model, optimizer, checkpointFile = 'ckpt_model.csv')
# start_epoch = 0
# start_round = 0


### just a case test here
#ill change this later
if(start_round > 73576):
    print("Issue!! Start round has been exceeded. Fixing Error")
    save_checkpoint(model, optimizer, epoch = start_epoch+1 , curr_round = 0, loss = 11111.1, checkpointFile = 'ckpt_model.csv')
    start_epoch = start_epoch+1
    start_round = 0


random_seed = seeds[int(start_epoch)]

print("=> Current Epoch: ", start_epoch, " and Random Seed: ", random_seed)


print("=> Splitting Data....")

train_dataset = newDataset(data_root, 'train', 'not_val' , split, max_one_hot , start_round, random_seed, start_round)

##uncomment this
# validation_dataset = newDataset(data_root, 'train', 'val' , split, max_one_hot , start_round, random_seed)


train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size , shuffle= False, num_workers = 0 )

# test_dataset = newDataset(data_root, 'test', split, max_one_hot , seed)

print("=> Finished Data Prep")




total_step = len(train_dataloader) + start_round+1
# for epoch in range(start_epoch, num_epochs): ## 2 epocs ## changed it to currently be like this for now

total_loss = 0
epoch = start_epoch

## we don't need this i dont think for what we're doing
# h, c = model.init_hidden(batch_size)

model.train()

if(epoch >= 4):
    print("=> Slowing learning rate")
    for g in optimizer.param_groups:
        g['lr'] = 0.001



# for g in optimizer.param_groups:
#     g['momentum'] = 0.7

for i, (attributes, features) in enumerate(train_dataloader, start = start_round+1): ##adsfasdfsfsdfdf

    #print("Attribute: ", attributes)
    #print("Feature: ", features)
    ## attributes >> this should give a size of [1,9]
#     attributes = attributes.reshape(-1, sequence_length, input_size)


#     print(attributes.shape)

    attributes = attributes.to(device) # (.float())
#   features = features
#     features = features.reshape(-1).to(device) ## this needs to be long

    features = features.reshape(-1).to(device) ## this needs to be long

#     features = features.reshape(batch_size, num_classes ).to(device)
    # Forward pass

    outputs = model(attributes)
    loss = criterion(outputs, features)

    # Backward and optimize
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

#     h = h.detach()

#     if(True):
#         print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch, num_epochs, i, total_step, loss.item()))

    total_loss += loss.item()

    if(i % 100 == 0):



        average_loss = total_loss / 100
        total_loss = 0

        print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch, num_epochs, i, total_step, average_loss))

        # Writing to file
        with open("logFile.txt", "a") as file1:
            # Writing data to a file
            file1.write('Epoch [{}/{}], Step [{}/{}],Avg Loss: {:.4f} \n'
                .format(epoch, num_epochs, i, total_step, average_loss))

#         torch.cuda.empty_cache()

    ##save checkpoint every 3000
    if(i % 20000 == 0 and i != 0):
        print("=> Saving Model Checkpoint....")
        save_checkpoint(model, optimizer, epoch, curr_round = i, loss = loss.item(), checkpointFile = 'ckpt_model.csv')

    ## didn't think I needed to uh add this part
    if( i >= total_step):
        print("=> One Epoch has finished")
        break

save_checkpoint(model, optimizer, epoch+1, curr_round = 0, loss = loss.item(), checkpointFile = 'ckpt_model.csv')
