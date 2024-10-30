# load necessary libraries
import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import beta

# define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# map pitches
def map_pitch_type(pitch_type):
    fastballs = ['FF', 'FT', 'SI']
    breaking_balls = ['SL', 'CU', 'KC']
    off_speed = ['CH', 'FS']
    if pitch_type in fastballs:
        return 'FB'
    elif pitch_type in breaking_balls:
        return 'BB'
    elif pitch_type in off_speed:
        return 'OS'
    else:
        return 'Other'

# load and preprocessdata
data_path = "C:\\Users\\david\\OneDrive\\Desktop\\Misc\\Code\\Sports  Analytics\\Reds\\data.csv"
data = pd.read_csv(data_path)
data['PITCH_GROUP'] = data['PITCH_TYPE'].apply(map_pitch_type)
data = data[data['PITCH_GROUP'] != 'Other']

# encode pitch group
label_encoder = LabelEncoder()
data['PITCH_GROUP_ENCODED'] = label_encoder.fit_transform(data['PITCH_GROUP'])

# prepare sequence data
sequence_length = 10  
sequences = []
labels = []

for i in range(len(data) - sequence_length):
    sequence = data['PITCH_GROUP_ENCODED'].values[i:i + sequence_length]
    label = data['PITCH_GROUP_ENCODED'].values[i + sequence_length]
    sequences.append(sequence)
    labels.append(label)

# convert to numpy arrays first, then to tensor datasets for runtime issues
X = torch.tensor(np.array(sequences), dtype=torch.long).to(device)
y = torch.tensor(np.array(labels), dtype=torch.long).to(device)
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# define the LTSM model
class PitchMixLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PitchMixLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out

# initialize model, loss, and optimizer
input_size = 1
hidden_size = 64
output_size = len(label_encoder.classes_)
model = PitchMixLSTM(input_size, hidden_size, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# train the model
num_epochs = 10
for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        inputs = inputs.unsqueeze(-1).float()  
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# initialize bayesian priors using historical proportions
predictions_path = "C:\\Users\\david\\OneDrive\\Desktop\\Misc\\Code\\Sports  Analytics\\Reds\\predictions.csv"
predictions = pd.read_csv(predictions_path)
predictions['Proportion_FB'] = np.nan
predictions['Proportion_BB'] = np.nan
predictions['Proportion_OS'] = np.nan

# set initial bayesian priors based on past proportions
for _, row in predictions.iterrows():
    batter_id = row['BATTER_ID']
    batter_data = data[data['BATTER_ID'] == batter_id]
    fb_prior = len(batter_data[batter_data['PITCH_GROUP'] == 'FB']) + 1
    bb_prior = len(batter_data[batter_data['PITCH_GROUP'] == 'BB']) + 1
    os_prior = len(batter_data[batter_data['PITCH_GROUP'] == 'OS']) + 1
    total_prior = fb_prior + bb_prior + os_prior

    # apply bayesian updating for predictions
    predictions.loc[predictions['BATTER_ID'] == batter_id, 'Proportion_FB'] = beta(fb_prior, total_prior - fb_prior).mean()
    predictions.loc[predictions['BATTER_ID'] == batter_id, 'Proportion_BB'] = beta(bb_prior, total_prior - bb_prior).mean()
    predictions.loc[predictions['BATTER_ID'] == batter_id, 'Proportion_OS'] = beta(os_prior, total_prior - os_prior).mean()

# save predictions 
output_path = "C:\\Users\\david\\OneDrive\\Desktop\\Misc\\Code\\Sports  Analytics\\Reds\\predictions_output.csv"
predictions.to_csv(output_path, index=False)
print(f"Predictions saved with Bayesian-updated pitch mix proportions to {output_path}")
