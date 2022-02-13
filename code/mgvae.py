import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Crippen import MolLogP, MolMR
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import warnings
warnings.filterwarnings(action='ignore')
import argparse

from graph_converter import graph_features, feature_size, graph_adjacency, graph2mol, vae_results

parser = argparse.ArgumentParser(description='Small Molecular Graph Conditional Variational Autoencoder for Multi-objective Optimization (logP & Molar Refractivity)')
parser.add_argument('--data', type=int, default=80000, help='Sampling (default=80000)')
parser.add_argument('--size', type=int, default=10, help='molecule size (default=10)')
parser.add_argument('--dataset', type=str, default='../data/ZINC_logP_MR.csv', help="dataset path (default='../data/ZINC_logP_MR.csv')")
parser.add_argument('--batch', type=int, default=100, help='batch size (default=100)')
parser.add_argument('--epochs', type=int, default=1000, help='epoch (default=1000)')
parser.add_argument('--test', type=float, default=0.1, help='test set ratio (default=0.1)')
parser.add_argument('--lr', type=float, default=0.00005, help='learning rate (default=0.00005)')
parser.add_argument('--gen', type=int, default=10000, help='number of molecules to be generated (default=10000)')
parser.add_argument('--output', type=str, default='../results/generated', help="output files path (default='../results/generated')")
args = parser.parse_args()

print()
print('  ##########   ########## ####    ####      ########    ##########        ####    ########')
print(' ############ ############ ###    ####    ############ ############      ######     ####  ')
print('####     #### ###     #### ###    ####   ####      #### ##     ####      ######     ####  ')
print(' #####        #####       ####    ####  ####           #####            ###  ###    ####  ')
print('  ########     ########   ####    ####  ####             #######   #### ##    ###   ####  ')
print('     #######      ####### ####    ####  ####       ####    ####### #### #########   ####  ')
print('####     #### ###    ##### ###    ####   ####      #### #     #####   ####    ####  ####  ')
print(' ############ ############ ##########     ############ ############  ####      #### ####  ')
print('  #########    #########     ######         ########    #########   ####        #### #####')
print()

print()
print('Small Molecular Graph Variational Autoencoder')
print()
print('- Laboratory:')
print('Computational Science and Artificial Intelligence Lab')
print('School of Mechanical Engineering')
print('Soongsil Univ.')
print('Republic of Korea')
print('csailabssu.quv.kr')
print()
print('- Developer:')
print('mhlee216.github.io')
print()
print(f'- Sampling: {args.data}')
print(f'- Molecule size: {args.size}')
print(f'- Dataset: {args.dataset}')
print(f'- Batch size: {args.batch}')
print(f'- Epoch: {args.epochs}')
print(f'- Test set ratio: {args.test}')
print(f'- Learning rate: {args.lr}')
print(f'- Generated molecules: {args.gen}')
print(f'- Output path: {args.output}')
print()


class VAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(VAE, self).__init__()
        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        # decoder part
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)
    
    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h)
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add(mu)
    
    def decoder(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return torch.sigmoid(self.fc6(h))
    
    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, out_dim))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var

def loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, out_dim), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

def one_hot(labels, class_size): 
    targets = torch.zeros(labels.shape[0], class_size)
    for i, label in enumerate(labels):
        targets[i, round(label.item())] = 1
    return Variable(targets)

def train(epoch):
    vae.train()
    train_loss = 0
    for batch_idx, graph in enumerate(train_loader):
        graph = graph.cuda()
        optimizer.zero_grad()
        recon_batch, mu, log_var = vae(graph)
        loss = loss_function(recon_batch, graph, mu, log_var)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print('> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))
    return train_loss / len(train_loader.dataset)

def test():
    vae.eval()
    test_loss= 0
    with torch.no_grad():
        for batch_idx, graph in enumerate(test_loader):
            graph = graph.cuda()
            recon, mu, log_var = vae(graph)
            test_loss += loss_function(recon, graph, mu, log_var).item()
    test_loss /= len(test_loader.dataset)
    print('> Test set loss: {:.4f}'.format(test_loss))
    return test_loss


df = pd.read_csv(args.dataset)
df = df[df['Length'] <= args.size].reset_index(drop=True)
print('- Total data:', df.shape[0])
try:
    df = df.sample(n=args.data).reset_index(drop=True)
    print('- Sampled data:', df.shape[0])
except:
    print(f'Sampling error: Set the value of --data lower than {df.shape[0]}.')
    quit()
print()

smiles = df['SMILES'].tolist()
data = [Chem.MolFromSmiles(line) for line in smiles]

atom_labels = sorted(set([atom.GetAtomicNum() for mol in data for atom in mol.GetAtoms()] + [0]))
atom_encoder_m = {l: i for i, l in enumerate(atom_labels)}
atom_decoder_m = {i: l for i, l in enumerate(atom_labels)}

bond_labels = [Chem.rdchem.BondType.ZERO] + list(sorted(set(bond.GetBondType() for mol in data for bond in mol.GetBonds())))
bond_encoder_m = {l: i for i, l in enumerate(bond_labels)}
bond_decoder_m = {i: l for i, l in enumerate(bond_labels)}


print('Converting to graphs...')
data_list = []
atom_number = args.size
for i in range(len(data)):
    try:
        length = [[0] for i in range(args.size)]
        length[int(df['Length'].iloc[i])-1] = [1]
        length = torch.tensor(length)
        data_list.append(torch.cat([length, 
                                    feature_size(data[i], atom_labels, atom_number), 
                                    graph_adjacency(data[i], atom_number, bond_encoder_m)], 1).float())
    except:
        print('Error:', df['SMILES'].iloc[i])
        continue

train_list = []
for i in range(len(data_list)):
    train_list.append(np.array([np.array(data_list[i])]))

bs = args.batch
tr = 1-args.test
train_loader = torch.utils.data.DataLoader(dataset=train_list[:int(len(train_list)*tr)], batch_size=bs, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(dataset=train_list[int(len(train_list)*tr):], batch_size=bs, shuffle=True, drop_last=True)

print()
print('- Train set:', len(train_list[:int(len(train_list)*tr)]))
print('- Test set:', len(train_list[int(len(train_list)*tr):]))
print()

row_dim = train_list[0][0].shape[0]
col_dim = train_list[0][0].shape[1]
out_dim = row_dim*col_dim
z_dim = 128
vae = VAE(x_dim=out_dim, h_dim1=512, h_dim2=256, z_dim=z_dim)
if torch.cuda.is_available():
    vae.cuda()

optimizer = optim.Adam(vae.parameters(), lr=args.lr)

print('Training the model...')
train_loss_list = []
test_loss_list = []
for epoch in range(1, args.epochs+1):
    train_loss = train(epoch)
    train_loss_list.append(train_loss)
    test_loss = test()
    test_loss_list.append(test_loss)

print()
print('Generating molecules...')
vae_df = vae_results(vae, args.gen, z_dim, atom_number, atom_labels, 
                     row_dim, col_dim, atom_decoder_m, bond_decoder_m)
vae_df.to_csv(f'{args.output}.csv', index=False)
print(f'Saving {args.output}.csv ({vae_df.shape[0]})...')

print()
print('Done!')
print()
