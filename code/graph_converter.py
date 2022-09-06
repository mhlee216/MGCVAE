import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Crippen import MolLogP, MolMR

def graph_features(mol, atom_labels, max_length=None):
    max_length = max_length if max_length is not None else mol.GetNumAtoms()
    features = np.array([[*[a.GetAtomicNum() == i for i in atom_labels]] for a in mol.GetAtoms()], dtype=np.int32)
    return np.vstack((features, np.zeros((max_length - features.shape[0], features.shape[1]))))

def feature_size(mol, atom_labels, max_length=None): 
    feature = graph_features(mol, atom_labels, max_length)
    feature = torch.cat([torch.tensor(feature), torch.zeros([max_length-feature.shape[0], feature.shape[1]])], 0)
    for i in range(feature.shape[0]):
        if 1 not in feature[i]:
            feature[i, 0] = 1
    return feature

def graph_adjacency(mol, atom_number, bond_encoder_m, connected=True):
    A = np.zeros(shape=(atom_number, atom_number), dtype=np.int32)
    begin, end = [b.GetBeginAtomIdx() for b in mol.GetBonds()], [b.GetEndAtomIdx() for b in mol.GetBonds()]
    bond_type = [bond_encoder_m[b.GetBondType()] for b in mol.GetBonds()]
    A[begin, end] = bond_type
    A[end, begin] = bond_type
    degree = np.sum(A[:mol.GetNumAtoms(), :mol.GetNumAtoms()], axis=-1)
    adj = A if connected and (degree > 0).all() else None
    for i in range(adj.shape[0]):
        adj[i, 0:i] = 0
    oh_list = []
    for i in range(adj.shape[0]):
        oh = np.zeros(shape=(atom_number, 5), dtype=np.int32)
        for j in range(adj.shape[1]):
            oh[j, adj[i][j]] = 1
        oh_list.append(torch.tensor(oh))
    return torch.cat([o for o in oh_list], 1)

def graph2mol(node_labels, adjacency, atom_decoder_m, bond_decoder_m, strict=False):
    mol = Chem.RWMol()
    for node_label in node_labels:
        mol.AddAtom(Chem.Atom(atom_decoder_m[node_label]))
    for start, end in zip(*np.nonzero(adjacency)):
        if start < end:
            mol.AddBond(int(start), int(end), bond_decoder_m[adjacency[start, end]])
    if strict:
        try:
            Chem.SanitizeMol(mol)
        except:
            mol = None
    return mol

def results(cvae, condition_1, condition_2, generate, z_dim, cond_dim, 
            size, atom_labels, row_dim, col_dim, atom_decoder_m, bond_decoder_m):
    with torch.no_grad():
        z = torch.randn(int(generate), z_dim).cuda()
        c1 = torch.zeros(int(generate), cond_dim).cuda()
        c1[:,condition_1] = 1
        c2 = torch.zeros(int(generate), cond_dim).cuda()
        c2[:,condition_2] = 1
        sample = cvae.decoder(z, c1, c2)
    smi = []
    logp = []
    mr = []
    for test_sample in sample.view(int(generate), 1, row_dim, col_dim).cpu():
        try:
            atom_num = test_sample[0][:, 0:1].max(dim=0).indices[0].item()+1
            if atom_num < size:
                test_sample = test_sample[0][:atom_num, :]
            else:
                test_sample = test_sample[0]
            atom_mat = test_sample[:, 1:len(atom_labels)+1]
            nodes_hard_max = torch.max(atom_mat, -1)[1]
            bond_mat = test_sample[:, len(atom_labels)+1:len(atom_labels)+1+5*atom_num]
            bond_mat = torch.tensor(np.array(bond_mat))
            bond_mats = []
            for i in range(0, bond_mat.shape[1], 5):
                B = np.zeros(shape=(atom_num, 5), dtype=np.int32)
                if i+6 > bond_mat.shape[1]:
                    bm = bond_mat[:,i:]
                    for n in range(bm.shape[0]):
                        b = bm[n]
                        B[n, torch.max(b, -1)[1]] = 1
                    bond_mats.append(B)
                else:
                    bm = bond_mat[:,i:i+5]
                    for n in range(bm.shape[0]):
                        b = bm[n]
                        B[n, torch.max(b, -1)[1]] = 1
                    bond_mats.append(B)
            edges_hard = torch.tensor(bond_mats)
            edges_hard_max = torch.max(edges_hard, -1)[1]
            mol = graph2mol(nodes_hard_max.numpy(), edges_hard_max.numpy(), 
                            atom_decoder_m, bond_decoder_m, strict=True)
            try:
                if '.' not in Chem.MolToSmiles(mol):
                    smi.append(Chem.MolToSmiles(mol))
                    logp.append(MolLogP(mol))
                    mr.append(MolMR(mol))
            except:
                continue
        except:
            continue
    cvae_df = pd.DataFrame({'SMILES':smi, 'C1':logp, 'C2':mr})
    return cvae_df
