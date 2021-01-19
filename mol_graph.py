import torch
import numpy as np
import rdkit.Chem as Chem
import xml.etree.ElementTree as ET

from .mol_features import *


# Supress warnings from rdkit
from rdkit import rdBase
from rdkit import RDLogger
rdBase.DisableLog('rdApp.error')
lg = RDLogger.logger()
lg.setLevel(RDLogger.ERROR)


class Atom:
    def __init__(self, idx, rd_atom=None, degree=None, is_dummy=False):
        """Initialize the atom object to keep track of its attributes.

        Args:
            idx: The index of the atom in the original molecule.
            rd_atom: If provided the rdkit atom object, used to extract
                features.
        """
        self.idx = idx
        self.bonds = []
        self.is_dummy = is_dummy

        if is_dummy:
            self.symbol = '*'  # Default wildcard/dummy symbol

        if rd_atom is not None:
            self.symbol = rd_atom
            #self.fc = rd_atom.GetFormalCharge()
            self.degree = degree
    def add_bond(self, bond):
        self.bonds.append(bond)  # Includes all the incoming bond indices


class Bond:
    def __init__(self, idx, out_atom_idx, in_atom_idx):#, rd_bond=None):
        """Initialize the bond object to keep track of its attributes."""
        self.idx = idx
        self.out_atom_idx = out_atom_idx
        self.in_atom_idx = in_atom_idx
        self.bond_type = Chem.rdchem.BondType.SINGLE

class Molecule:
    def __init__(self, atoms, bonds):
        self.atoms = atoms
        self.bonds = bonds

    def get_bond(self, atom_1, atom_2):
        # If bond does not exist between atom_1 and atom_2, return None
        for bond in self.atoms[atom_1].bonds:
            if atom_2 == bond.out_atom_idx or atom_2 == bond.in_atom_idx:
                return bond
        return None


class MolGraph:
    def __init__(self, smiles_list):
        """Initialize the molecular graph inputs for the smiles list.

        Args:
            smiles_list: The input smiles strings in a list
        """
        self.smiles_list = smiles_list

        self.mols = []  # Molecule objects list
        self.scope = []  # Tuples of (st, le) for atoms for mols
        self.rd_mols = []

        self._parse_molecules(smiles_list)
        self.n_mols = len(self.mols)


    def _parse_molecules(self, smile):
        """Turn the smiles into atom and bonds through rdkit.

        Every bond is recorded as two directional bonds, and for each atom,
            keep track of all the incoming bonds, since these are necessary for
            aggregating the final atom feature output in the conv net.

        Args:
            smiles_list: A list of input smiles strings. Assumes that the given
                strings are valid.
            max_atoms: If provided, truncate graphs to this size.
        """
        print('enter _parse_molecules')
        a_offset = 0  # atom offset

        #for smile in smiles_list:
        for smiles in smile:
            rd_mol = ET.parse('data/fp/' + smiles)
            root = rd_mol.getroot()
            self.rd_mols.append(rd_mol)
            
            mol_atoms = []
            for elem in root[0].findall('node'):
                atom_idx = int(elem.attrib['id'][-1]) - 1
                r_atom = str(float(elem[0][0].text) + float(elem[1][0].text))
                deg = 0
                for elem in root[0].findall('edge'):
                    if int(elem.attrib['from'][-1]) - 1 == atom_idx:
                        deg += 1
                for elem in root[0].findall('edge'):
                    if int(elem.attrib['to'][-1]) - 1 == atom_idx:
                        deg += 1
                mol_atoms.append(Atom(idx = atom_idx, rd_atom = r_atom, degree = deg))
                
            mol_bonds = []
            for elem in root[0].findall('edge'):
                atom_1_idx = int(elem.attrib['from'][-1]) - 1
                atom_2_idx = int(elem.attrib['to'][-1]) - 1
                
                bond_idx = len(mol_bonds)
                new_bond = Bond(bond_idx, atom_1_idx, atom_2_idx)
                mol_bonds.append(new_bond)
                mol_atoms[atom_2_idx].add_bond(new_bond)
                
                new_bond = Bond(bond_idx, atom_2_idx, atom_1_idx)
                mol_bonds.append(new_bond)
                mol_atoms[atom_1_idx].add_bond(new_bond)
                
            new_mol = Molecule(mol_atoms, mol_bonds)
            self.mols.append(new_mol)

            self.scope.append((a_offset, len(mol_atoms)))
            a_offset += len(mol_atoms)

    def get_mol_sz(self, device='cpu'):
        mol_sizes = []

        for st, le in self.scope:
            mol_sizes.append(le)

        return torch.tensor(mol_sizes, device=device).float()


    def get_graph_inputs(self, device='cpu', output_tensors=True):
        """Constructs the graph inputs for the conv net.

        Returns:
            A tuple of tensors/numpy arrays that contains the input to the GCN.
        """
        n_atom_feats = N_ATOM_FEATS
        n_bond_feats = N_BOND_FEATS
        max_neighbors = MAX_NEIGHBORS

        # The feature matrices for the atoms and bonds
        fatoms = []
        fbonds = [np.zeros(n_atom_feats + n_bond_feats)]  # Zero padded
        #print(np.zeros(n_atom_feats + n_bond_feats).shape)
        # The graph matrices for aggregation in conv net
        agraph = []
        bgraph = [np.zeros([1, max_neighbors])]  # Zero padded
        b_offset = 1  # Account for padding

        for mol_idx, mol in enumerate(self.mols):
            atoms, bonds = mol.atoms, mol.bonds
            cur_agraph = np.zeros([len(atoms), max_neighbors])
            cur_bgraph = np.zeros([len(bonds), max_neighbors])

            for atom_idx, atom in enumerate(atoms):
                atom_features = get_atom_features(atom)
                fatoms.append(atom_features)
                for nei_idx, bond in enumerate(atom.bonds):
                    cur_agraph[int(atom.idx), int(nei_idx)] = bond.idx + b_offset
            for bond in bonds:
                out_atom = atoms[bond.out_atom_idx]
                #print(get_atom_features(out_atom).shape)
                #print(get_bond_features(bond).shape)
                bond_features = np.concatenate([
                    get_atom_features(out_atom),
                    get_bond_features(bond)], axis=0)
                #print(bond_features.shape)
                #print(' ')
                fbonds.append(bond_features)
                for i, in_bond in enumerate(out_atom.bonds):
                    if bonds[in_bond.idx].out_atom_idx != bond.in_atom_idx:
                        cur_bgraph[int(bond.idx), int(i)] = in_bond.idx + b_offset

            agraph.append(cur_agraph)
            bgraph.append(cur_bgraph)
            b_offset += len(bonds)
        
        #fb = [np.zeros(len(fbonds[1]))] + fbonds[1:]
        #print(fbonds)
        fatoms = np.stack(fatoms, axis=0)
        fbonds = np.stack(fbonds, axis=0)
        agraph = np.concatenate(agraph, axis=0)
        bgraph = np.concatenate(bgraph, axis=0)

        if output_tensors:
            fatoms = torch.tensor(fatoms, device=device).float()
            fbonds = torch.tensor(fbonds, device=device).float()
            agraph = torch.tensor(agraph, device=device).long()
            bgraph = torch.tensor(bgraph, device=device).long()

        graph_inputs = [fatoms, fbonds, agraph, bgraph]
        print('get_graph_inputs is ok')
        return (graph_inputs, self.scope)
