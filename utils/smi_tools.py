import numpy as np
import pandas as pd
import argparse
import os
import re
import random
import textdistance
import multiprocessing
from tqdm import tqdm

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def clear_map_canonical_smiles(smi, canonical=True, root=-1):
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        for atom in mol.GetAtoms():
            if atom.HasProp('molAtomMapNumber'):
                atom.ClearProp('molAtomMapNumber')
        # re-traverse the mol from the input root
        return Chem.MolToSmiles(mol, isomericSmiles=True, rootedAtAtom=root, canonical=canonical)
    else:
        return smi

def get_cano_map_number(smi, root=-1):
    atommap_mol = Chem.MolFromSmiles(smi)
    canonical_mol = Chem.MolFromSmiles(clear_map_canonical_smiles(smi, root=root))
    # find the matching substructure and create the local mapping
    cano2atommapIdx = atommap_mol.GetSubstructMatch(canonical_mol)  # dict? {"order id in canonical_mol" : "id in atommap_mol"}
    correct_mapped = [canonical_mol.GetAtomWithIdx(i).GetSymbol() == atommap_mol.GetAtomWithIdx(index).GetSymbol() for
                      i, index in enumerate(cano2atommapIdx)]
    atom_number = len(canonical_mol.GetAtoms())
    # can't find mapping relationship from atommap_mol to cano_mol? reverse the mapping.
    if np.sum(correct_mapped) < atom_number or len(cano2atommapIdx) < atom_number:
        cano2atommapIdx = [0] * atom_number
        atommap2canoIdx = canonical_mol.GetSubstructMatch(atommap_mol)      # {"order id in atommap_mol" : "id in canonical_moll"}
        if len(atommap2canoIdx) != atom_number:
            return None         # wrong mapping, even violate the traverse rule ?
        for i, index in enumerate(atommap2canoIdx):
            cano2atommapIdx[index] = i      # cano_id -> atommap_id
    # indices of atoms are the same as the ones of array
    id2atommap = [atom.GetAtomMapNum() for atom in atommap_mol.GetAtoms()]

    # cano_id -> atommap_id -> atomma_map_num
    return [id2atommap[cano2atommapIdx[i]] for i in range(atom_number)]


def canonical_smiles_with_am(smi):
    """Canonicalize a SMILES with atom mapping"""
    atomIdx2am, pivot2atomIdx = {}, {}
    mol = Chem.MolFromSmiles(smi)
    atom_ordering = []
    for atom in mol.GetAtoms():
        if atom.HasProp('molAtomMapNumber'):
            atomIdx2am[atom.GetIdx()] = atom.GetProp('molAtomMapNumber')
            atom.ClearProp('molAtomMapNumber')
        else:
            atomIdx2am[atom.GetIdx()] = '0'
        atom_ordering.append(atom.GetIdx())
    unmapped_smi = Chem.MolFragmentToSmiles(mol, atomsToUse=atom_ordering, canonical=False)
    mol = Chem.MolFromSmiles(unmapped_smi)
    cano_atom_ordering = list(Chem.CanonicalRankAtoms(mol))
    for i, j in enumerate(cano_atom_ordering):
        pivot2atomIdx[j + 1] = i
        mol.GetAtomWithIdx(i).SetIntProp('molAtomMapNumber', j + 1)
    new_tokens = []
    for token in smi_tokenizer(Chem.MolToSmiles(mol)):
        if re.match('.*:([0-9]+)]', token):
            pivot = re.match('.*(:[0-9]+])', token).group(1)
            token = token.replace(pivot, ':{}]'.format(atomIdx2am[pivot2atomIdx[int(pivot[1:-1])]]))
        new_tokens.append(token)
    canonical_smi = ''.join(new_tokens)
    # canonical reactants order
    if '.' in canonical_smi:
        canonical_smi_list = canonical_smi.split('.')
        canonical_smi_list = sorted(canonical_smi_list, key=lambda x: (len(x), x))
        canonical_smi = '.'.join(canonical_smi_list)
    return canonical_smi


"""
Root-aligned augmentation
copied from https://github.com/otori-bird/retrosynthesis/blob/main/preprocessing/generate_PtoR_data.py
"""

# choose root id by atom_map_num
def get_root_id(mol,root_map_number):
    root = -1
    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetAtomMapNum() == root_map_number:
            root = i
            break
    return root


"""multiprocess"""
def preprocess(save_dir, reactants, products,set_name, augmentation=1, reaction_types=None,root_aligned=True,character=False, processes=-1):
    """
    preprocess reaction data to extract graph adjacency matrix and features
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data = [{
        "reactant": i,
        "product": j,
        "augmentation": augmentation,
        "root_aligned": root_aligned,
    } for i, j in zip(reactants, products)]
    src_data = []
    tgt_data = []
    skip_dict = {
        'invalid_p': 0,
        'invalid_r': 0,
        'small_p': 0,
        'small_r': 0,
        'error_mapping': 0,
        'error_mapping_p': 0,
        'empty_p': 0,
        'empty_r': 0,
    }
    processes = multiprocessing.cpu_count() if processes < 0 else processes
    pool = multiprocessing.Pool(processes=processes)
    results = pool.map(func=multi_process, iterable=data)
    pool.close()
    pool.join()
    edit_distances = []
    for result in tqdm(results):
        if result['status'] != 0:
            skip_dict[result['status']] += 1
            continue
        if character:
            for i in range(len(result['src_data'])):
                result['src_data'][i] = " ".join([char for char in "".join(result['src_data'][i].split())])
            for i in range(len(result['tgt_data'])):
                result['tgt_data'][i] = " ".join([char for char in "".join(result['tgt_data'][i].split())])
        edit_distances.append(result['edit_distance'])
        src_data.extend(result['src_data'])
        tgt_data.extend(result['tgt_data'])
    print("Avg. edit distance:", np.mean(edit_distances))
    print('size', len(src_data))
    for key, value in skip_dict.items():
        print(f"{key}:{value},{value / len(reactants)}")
    if augmentation != 999:
        with open(
                os.path.join(save_dir, 'src-{}.txt'.format(set_name)), 'w') as f:
            for src in src_data:
                f.write('{}\n'.format(src))

        with open(
                os.path.join(save_dir, 'tgt-{}.txt'.format(set_name)), 'w') as f:
            for tgt in tgt_data:
                f.write('{}\n'.format(tgt))
    return src_data, tgt_data


def multi_process(data):
    pt = re.compile(r':(\d+)]')        # patterns of map numbers
    product = data['product']
    reactant = data['reactant']
    augmentation = data['augmentation']
    pro_mol = Chem.MolFromSmiles(product)
    rea_mol = Chem.MolFromSmiles(reactant)
    """checking data quality"""
    rids = sorted(re.findall(pt, reactant))
    pids = sorted(re.findall(pt, product))
    return_status = {
        "status":0,
        "src_data":[],
        "tgt_data":[],
        "edit_distance":0,
    }
    # if ",".join(rids) != ",".join(pids):  # mapping is not 1:1
    #     return_status["status"] = "error_mapping"
    # if len(set(rids)) != len(rids):  # mapping is not 1:1
    #     return_status["status"] = "error_mapping"
    # if len(set(pids)) != len(pids):  # mapping is not 1:1
    #     return_status["status"] = "error_mapping"
    if "" == product:
        return_status["status"] = "empty_p"
    if "" == reactant:
        return_status["status"] = "empty_r"
    if rea_mol is None:
        return_status["status"] = "invalid_r"
    if len(rea_mol.GetAtoms()) < 5:
        return_status["status"] = "small_r"
    if pro_mol is None:
        return_status["status"] = "invalid_p"
    if len(pro_mol.GetAtoms()) == 1:
        return_status["status"] = "small_p"
    if not all([a.HasProp('molAtomMapNumber') for a in pro_mol.GetAtoms()]):
        return_status["status"] = "error_mapping_p"
    """finishing checking data quality"""

    if return_status['status'] == 0:
        pro_atom_map_numbers = list(map(int, re.findall(r"(?<=:)\d+", product)))
        reactant = reactant.split(".")
        if data['root_aligned']:
            reversable = False  # no shuffle
            # augmentation = 100
            if augmentation == 999:
                product_roots = pro_atom_map_numbers
                times = len(product_roots)
            else:
                product_roots = [-1]
                # reversable = len(reactant) > 1

                # actually aug time = num_prod_atoms
                max_times = len(pro_atom_map_numbers)
                times = min(augmentation, max_times)
                if times < augmentation:  # times = max_times
                    product_roots.extend(pro_atom_map_numbers)
                    product_roots.extend(random.choices(product_roots, k=augmentation - len(product_roots)))
                else:  # times = augmentation
                    while len(product_roots) < times:
                        product_roots.append(random.sample(pro_atom_map_numbers, 1)[0])
                        # pro_atom_map_numbers.remove(product_roots[-1])
                        if product_roots[-1] in product_roots[:-1]:
                            product_roots.pop()
                times = len(product_roots)
                assert times == augmentation
                if reversable:
                    times = int(times / 2)
            # candidates = []
            for k in range(times):
                pro_root_atom_map = product_roots[k]
                pro_root = get_root_id(pro_mol, root_map_number=pro_root_atom_map)
                cano_atom_map = get_cano_map_number(product, root=pro_root)
                if cano_atom_map is None:
                    return_status["status"] = "error_mapping"
                    return return_status
                pro_smi = clear_map_canonical_smiles(product, canonical=True, root=pro_root)
                aligned_reactants = []
                aligned_reactants_order = []
                rea_atom_map_numbers = [list(map(int, re.findall(r"(?<=:)\d+", rea))) for rea in reactant]
                used_indices = []
                for i, rea_map_number in enumerate(rea_atom_map_numbers):
                    for j, map_number in enumerate(cano_atom_map):
                        # select mapping reactans
                        if map_number in rea_map_number:
                            rea_root = get_root_id(Chem.MolFromSmiles(reactant[i]), root_map_number=map_number)
                            rea_smi = clear_map_canonical_smiles(reactant[i], canonical=True, root=rea_root)
                            aligned_reactants.append(rea_smi)
                            aligned_reactants_order.append(j)
                            used_indices.append(i)
                            break
                sorted_reactants = sorted(list(zip(aligned_reactants, aligned_reactants_order)), key=lambda x: x[1])
                aligned_reactants = [item[0] for item in sorted_reactants]
                reactant_smi = ".".join(aligned_reactants)
                product_tokens = smi_tokenizer(pro_smi)
                reactant_tokens = smi_tokenizer(reactant_smi)

                return_status['src_data'].append(product_tokens)
                return_status['tgt_data'].append(reactant_tokens)

                if reversable:
                    aligned_reactants.reverse()
                    reactant_smi = ".".join(aligned_reactants)
                    product_tokens = smi_tokenizer(pro_smi)
                    reactant_tokens = smi_tokenizer(reactant_smi)
                    return_status['src_data'].append(product_tokens)
                    return_status['tgt_data'].append(reactant_tokens)
            assert len(return_status['src_data']) == data['augmentation']
        else:
            cano_product = clear_map_canonical_smiles(product)
            cano_reactanct = ".".join([clear_map_canonical_smiles(rea) for rea in reactant if len(set(map(int, re.findall(r"(?<=:)\d+", rea))) & set(pro_atom_map_numbers)) > 0 ])
            return_status['src_data'].append(smi_tokenizer(cano_product))
            return_status['tgt_data'].append(smi_tokenizer(cano_reactanct))
            pro_mol = Chem.MolFromSmiles(cano_product)
            rea_mols = [Chem.MolFromSmiles(rea) for rea in cano_reactanct.split(".")]
            for i in range(int(augmentation-1)):
                pro_smi = Chem.MolToSmiles(pro_mol,doRandom=True)
                rea_smi = [Chem.MolToSmiles(rea_mol,doRandom=True) for rea_mol in rea_mols]
                rea_smi = ".".join(rea_smi)
                return_status['src_data'].append(smi_tokenizer(pro_smi))
                return_status['tgt_data'].append(smi_tokenizer(rea_smi))
        edit_distances = []
        for src,tgt in zip(return_status['src_data'],return_status['tgt_data']):
            edit_distances.append(textdistance.levenshtein.distance(src.split(),tgt.split()))
        return_status['edit_distance'] = np.mean(edit_distances)
    return return_status


"""
random augmentation, copied from retroformer:
"""
def clear_map_smiles(smi, canonical=False, randomize=False):
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        for atom in mol.GetAtoms():
            if atom.HasProp('molAtomMapNumber'):
                atom.ClearProp('molAtomMapNumber')
        return Chem.MolToSmiles(mol, isomericSmiles=True, doRandom=randomize, canonical=canonical)
    else:
        return smi

def get_cooked_smi(atommap_smi, randomize=False):
    """
    get cooked[canonical/random] smiles (with, without) atom-map
    """
    if '.' in atommap_smi:
        atommap_smi_list = atommap_smi.split('.')
        cooked_smi_am_list = []
        for smi in atommap_smi_list:
            cooked_smi_am = get_cooked_smi(smi)
            cooked_smi_am_list.append(cooked_smi_am)
        # re-permute reacts by specific probability(np.random.rand()) if randomize
        cooked_smi_am_list = sorted(cooked_smi_am_list, key=lambda x: len(x), reverse=(randomize and np.random.rand() > 0.5))
        cooked_smi_am = '.'.join(cooked_smi_am_list)
    else:
        atommap_mol = Chem.MolFromSmiles(atommap_smi)
        cooked_smi = clear_map_smiles(atommap_smi, canonical=True, randomize=randomize)
        cooked_mol = Chem.MolFromSmiles(cooked_smi)
        cooked2atommapIdx = atommap_mol.GetSubstructMatch(cooked_mol)
        id2atommap = [atom.GetAtomMapNum() for atom in atommap_mol.GetAtoms()]
        try:
            cooked_atom_map = [id2atommap[cooked2atommapIdx[i]] for i in range(len(cooked_mol.GetAtoms()))]
            for i, atom_map in enumerate(cooked_atom_map):
                # if atom_map != 0:
                cooked_mol.GetAtomWithIdx(i).SetIntProp('molAtomMapNumber', atom_map)
            cooked_smi_am = Chem.MolToSmiles(cooked_mol, isomericSmiles=True, canonical=False)
        except:
            # logger.info(atommap_smi)
            cooked_smi_am = atommap_smi
    return cooked_smi_am

"""
tokenize smiles
"""
def smi_tokenizer(smi):
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    if smi != ''.join(tokens):
        print('ERROR:', smi, ''.join(tokens))
    assert smi == ''.join(tokens)
    return ' '.join(tokens)

