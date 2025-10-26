dict_a = {1: [2, 3]}
dict_b = {1: [4]}

dict_a.update(dict_b)

print(f"dict_a = {dict_a}")


from rdkit import Chem

smi = "COc1ccc(Cl)cc1CO.O=S(Cl)Cl"
mol = Chem.MolFromSmiles(smi)
check_smi = Chem.MolToSmiles(mol, isomericSmiles=True)
