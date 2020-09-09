import numpy as np

val_names = np.load('../val_names.npy')
dimer_ind = np.where(val_names == 'SSI-051GLN-089GLU-2-dimer')
dimer_pred = np.load('../test_pred_charges.npy')[dimer_ind]
dimer_pred = dimer_pred[0][:16]

dimer_lab = np.load('../test_lab_charges.npy')[dimer_ind]
dimer_lab = dimer_lab[0][:16] 
#print(dimer_lab)

monomer_lab = np.load('GLN_GLU_monomers_label.npy')
#print(monomer_lab)

raw_monomer_pred = np.squeeze(np.load('monomer_preds.npy'))
monomer_pred = np.concatenate([raw_monomer_pred[1], raw_monomer_pred[0]])[:16]
#print(monomer_preds)

pol_lab = dimer_lab - monomer_lab
pol_pred = dimer_pred - monomer_pred
pol_err = pol_lab - pol_pred

print(pol_lab)
print(pol_pred)
print(pol_err)
