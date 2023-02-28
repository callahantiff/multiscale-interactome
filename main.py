#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import needed libraries
from msi.msi import *
from diff_prof.diffusion_profiles import *


### STEP 1 -- Build Multiscale Interactome
# set paths to data files
core_dir = 'resources/data/'
drug2protein_file_path = core_dir + '1_drug_to_protein.tsv'
indication2protein_file_path = core_dir + '2_indication_to_protein.tsv'
protein2protein_file_path = core_dir + '3_protein_to_protein.tsv'
protein2biological_function_file_path = core_dir + '4_protein_to_biological_function.tsv'
biological_function2biological_function_file_path = core_dir + '5_biological_function_to_biological_function.tsv'

# construct the multiscale interactome
msi = MSI(drug2protein_directed=False, indication2protein_directed=False, protein2protein_directed=False,
          protein2biological_function_directed=False, biological_function2biological_function_directed=True,
          drug2protein_file=drug2protein_file_path, indication2protein_file=indication2protein_file_path, protein2protein_file=protein2protein_file_path,
          protein2biological_function_file=protein2biological_function_file_path,
          biological_function2biological_function_file=biological_function2biological_function_file_path)
msi.load()

# save network and node information dict (also gets saved when building diffusion profiles)
msi.save_graph('resources/data/mis_graph_v1.pkl')
msi.save_node2idx('resources/data/mis_graph_v1_node2idx.pkl')


### STEP 2 -- Build Diffusion Profiles
# set inputs with default values according to README instructions
alpha = 0.8595436247434408
weight_dict = {
    'down_biological_function':  4.4863053901688685,
    'indication': 3.541889556309463,
    'biological_function': 6.583155399238509,
    'up_biological_function': 2.09685000906964,
    'protein': 4.396695660380823,
    'drug': 3.2071696595616364
}

# derive diffusion profiles
dp = DiffusionProfiles(alpha=alpha, max_iter=1000, tol=1e-06, weights=weight_dict,
                       num_cores=int(multiprocessing.cpu_count()/2) - 4,
                       save_load_file_path="results/")
dp.calculate_diffusion_profiles(msi)

# load a saved diffusion profile
dp_saved = DiffusionProfiles(alpha=None, max_iter=None, tol=None, weights=None, num_cores=None,
                             save_load_file_path="results/")
msi.load_saved_node_idx_mapping_and_nodelist(dp_saved.save_load_file_path)
dp_saved.load_diffusion_profiles(msi.drugs_in_graph + msi.indications_in_graph)

# view diffusion profile for a specific drug
# examples --> diffusion profile for Rosuvastatin (DB01098)
drug_example = dp_saved.drug_or_indication2diffusion_profile["DB01098"]
# drug_example
