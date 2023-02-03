#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import needed libraries
from msi.msi import *
from diff_prof.diffusion_profiles import *


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
          drug2protein_file_path=drug2protein_file_path, indication2protein_file_path=indication2protein_file_path,
          protein2protein_file_path=protein2protein_file_path,
          protein2biological_function_file_path=protein2biological_function_file_path,
          biological_function2biological_function_file_path=biological_function2biological_function_file_path)
msi.load()
