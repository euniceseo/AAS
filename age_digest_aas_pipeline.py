#!/usr/bin/env python
# coding: utf-8

# In[1]:


# imports
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pickle
from scipy.stats import pearsonr
plt.rcParams['figure.dpi'] = 500


# In[2]:


qmtp_file_path = '/Users/eunicekoo/Downloads/fastas/combined_MTP_dict.p'

with open(qmtp_file_path, 'rb') as f:
    qMTP_dict = pickle.load(f)

mtp_dict = {
    key: {i: item for i, item in enumerate(value)} if isinstance(value, list) else {0: value}
    for key, value in qMTP_dict.items()
}


# In[3]:


val_evidence_dict = {}

diann_dir = '/Users/eunicekoo/Downloads/MTP FASTA DIANN 021025/'
aa_subs_dir = '/Users/eunicekoo/Downloads/fastas/'

# split up the evidence based on raw file
file_paths = [
    "report_tp_1.tsv",
    "report_tp_2.tsv",
    "report_tp_3_5.tsv",
    "report_tp_4.tsv",
    "report_tp_6_2.tsv"
]

report_file_1_9_lysc_trp = pd.concat([pd.read_csv(f, sep='\t') for f in file_paths], ignore_index=True)
report_file_1_9_lysc = pd.read_csv('report.tsv', sep='\t', engine='python')

report_file_1_9_lysc_trp = report_file_1_9_lysc_trp.loc[(report_file_1_9_lysc_trp['PEP'] <= 0.01), :]
report_file_1_9_lysc = report_file_1_9_lysc.loc[(report_file_1_9_lysc['PEP'] <= 0.01), :]

report_dfs_1_9_lysc_trp = {}
report_dfs_1_9_lysc = {}

raw_files_list_1_9_lysc_trp = list(set(report_file_1_9_lysc_trp['File.Name']))
raw_files_list_1_9_lysc = list(set(report_file_1_9_lysc['File.Name']))

for raw_file in raw_files_list_1_9_lysc_trp:
    sub_df = report_file_1_9_lysc_trp[report_file_1_9_lysc_trp['File.Name'] == raw_file].reset_index(drop=True)
    report_dfs_1_9_lysc_trp[raw_file] = sub_df

for raw_file in raw_files_list_1_9_lysc:
    sub_df = report_file_1_9_lysc[report_file_1_9_lysc['File.Name'] == raw_file].reset_index(drop=True)
    report_dfs_1_9_lysc[raw_file] = sub_df
    
val_evidence_dict = {
    'lysc+trp': report_dfs_1_9_lysc_trp,
    'lysc': report_dfs_1_9_lysc
    }

pickle.dump(val_evidence_dict, open(aa_subs_dir+'validation_search_evidence_dict.p', 'wb'))


# In[4]:


# validaition 2 script

# function for determining if sequence is identified in validation search
def seq_in_val_search(idx, digest, raw_file):
    """
    Input: index of mtp_dict[tmt_set], tmt_set (sample)
    Output: if peptide is found, output = [index in mtp_dict, index in evidence file, index in mtp list at idx]
    """
    val_ev_df = val_evidence_dict[digest][raw_file]
    ev_seqs = list(val_ev_df['Stripped.Sequence'].values)
    
    mtp_list = mtp_dict['mistranslated sequence'][idx]
    # some mtp entries have >1 putative AAS. If > identified in validation search, return as separate results
    
    for i, mtp in enumerate(mtp_list):
        if mtp in ev_seqs:
            # print(mtp_list)
            # print([idx, [i for i, x in enumerate(ev_seqs) if x==mtp], i])
            return([idx, [i for i, x in enumerate(ev_seqs) if x==mtp], i])
        else:
            return None
    
# apply validation search function to each mtp entry.
print('Identifying mtps that are found in regular search')

samples = [key for subdict in val_evidence_dict.values() if isinstance(subdict, dict) for key in subdict.keys()]
val_hit_lists = {s:[] for s in samples}

for digest_name, digest in val_evidence_dict.items():
    for raw_file in digest:
        for idx, value in tqdm(mtp_dict['Raw file'].items(), desc=f"{digest_name} + {raw_file}", leave=False):
            result = seq_in_val_search(idx, digest_name, raw_file)
            if result:
                val_hit_lists[raw_file].append(result)


# loop through lists of results, create new dict of validated MTPs with link to evidence file index
val_mtp_dict = {}

for s, val_list in val_hit_lists.items():
    val_mtp_dict[s] = {k:{} for k in mtp_dict.keys()}
    val_mtp_dict[s]['idx_val_evidence'] = {}

    for i, val in enumerate(val_list):
        mtp_idx = val[0]
        seq_idx = val[2]
        ev_idx = val[1]

        for k in mtp_dict.keys():
            if (isinstance(mtp_dict[k][mtp_idx], list)) and len(mtp_dict[k][mtp_idx])>0: # this is to make sure that we are extracting the correct AAS data and q-values for the mtp found out of list of mtps
                val_mtp_dict[s][k][i] = mtp_dict[k][mtp_idx][seq_idx]
            else:
                val_mtp_dict[s][k][i] = mtp_dict[k][mtp_idx]
        val_mtp_dict[s]['idx_val_evidence'][i] = ev_idx
    print(len(val_mtp_dict[s]['idx_val_evidence']))

pickle.dump(val_mtp_dict, open(aa_subs_dir+'Validated_MTP_dict.p', 'wb'))
val_mtp_dict = pickle.load(open(aa_subs_dir+'Validated_MTP_dict.p', 'rb'))

# function to determine the number of fragment ions supporting site of AAS
def n_frags_over_MTP(frag_match, mtp, sub_idx, tmt_set):
    """
    Input: fragment matches for peptide from msms.txt (MQ output file), peptide sequence, index of AAS on sequence, tmt_set
    Output: number of fragment ions covering site of AAS
    """
    count = 0
    for f, frag in enumerate(frag_match):
        mtp_frag=0
        if ('NH3' not in frag) and ('H2O' not in frag) and ('(' not in frag) and ('a' not in frag):
            if 'b' in frag:
                frag_start = 0
                frag_end = int(frag[1:])
                frag_seq = mtp[frag_start:frag_end]
                if frag_end>sub_idx:
                    mtp_frag = 1
            elif 'y' in frag:
                frag_start = -int(frag[1:])
                frag_seq = mtp[frag_start:]
                if len(mtp)+frag_start <= sub_idx:
                    mtp_frag=1
            count+=mtp_frag
        
    return(count)

# apply function and annotate val_mtp_dict
for digest_name, digest in val_evidence_dict.items():
    for s in digest:
        ev = val_evidence_dict[digest_name][s]
        
        val_mtp_dict[s]['fragment_evidence'] = {}
        for k,v in val_mtp_dict[s]['aa subs'].items():
            seq = val_mtp_dict[s]['mistranslated sequence'][k]
            bp = val_mtp_dict[s]['DP Base Sequence'][k]
            sub_idx = [i for i,x in enumerate(bp) if seq[i]!=x][0]
            
            ev_idx = val_mtp_dict[s]['idx_val_evidence'][k]
            val_mtp_dict[s]['fragment_evidence'][k] = 0
            for idx in ev_idx:
                row = ev.iloc[idx,:]
                fragment_row = row['Fragment.Info']
    
                if len(fragment_row)>0:
                    frag_match = fragment_row.split(';')
                    ion_types = [re.match(r'([yb]\d+)', frag).group(1) for frag in frag_match if re.match(r'([yb]\d+)', frag)]
                    count_frags = n_frags_over_MTP(ion_types, seq, sub_idx, 'sample')
                    
                    if count_frags>val_mtp_dict[s]['fragment_evidence'][k]:
                        val_mtp_dict[s]['fragment_evidence'][k] = count_frags


# filter val_mtp_dict for those with b/y ion evidence
val_ion_mtp_dict = {outer_key: {inner_key: {} for inner_key in inner_dict.keys()} for outer_key, inner_dict in val_evidence_dict.items() if isinstance(inner_dict, dict)}

for digest_name, digest in val_evidence_dict.items():
    for s in digest:
        ion_idx = [i for i,x in val_mtp_dict[s]['fragment_evidence'].items() if x>1]
        for k,v in val_mtp_dict[s].items():
            val_ion_mtp_dict[digest_name][s][k] = {i:x for i,x in v.items() if i in ion_idx}


# In[5]:


# digest overlap calculation

matched_report_lysc_trp = {}
matched_report_lysc = {}

global_seqs_lysc_trp = []
for df in val_ion_mtp_dict['lysc+trp'].values():
    global_seqs_lysc_trp.extend(list(df['mistranslated sequence'].values()))

global_seqs_lysc = []
for df in val_ion_mtp_dict['lysc'].values():
    global_seqs_lysc.extend(list(df['mistranslated sequence'].values()))


for digest_name, digest in val_ion_mtp_dict.items():
    for raw_file, data_dict in digest.items():
        
        if digest_name == 'lysc+trp':
            mask = {
                key: any(str(s) == str(x) or str(s) in str(x) for x in global_seqs_lysc_trp)
                for key, s in data_dict['mistranslated sequence'].items()
            }

        elif digest_name == 'lysc':
            mask = {
                key: any(str(s) == str(x)for x in global_seqs_lysc)
                for key, s in data_dict['mistranslated sequence'].items()
            }

        for col in data_dict:
            data_dict[col] = {key: value for key, value in data_dict[col].items() if mask.get(key, False)}

pickle.dump(val_ion_mtp_dict, open(aa_subs_dir+'Ion_validated_MTP_dict.p', 'wb'))


# In[6]:


# quant script

mtp_dict = pickle.load(open(aa_subs_dir+'Ion_validated_MTP_dict.p', 'rb'))
samples = list(mtp_dict.keys())
val_evidence_dict = pickle.load(open(aa_subs_dir+'Validation_search_evidence_dict.p', 'rb'))

unq_pairs = []
unq_pair_dicts = []

for digest_name, digest in mtp_dict.items():
    for s, s_dict in digest.items():
        for i, v in s_dict['aa subs'].items():
            sub    = v
            mtp    = s_dict['mistranslated sequence'][i]
            bp     = s_dict['DP Base Sequence'][i]
            ev_idx = s_dict['idx_val_evidence'][i]
            pair = [mtp, bp]
            if pair not in unq_pairs:
                unq_pairs.append(pair)
                unq_pair_dicts.append({
                    'MTP': mtp, 
                    'BP': bp, 
                    'AAS': sub, 
                    'raw_files': {digest_name: [s]},
                    'ev_idx_list': {digest_name: ev_idx},
                    'raw_file_evidence': {digest_name: {s: [ev_idx]}},
                    'digest_types': set([digest_name])
                })
            else:
                curr_dict_idx = [j for j, x in enumerate(unq_pair_dicts) if (x['MTP'] == mtp and x['BP'] == bp)][0]
                if digest_name in unq_pair_dicts[curr_dict_idx]['raw_files']:
                    if s not in unq_pair_dicts[curr_dict_idx]['raw_files'][digest_name]:
                        unq_pair_dicts[curr_dict_idx]['raw_files'][digest_name].append(s)
                else:
                    unq_pair_dicts[curr_dict_idx]['raw_files'][digest_name] = [s]
                
                if digest_name in unq_pair_dicts[curr_dict_idx]['ev_idx_list']:
                    unq_pair_dicts[curr_dict_idx]['ev_idx_list'][digest_name].extend(ev_idx)
                else:
                    unq_pair_dicts[curr_dict_idx]['ev_idx_list'][digest_name] = ev_idx
                
                if digest_name in unq_pair_dicts[curr_dict_idx]['raw_file_evidence']:
                    if s in unq_pair_dicts[curr_dict_idx]['raw_file_evidence'][digest_name]:
                        unq_pair_dicts[curr_dict_idx]['raw_file_evidence'][digest_name][s].extend(ev_idx)
                    else:
                        unq_pair_dicts[curr_dict_idx]['raw_file_evidence'][digest_name][s] = ev_idx
                else:
                    unq_pair_dicts[curr_dict_idx]['raw_file_evidence'][digest_name] = {s: ev_idx}
                
                unq_pair_dicts[curr_dict_idx]['digest_types'].add(digest_name)

    
MTP_quant_dict = {}

for i, pair in enumerate(unq_pair_dicts):
    curr_dict = {
        'MTP_seq': pair['MTP'], 
        'BP_seq': pair['BP'], 
        'sub_index': [idx for idx, x in enumerate(pair['MTP']) if pair['MTP'][idx] != pair['BP'][idx]], 
        'aa_sub': pair['AAS'],
        'raw_files': pair['raw_files'], 
        'raw_file_evidence': pair['raw_file_evidence'],
        'MTP_PrecInt': {},
        'BP_PrecInt': {},
        'Norm_MTP_PrecInt': {},
        'Norm_BP_PrecInt': {},
        'Prec_ratio': {},
        'digest': list(pair['digest_types'])
    }
    MTP_quant_dict[i] = curr_dict
    

def median_normalize(sample_raw):
    """
    Input: list of raw precursor intensities for tissue
    Output: median-normalized list of precursor intensities for tissue
    """
    sample_median = np.median(sample_raw)
    sample_norm = [x / sample_median for x in sample_raw]
    return sample_norm
    

def precursor_intensity_quant(k, tissue, digest_name):
    """
    Input: k = index of MTP in MTP_quant_dict, tissue (raw file), digest_name.
    Output: Returns precursor intensities for MTP and BP along with their normalized intensities and log2 ratio.
            Precursor intensities represent the sum of all precursor quantities mapped to the peptide.
    """
    bp = MTP_quant_dict[k]['BP_seq']
    mtp = MTP_quant_dict[k]['MTP_seq']
    ev_df = val_evidence_dict[digest_name][tissue]
    bp_ev_df = ev_df.loc[ev_df['Stripped.Sequence'] == bp, :]
    bp_prec_int = np.sum([x for x in bp_ev_df['Precursor.Quantity'].values if ~np.isnan(x)])
    norm_bp_prec_int = np.sum([x for x in bp_ev_df['Precursor.Quantity'].values if ~np.isnan(x)])
    
    mtp_ev_df = ev_df.loc[ev_df['Stripped.Sequence'] == mtp, :]
    mtp_prec_int = np.sum([x for x in mtp_ev_df['Precursor.Quantity'].values if ~np.isnan(x)])
    norm_mtp_prec_int = np.sum([x for x in mtp_ev_df['Precursor.Quantity'].values if ~np.isnan(x)])
    
    prec_ratio = np.log2(mtp_prec_int / bp_prec_int)
    return [mtp_prec_int, bp_prec_int, norm_mtp_prec_int, norm_bp_prec_int, prec_ratio]


"""Quantification of precursor and reporter ions"""

for digest_name, digest in val_evidence_dict.items():
    for s in digest:
        ev_df = val_evidence_dict[digest_name][s]
        tissue = s
        
        for k, v in MTP_quant_dict.items():

            if any(s in file_list for file_list in v['raw_files'].values()):
                mtp = v['MTP_seq']
                bp = v['BP_seq']
                bp_ev = ev_df.loc[ev_df['Stripped.Sequence'] == bp, :]
                mtp_ev = ev_df.loc[ev_df['Stripped.Sequence'] == mtp, :]
        
                mtp_prec_int, bp_prec_int, norm_mtp_prec_int, norm_bp_prec_int, prec_ratio = precursor_intensity_quant(k, tissue, digest_name)
                v.setdefault('BP_PrecInt', {}).setdefault(digest_name, {})[tissue] = bp_prec_int
                v.setdefault('MTP_PrecInt', {}).setdefault(digest_name, {})[tissue] = mtp_prec_int
                v.setdefault('Norm_BP_PrecInt', {}).setdefault(digest_name, {})[tissue] = norm_bp_prec_int
                v.setdefault('Norm_MTP_PrecInt', {}).setdefault(digest_name, {})[tissue] = norm_mtp_prec_int
                v.setdefault('Prec_ratio', {}).setdefault(digest_name, {})[tissue] = prec_ratio

pickle.dump(MTP_quant_dict, open(aa_subs_dir+'MTP_quant_dict.p', 'wb')) 


# In[7]:


# run to age. also gets exported into spreadsheet
run_to_age = {
    1: 24, 2: 15, 3: 15, 4: 2, 5: 9, 6: 2, 7: 2, 8: 2, 9: 2, 10: 2, 11: 9, 12: 24, 13: 15, 14: 9, 15: 24, 16: 9, 17: 9, 18: 15, 19: 2, 20: 9, 21: 2, 22: 24, 23: 15, 24: 15, 25: 2
}

# get run number from the raw file
def get_run_number(raw_file):
    tp_match = re.search(r'TP_(\d+)_', raw_file)
    if tp_match:
        return int(tp_match.group(1))
    
    run_match = re.search(r'BRAIN_RUN_(\d+)_', raw_file)
    if run_match:
        return int(run_match.group(1))
    return None


# In[9]:


# making matrix with incomplete data

all_raw_files = raw_files_list_1_9_lysc_trp + raw_files_list_1_9_lysc

data = []

for idx, entry in MTP_quant_dict.items():
    row = {
        'MTP_seq': entry['MTP_seq'],
        'BP_seq': entry['BP_seq']
    }
    
    for raw_file in all_raw_files:
        prec_ratio = None
        for digest in entry['raw_file_evidence'].keys():
            if raw_file in entry['raw_file_evidence'][digest]:
                if digest in entry['Prec_ratio'] and raw_file in entry['Prec_ratio'][digest]:
                    prec_ratio = entry['Prec_ratio'][digest][raw_file]
                    break
        
        row[raw_file] = prec_ratio
    
    data.append(row)

df = pd.DataFrame(data)
df.set_index(['MTP_seq', 'BP_seq'], inplace=True)
df = df[all_raw_files]

age_data = []
for raw_file in all_raw_files:
    filename = raw_file.split('/')[-1]
    run_number = get_run_number(raw_file)
    if run_number in run_to_age:
        age_data.append({
            'Raw File': filename,
            'Full Path': raw_file,
            'Age Group': run_to_age[run_number]
        })

age_df = pd.DataFrame(age_data)


# In[10]:


# computing pearson correlation

df_long = df.reset_index().melt(
    id_vars=['MTP_seq', 'BP_seq'],
    var_name='RawFile',
    value_name='RAAS'
)

rawfile_to_age = dict(zip(age_df['Raw File'], age_df['Age Group']))
df_long['Age'] = df_long['RawFile'].map(rawfile_to_age)

df_long = df_long[
    df_long['RAAS'].notna() &
    df_long['Age'].notna() &
    (df_long['RAAS'] != 0) &
    (df_long['RAAS'] != np.inf) &
    (df_long['RAAS'] != -np.inf)
]

results = []
for (mtp, bp), group in df_long.groupby(['MTP_seq', 'BP_seq']):
    if len(group) >= 2:
        raas = pd.to_numeric(group['RAAS'], errors='coerce')
        age = pd.to_numeric(group['Age'], errors='coerce')
        mask = (~raas.isna()) & (~age.isna())
        if mask.sum() >= 3:
            unique_ages = age[mask].nunique()
            r, p = pearsonr(raas[mask], age[mask])
            results.append({
                'MTP_seq': mtp,
                'BP_seq': bp,
                'R Value': r,
                'P Value': p,
                '# Raw Files': mask.sum(),
                '# Age Groups': unique_ages 
            })

cor_df = pd.DataFrame(results)

# merging correlation df into spreadsheet

df_reset = df.reset_index()
df_merged = df_reset.merge(
    cor_df[['MTP_seq', 'BP_seq', 'R Value', 'P Value', '# Raw Files', '# Age Groups']],
    on=['MTP_seq', 'BP_seq'],
    how='left'
)
df_merged.set_index(['MTP_seq', 'BP_seq'], inplace=True)

with pd.ExcelWriter('raas_complete.xlsx') as writer:
    df_merged.to_excel(writer, sheet_name='RAAS')
    age_df.to_excel(writer, sheet_name='Age Groups', index=False)


# In[11]:


# calculation code to get peptides present in all four age groups

def get_run_number(raw_file):
    tp_match = re.search(r'TP_(\d+)_', raw_file)
    if tp_match:
        return int(tp_match.group(1))
    
    run_match = re.search(r'BRAIN_RUN_(\d+)_', raw_file)
    if run_match:
        return int(run_match.group(1))
    
    return None

peptide_pairs_by_age = {2: set(), 9: set(), 15: set(), 24: set()}

raas_by_peptide = {}

for peptide in MTP_quant_dict.values():
    mtp_seq = peptide['MTP_seq']
    bp_seq = peptide['BP_seq']
    pair = (bp_seq, mtp_seq)
    
    if pair not in raas_by_peptide:
        raas_by_peptide[pair] = {2: [], 9: [], 15: [], 24: []}
    
    age_groups_with_both = set()
    
    for digest_name, digest_vals in peptide['Prec_ratio'].items():
        for raw_file, value in digest_vals.items():
            try:
                val = float(value)
                if not np.isfinite(val):
                    continue
            except:
                continue
            
            run = get_run_number(raw_file)
            if run is not None:
                age = run_to_age.get(run)
                if age in peptide_pairs_by_age:
                    age_groups_with_both.add(age)
                    raas_by_peptide[pair][age].append(val)

    for age in age_groups_with_both:
        peptide_pairs_by_age[age].add(pair)

common_pairs = set.intersection(*peptide_pairs_by_age.values())
print(f"Number of BP-MTP pairs present in all age groups: {len(common_pairs)}")

valid_peptides = {}
for pair in common_pairs:
    has_valid_values = True
    median_raas_by_age = {}
    n_points_by_age = {}
    
    for age in [2, 9, 15, 24]:
        values = raas_by_peptide[pair][age]
        if not values:
            has_valid_values = False
            break
        median_raas_by_age[age] = np.median(values) 
        n_points_by_age[age] = len(values) 
    
    if has_valid_values:
        valid_peptides[pair] = {
            'median_raas': median_raas_by_age,
            'n_points': n_points_by_age
        }

