#!/usr/bin/env python3
"""Verify completed full-data GRU, preprocessing, coverage and saved predictions."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from gru_common import (IDENTITY, SiteGRU, fit_preprocessing, load_data, load_spec,
                        predict, setup_runtime, sha256, transform, write_json)

HERE = Path(__file__).resolve().parent
out = HERE / 'model'
manifest = json.loads((out/'run_manifest.json').read_text())
assert manifest['status']=='COMPLETE'
assert manifest['refit_epochs']==32 and manifest['final_fit']['epochs_run']==32
assert manifest['arguments']['mode']=='fixed' and not manifest['test_evaluated']
selection=json.loads((HERE/'configuration_selection.json').read_text())
for k,v in selection['selected_config'].items():
    if k!='epochs': assert manifest['config'][k]==v
train, _, _ = load_data(load_spec(HERE/'datasets.json'),test_metadata_only=True)
assert len(train)==172274
assert train.groupby('dataset_id').size().to_dict()=={'dataset0':97879,'dataset1':73380,'dataset2':1015}
rows=pd.read_csv(out/'training_rows.csv.gz')
pd.testing.assert_frame_equal(rows, train[IDENTITY+['original_split','label','fold']].astype({'dataset_id':str,'transcript_id':str}).reset_index(drop=True))
prep=json.loads((out/'preprocessing.json').read_text())
expected=fit_preprocessing(train)
for field in ['sequence','coverage']:
    for stat in ['mean','scale']:
        np.testing.assert_allclose(prep[field][stat],expected[field][stat],rtol=1e-12,atol=1e-12)
for name,digest in manifest['artifacts'].items(): assert sha256(out/name)==digest
for entry in manifest['inputs'].values():
    for split in ['train','test']: assert sha256(entry[f'{split}_file'])==entry[f'{split}_sha256']
device=setup_runtime(42,'cpu',4)
state=torch.load(out/'model.pt',map_location=device,weights_only=True)
assert state['epoch']==32 and state['config']==manifest['config']
model=SiteGRU(state['config']['hidden_size'],state['config']['dropout']).to(device)
model.load_state_dict(state['state_dict'])
scores=predict(model,transform(train,prep),device)
saved=pd.read_csv(out/'train_outputs/detailed_predictions_train_apparent.csv.gz')
pd.testing.assert_frame_equal(saved[IDENTITY],train[IDENTITY].astype({'dataset_id':str,'transcript_id':str}).reset_index(drop=True))
np.testing.assert_allclose(scores,saved.score,rtol=1e-6,atol=1e-7)
h=pd.read_csv(out/'training_history.csv')
assert h.epoch.tolist()==list(range(1,33)) and 'val_PR_AUC' not in h
assert np.isfinite(h.train_BCE).all()
audit=json.loads((out/'split_audit.json').read_text())
assert audit['shared_sites']==audit['shared_biological_groups']==0
result={'status':'PASS','n_training_rows':len(train),'original_split_counts':train.original_split.value_counts().to_dict(),'round6_config_matches':True,'all_32_epochs_completed':True,'preprocessing_matches_merged_training_only':True,'saved_model_predictions_match':True,'artifact_and_input_hashes_match':True,'test_site_and_group_overlap':0,'test_evaluated':False}
write_json(out/'verification.json',result)
print(json.dumps(result,indent=2))
