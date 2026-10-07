"""Fit ONE shared GB, MLP, or MIL on training portions of dataset0/1/2; keep test sites unseen."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from clean_data import model_inputs, cleaned_sites, sha256
from features import KEY, SCHEMA, MIL_SCHEMA, pooled_features, bag_features
from metrics import pr_auc_score
def prepare_groups(labels, mode='gene', gene_map=None):
    labels = labels.copy()
    required = KEY + ['label']
    if not set(required).issubset(labels) or labels[required].isna().any().any():
        raise ValueError('Missing training coordinates or labels')
    if labels.duplicated(KEY).any() or not labels.label.isin([0, 1]).all():
        raise ValueError('Training requires unique coordinates and binary labels')
    if 'gene_id' not in labels:
        labels['gene_id'] = pd.NA
    known = labels.dropna(subset=['gene_id'])
    if (known.groupby('transcript_id').gene_id.nunique() > 1).any():
        raise ValueError('Conflicting transcript-to-gene mapping')
    info = dict(mode=mode, gene_map=None, limitations=[])
    if gene_map:
        if mode != 'gene-or-transcript':
            raise ValueError('--gene-map is only used with gene-or-transcript grouping')
        # Explicitly read annotation columns only, never reference sample labels.
        mapping = pd.read_csv(gene_map, usecols=['transcript_id','gene_id'], dtype=str)
        if mapping.isna().any().any() or (mapping.groupby('transcript_id').gene_id.nunique() > 1).any():
            raise ValueError('Invalid annotation mapping')
        mapping = mapping.drop_duplicates('transcript_id').set_index('transcript_id').gene_id
        mapped = labels.transcript_id.map(mapping)
        conflict = labels.gene_id.notna() & mapped.notna() & (labels.gene_id != mapped)
        if conflict.any(): raise ValueError('Input and reference gene annotations disagree')
        labels['gene_id'] = labels.gene_id.fillna(mapped)
        info['gene_map'] = dict(path=str(Path(gene_map).resolve()), sha256=sha256(gene_map),
                                columns_used=['transcript_id','gene_id'])
    if mode == 'gene':
        if labels.gene_id.isna().any(): raise ValueError('Gene grouping requires gene_id for every site')
        labels['group_id'] = 'gene:' + labels.gene_id
        labels['group_source'] = 'gene'
    elif mode == 'gene-or-transcript':
        labels['group_id'] = np.where(labels.gene_id.notna(), 'gene:' + labels.gene_id.fillna(''),
                                     'transcript:' + labels.transcript_id)
        labels['group_source'] = np.where(labels.gene_id.notna(), 'gene', 'transcript_fallback')
        info['limitations'].append('Unmapped transcripts are grouped separately; unknown isoform relationships may cross partitions.')
    elif mode == 'mixture-position':
        if 'modification_fraction' not in labels:
            raise ValueError('Mixture grouping requires preserved modification_fraction')
        fraction = labels.modification_fraction
        if not fraction.between(0, 1).all() or not (labels.label == (fraction > 0).astype(int)).all():
            raise ValueError('Mixture targets must equal int(modification_fraction > 0)')
        if labels.groupby('transcript_id').modification_fraction.nunique().max() != 1:
            raise ValueError('Expected each transcript ID to represent one mixture condition')
        labels['group_id'] = 'mixture_position:' + labels.transcript_position
        labels['group_source'] = 'assumed_shared_position'
        info['limitations'].append('Same numeric positions across condition IDs are assumed to identify shared sites; identical 7-mers are checked, but gene identities and read provenance are unavailable.')
    else:
        raise ValueError(f'Unknown grouping mode: {mode}')
    info['sites_by_group_source'] = labels.group_source.value_counts().to_dict()
    info['n_groups'] = int(labels.group_id.nunique())
    return labels, info


def unified_groups(label_tables):
    """Use global genes/transcripts across samples; keep dataset2 mixtures together."""
    annotation_rows=[frame[['transcript_id','gene_id']].dropna()
                     for frame in label_tables.values() if 'gene_id' in frame]
    known=pd.concat(annotation_rows,ignore_index=True) if annotation_rows else pd.DataFrame(columns=['transcript_id','gene_id'])
    if (known.groupby('transcript_id').gene_id.nunique()>1).any():
        raise ValueError('Conflicting cross-dataset transcript-to-gene annotation')
    mapping = known.drop_duplicates('transcript_id').set_index('transcript_id').gene_id
    grouped=[]
    for dataset,original in label_tables.items():
        frame=original.copy()
        if dataset=='dataset2':
            frame,_=prepare_groups(frame,'mixture-position')
            frame['group_id']='dataset2:'+frame.group_id
        else:
            if 'gene_id' not in frame: frame['gene_id']=pd.NA
            frame['gene_id']=frame.gene_id.fillna(frame.transcript_id.map(mapping))
            frame,_=prepare_groups(frame,'gene-or-transcript')
        frame.insert(0,'dataset',dataset)
        grouped.append(frame)
    table=pd.concat(grouped,ignore_index=True)
    if table.duplicated(['dataset']+KEY).any(): raise ValueError('Duplicate sample coordinates')
    return table


def unified_split(table,folds,seed):
    # Joint strata help preserve each sample's class representation.
    strata=table.dataset+':'+table.label.astype(int).astype(str)
    outer=StratifiedGroupKFold(n_splits=folds,shuffle=True,random_state=seed)
    development,test=next(outer.split(np.zeros(len(table)),strata,table.group_id))
    inner=StratifiedGroupKFold(n_splits=folds,shuffle=True,random_state=seed+1)
    fit_rel,val_rel=next(inner.split(np.zeros(len(development)),strata.iloc[development],table.iloc[development].group_id))
    fit,validation=development[fit_rel],development[val_rel]
    groups=[set(table.iloc[indices].group_id) for indices in (fit,validation,test)]
    if not (groups[0].isdisjoint(groups[1]) and groups[0].isdisjoint(groups[2]) and groups[1].isdisjoint(groups[2])):
        raise ValueError('Cross-partition group overlap')
    for indices in (fit,validation,test):
        for dataset in ('dataset0','dataset1','dataset2'):
            frame=table.iloc[indices]; frame=frame.loc[frame.dataset==dataset]
            if frame.label.nunique()!=2: raise ValueError(f'Both classes required for {dataset} in every partition')
    return fit,validation,test


def train_unified(args):
    """Select one shared model on validation and evaluate genuinely held-out sites."""
    if min(args.epochs,args.batch_size,args.threads,args.max_reads)<1 or args.folds<2 or args.patience<0 or args.learning_rate<=0:
        raise ValueError('Invalid training settings')
    if args.selection_metric!='pr_auc': raise ValueError('Shared-model selection uses validation PR-AUC')
    out=Path(args.out_dir)
    if out.exists(): raise FileExistsError(f'Choose a new output directory: {out}')
    with ExitStack() as stack:
        paths={};audits={};labels={}
        if args.clean_dirs:
            entries=[(Path(path).name,dict(clean_dir=path)) for path in args.clean_dirs]
        else:
            entries=[(Path(path).name.split('.')[0],dict(data=path,labels=label))
                     for path,label in zip(args.data_files,args.label_files)]
        if len(entries)!=3 or {name for name,_ in entries}!={'dataset0','dataset1','dataset2'}:
            raise ValueError('Combined mode expects dataset0, dataset1, dataset2 (directory names or raw filename stems)')
        for dataset,kwargs in sorted(entries):
            data_path,label_path,audit=stack.enter_context(model_inputs(**kwargs,require_labels=True))
            paths[dataset]=data_path;audits[dataset]=audit
            labels[dataset]=pd.read_csv(label_path,dtype={k:str for k in KEY+['gene_id']})
        table=unified_groups(labels)
        lookup={key:i for i,key in enumerate(table[['dataset']+KEY].itertuples(index=False,name=None))}
        rows=[];vectors=[];masks=[];sequences=[];motifs={};counts={}
        for dataset,path in paths.items():
            count=0
            for key,motif,reads in cleaned_sites(path):
                count+=1
                if dataset=='dataset2':
                    if motifs.setdefault(key[1],motif)!=motif: raise ValueError('Inconsistent dataset2 mixture sequences')
                index=lookup.get((dataset,*key))
                if index is not None:
                    rows.append(index)
                    if args.model_type == 'mil':
                        bag,mask,sequence=bag_features(key,motif,reads,args.max_reads,args.seed)
                        vectors.append(bag);masks.append(mask);sequences.append(sequence)
                    else:
                        vectors.append(pooled_features(motif,reads).astype(np.float32))
                if count%25000==0: print(f'{dataset}: processed {count:,} sites',flush=True)
            counts[dataset]=count
        table=table.iloc[rows].reset_index(drop=True)
        X=np.stack(vectors);del vectors
        y=table.label.astype(int).to_numpy()
        arrays=(X,np.stack(masks),np.stack(sequences)) if args.model_type=='mil' else (X,)
        del masks,sequences
        fit,validation,test=unified_split(table,args.folds,args.seed)
        out.mkdir(parents=True)
        manifest=table.copy();manifest['split']='train'
        manifest.loc[validation,'split']='validation';manifest.loc[test,'split']='test'
        manifest.to_csv(out/'split_manifest.csv',index=False)
        summaries=[]
        for (dataset,split),frame in manifest.groupby(['dataset','split']):
            summaries.append(dict(dataset=dataset,split=split,n_sites=len(frame),n_positive=int(frame.label.sum()),n_groups=frame.group_id.nunique()))
        pd.DataFrame(summaries).to_csv(out/'split_summary.csv',index=False)
        weight=float((y[fit]==0).sum()/(y[fit]==1).sum()) if args.class_weight=='balanced' else 1.
        if args.model_type=='gb':
            from sklearn.ensemble import HistGradientBoostingClassifier
            from threadpoolctl import threadpool_limits
            from itertools import product
            history=[];best_score=-np.inf
            with threadpool_limits(limits=args.threads):
                grid=list(product((7,15,31,63),(100,200,400,600),(.01,.03,.1,.2),(0.,1.,10.,30.),(10,20,50,100)))
                rng=np.random.default_rng(args.seed)
                choices=[(31,300,.03,10.,20)]+[grid[i] for i in rng.choice(len(grid),31,replace=False)]
                for leaves,iterations,rate,l2,min_leaf in choices:
                    params=dict(max_leaf_nodes=leaves,max_iter=iterations,learning_rate=rate,l2_regularization=l2,min_samples_leaf=min_leaf)
                    candidate=HistGradientBoostingClassifier(**params,early_stopping=False,random_state=args.seed)
                    candidate.fit(X[fit],y[fit],sample_weight=np.where(y[fit]==1,weight,1.))
                    val_score=candidate.predict_proba(X[validation])[:,1]
                    value=pr_auc_score(y[validation],val_score)
                    record=dict(candidate=len(history)+1,params=params,validation_pr_auc=value,
                                validation_ap=float(average_precision_score(y[validation],val_score)))
                    history.append(record);print(record,flush=True)
                    if value>best_score:
                        model=candidate;best_score=value;best_params=params;best_candidate=record['candidate']
                scores=model.predict_proba(X[test])[:,1]
            selection=dict(metric='pooled_validation_pr_auc',best_score=best_score,best_params=best_params,
                           best_candidate=best_candidate,candidate_count=len(history))
            artifact=dict(kind='gb',schema=SCHEMA,model=model,seed=args.seed,best_candidate=best_candidate)
        else:
            import torch
            from neural import train_network,probabilities
            torch.set_num_threads(args.threads)
            # Screen a reproducible broad sample, then train the best candidates longer.
            configs=[]
            rng=np.random.default_rng(args.seed)
            base=dict(network_config={},learning_rate=args.learning_rate,weight_decay=.0001,batch_size=args.batch_size,class_weight=args.class_weight)
            configs.append(base)
            for i in range(11):
                if args.model_type=='mlp':
                    config=dict(hidden=[[32,16],[64,32],[128,64],[256,128]][i%4],dropout=[0.,.1,.3,.5][i%4])
                else:
                    config=dict(width=[16,32,64][i%3],attention_width=[8,16,32][(i//3)%3],dropout=[0.,.1,.3,.5][i%4])
                configs.append(dict(network_config=config,learning_rate=[.0001,.0003,.001,.003][(i//3)%4],
                    weight_decay=[0.,.0001,.001][i%3],batch_size=[128,256,512][i%3],
                    class_weight='balanced' if i%3==0 else 'none'))
            records=[];screen=[]
            def trial(candidate,epochs,stage):
                config=configs[candidate]
                positive_weight=float((y[fit]==0).sum()/(y[fit]==1).sum()) if config['class_weight']=='balanced' else 1.
                print(f'{args.model_type} {stage} candidate {candidate+1}: {config}',flush=True)
                result=train_network(args.model_type,arrays,y,fit,validation,args.seed,epochs,config['batch_size'],
                    positive_weight,'pr_auc',config['learning_rate'],args.patience,config['network_config'],config['weight_decay'])
                net,mu,sd,trial_history,epoch=result
                for record in trial_history:
                    records.append(dict(record,candidate=candidate+1,stage=stage,params=config))
                (out/'tuning_history.json').write_text(json.dumps(records,indent=2)+'\n')
                return result,max(row['validation_pr_auc'] for row in trial_history)
            for candidate in range(len(configs)):
                result,value=trial(candidate,4,'screen')
                screen.append((value,candidate))
                del result
            finalists=[0]+[candidate for _,candidate in sorted(screen,reverse=True) if candidate!=0][:2]
            best_score=-np.inf
            for candidate in finalists:
                result,value=trial(candidate,args.epochs,'final')
                if value>best_score:
                    model,mean,scale,history,best_epoch=result
                    best_score=value;best_candidate=candidate+1;best_params=configs[candidate]
            weight=float((y[fit]==0).sum()/(y[fit]==1).sum()) if best_params['class_weight']=='balanced' else 1.
            selection=dict(metric='pooled_validation_pr_auc',best_score=best_score,best_epoch=best_epoch,
                epochs_run=len(history),best_candidate=best_candidate,best_params=best_params,
                screened_candidates=len(configs),finalists=[i+1 for i in finalists],screen_epochs=4)
            scores=probabilities(model,args.model_type,arrays,test,mean,scale,args.batch_size)
            artifact=dict(kind=args.model_type,schema=MIL_SCHEMA if args.model_type=='mil' else SCHEMA,
                          seed=args.seed,max_reads=args.max_reads,network_config=best_params['network_config'],
                          state={k:v.detach().cpu().numpy() for k,v in model.state_dict().items()},
                          mean=mean,scale=scale,best_epoch=best_epoch,torch_version=str(torch.__version__))
        (out/'validation_history.json').write_text(json.dumps(history,indent=2)+'\n')
        heldout=table.iloc[test].copy();heldout['score']=scores
        evaluation=out/'evaluation';evaluation.mkdir()
        metric_rows=[]
        for dataset in ('dataset0','dataset1','dataset2','pooled'):
            frame=heldout if dataset=='pooled' else heldout.loc[heldout.dataset==dataset]
            metric_rows.append(dict(dataset=dataset,model='shared_'+args.model_type,evaluation='heldout_test',
                n_sites=len(frame),n_positive=int(frame.label.sum()),prevalence=float(frame.label.mean()),
                average_precision=float(average_precision_score(frame.label,frame.score)),
                pr_auc=pr_auc_score(frame.label,frame.score),roc_auc=float(roc_auc_score(frame.label,frame.score)),
                brier=float(brier_score_loss(frame.label,frame.score))))
            if dataset!='pooled': frame[KEY+['score']].to_csv(evaluation/f'{dataset}_heldout_predictions.csv',index=False)
        pd.DataFrame(metric_rows).to_csv(out/'evaluation_metrics.csv',index=False)
        # Save the selected development checkpoint unchanged: test sites stay unseen.
        report=dict(protocol=f'One shared {args.model_type.upper()} fitted on joint training portions; selected by pooled validation PR-AUC; test portions never fitted.',
            settings=vars(args),model_selection=selection,preprocessing=audits,
            json_sites=counts,supplied_labelled_sites=len(table),
            training_sites=len(fit),validation_sites=len(validation),test_sites=len(test),
            training_sites_by_dataset=table.iloc[fit].dataset.value_counts().to_dict(),
            supplied_sites_by_dataset=table.dataset.value_counts().to_dict(),
            training_positives=int(y[fit].sum()),class_weight=('balanced' if weight!=1. else 'none'),positive_weight=weight,
            grouping='Global gene IDs and transcript fallbacks across dataset0/1; matching dataset2 mixture positions kept together.',
            heldout_metrics_model='The saved model.joblib is the exact checkpoint used for held-out evaluation; no all-data refit.',
            limitations=['Dataset2 grouping assumes same position/sequence represents the same underlying site; gene/read provenance is unknown.',
                         'Dataset0/1 transcript-to-gene mappings assume compatible reference annotation.',
                         'Unweighted pooling and pooled validation PR-AUC are dominated by larger datasets.',
                         'Previously explored datasets make the evaluation exploratory.',
                         'Full submission files include training, validation, and test sites; evaluation metrics use test sites only.'])
        artifact['report']=report
        joblib.dump(artifact,out/'model.joblib')
        (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
        print('Saved one final model:',out/'model.joblib',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    inputs=p.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--clean-dirs',nargs='+',help='Cleaned dataset0/1/2 directories')
    inputs.add_argument('--data-files',nargs='+',help='Raw dataset0/1/2 JSON or JSON.gz files')
    p.add_argument('--label-files',nargs='+',help='Matching labels in the same order as --data-files')
    p.add_argument('--model-type',choices=['gb','mlp','mil'],default='mlp')
    p.add_argument('--out-dir',help='Default: results/<model-type>')
    p.add_argument('--max-reads',type=int,default=64)
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--patience',type=int,default=6)
    p.add_argument('--batch-size',type=int,default=256,help='Neural baseline and held-out inference batch size; search also tests other sizes')
    p.add_argument('--learning-rate',type=float,default=.001,help='Neural baseline learning rate; search also tests other values')
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--folds',type=int,default=5)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--class-weight',choices=['none','balanced'],default='none',help='GB weighting and neural baseline weighting; neural search also compares both')
    p.set_defaults(selection_metric='pr_auc')
    args=p.parse_args()
    if args.out_dir is None: args.out_dir=f'results/{args.model_type}'
    if args.clean_dirs and args.label_files:
        p.error('--label-files requires --data-files')
    if args.data_files and (not args.label_files or len(args.label_files)!=len(args.data_files)):
        p.error('Supply matching --label-files entries for all --data-files')
    try: train_unified(args)
    except (ValueError,OSError) as error: p.error(str(error))


if __name__=='__main__':
    main()
