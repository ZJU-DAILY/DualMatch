# DualMatch

Source code for "Unsupervised Entity Alignment for Temporal Knowledge Graphs".

## Dataset

-ICEWS05-15

-YAGO-WIKI50K

-YAGO-WIKI20K



    ent_ids_1: ids for entities in source KG;
    ent_ids_2: ids for entities in target KG;
    ref_ent_ids: entity links encoded by ids;
    triples_1: relation triples encoded by ids in source KG;
    triples_2: relation triples encoded by ids in target KG;
    rel_ids_1: ids for entities in source KG;
    rel_ids_2: ids for entities in target KG;
    sup_pairs + ref_pairs: entity alignments
    
    

## Run
To perform EA on ICEWS05-15 in unsupervised manner:

`python main.py --ds 0 --unsup`

To perform EA on YAGO-WIKI50K in Less seed setting:

`python main.py --ds 1 --train_ratio 20`

To perform EA on YAGO-WIKI20K in Normal setting and evaluate separately:

`python main.py --ds 2 --sep_eval`


## Acknowledgements
 
We refer to the code of Dual-AMN, TEA-GNN, and SEU.
Thanks for their great contributions!

