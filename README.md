# DualMatch

Source code for ["Unsupervised Entity Alignment for Temporal Knowledge Graphs"](https://doi.org/10.1145/3543507.3583381). The ACM Web Conference 2023.

A 3-minute short video briefly describing our method can be found [Here](https://youtu.be/kqSx9mdj6Co). 

## Dataset

We thank [TEA-GNN](https://github.com/soledad921/TEA-GNN) for providing the datasets.

-ICEWS05-15

-YAGO-WIKI50K

-YAGO-WIKI20K



    ent_ids_1: ids for entities in source KG;
    ent_ids_2: ids for entities in target KG;
    ref_ent_ids: entity links encoded by ids;
    triples_1: relation triples encoded by ids in source KG;
    triples_2: relation triples encoded by ids in target KG;
    rel_ids_1: ids for relations in source KG;
    rel_ids_2: ids for relations in target KG;
    sup_pairs + ref_pairs: entity alignments
    
    

## Run
To perform EA on ICEWS05-15 in unsupervised manner:

`python main.py --ds 0 --unsup`

To perform EA on YAGO-WIKI50K in Less seed setting:

`python main.py --ds 1 --train_ratio 20`

To perform EA on YAGO-WIKI20K in Normal setting and evaluate separately:

`python main.py --ds 2 --sep_eval`

**Note:** Part of the code needs to be run on CPU. We plan to fix this issue in the future.


## Acknowledgements
 
We refer to the code of Dual-AMN, TEA-GNN, and SEU.
Thanks for their great contributions!

Code credit to [Xiaoze Liu](https://github.com/joker-xii) and [Junyang Wu](https://github.com/Immortals88/). 

## Citation

If you find this work useful, please cite

```
    @inproceedings{DBLP:conf/www/LiuW00G23,
      author       = {Xiaoze Liu and
                      Junyang Wu and
                      Tianyi Li and
                      Lu Chen and
                      Yunjun Gao},
      title        = {Unsupervised Entity Alignment for Temporal Knowledge Graphs},
      booktitle    = {Proceedings of the {ACM} Web Conference 2023, {WWW} 2023, Austin,
                      TX, USA, 30 April 2023 - 4 May 2023},
      pages        = {2528--2538},
      publisher    = {{ACM}},
      year         = {2023},
      url          = {https://doi.org/10.1145/3543507.3583381},
      doi          = {10.1145/3543507.3583381},
    }
```


