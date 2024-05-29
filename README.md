# :sparkles: :sparkles: <u>ORCDF</u>-KDD 2024 :sparkles::sparkles:

:smile_cat: Welcome to ORCDF, this is a comprehensive repository specializing in ***ORCDF: An Oversmoothing-Resistant Cognitive Diagnosis
Framework for Student Learning in Online Education Systems*** published in KDD 2024.





# Example

Here, we provide an example for running ORCDF. The steps for running the other benchmarks are the same.

Run the code:

```
python exp_orcdf.py --method=orcdf --if_type=ncd --data_type=Assist17 --ssl_temp=0.5  --ssl_weight=1e-3  --flip_ratio=0.15  --test_size=0.2 --seed=0 --batch_size=4096 --device=cuda:0 --gcn_layers=3 --epoch=12  --keep_prob=1.0  --lr=4e-3 --weight_decay=0
```

- if_type: To control which Cognitive Diagnosis Model  to integrate into ORCDF, options like kancd, ncd, and others can be selected.
- gcn_layer: Controlling the number of layers in the Response-aware Graph Convolution (RGC).



# Reference :thought_balloon:

Hong Qian, Shuo Liu, Mingjia Li, Bingdong Li, Zhi liu, Aimin Zhou "ORCDF: An Oversmoothing-Resistant Cognitive Diagnosis Framework for Student Learning in Online Education Systems." In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2024.



## Bibtex

## 

```
@inproceedings{liu2024kdd,
author = {Hong Qian, Shuo Liu, Mingjia Li, Bingdong Li, Zhi liu, Aimin Zhou},
booktitle = {Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
title = {ORCDF: An Oversmoothing-Resistant Cognitive Diagnosis Framework for Student Learning in Online Education Systems},
year = {2024},
address={Barcelona, Spain}
}
```

