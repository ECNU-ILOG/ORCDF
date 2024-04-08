# Implemention of ORCDF: An Oversmoothing-Resistant Cognitive Diagnosis Framework for Student Learning in Online Education Systems





# Update :fire::fire:

> We have updated the Junyi, Nips20, and XES3G5M datasets. If there are any further questions from the reviews, please feel free to raise them, and we will provide detailed responses after April 11 (AOE).







# Example

Here, we provide an example for running ORCDF. The steps for running the other benchmarks are the same.

Run the code:

```
python exp_orcdf.py --method=orcdf --if_type=ncd --data_type=Assist17 --ssl_temp=0.5  --ssl_weight=1e-3  --flip_ratio=0.15  --test_size=0.2 --seed=0 --batch_size=4096 --device=cuda:0 --gcn_layers=3 --epoch=12  --keep_prob=1.0  --lr=4e-3 --weight_decay=0
```

- if_type: To control which Cognitive Diagnosis Model  to integrate into ORCDF, options like kancd, ncd, and others can be selected.
- gcn_layer: Controlling the number of layers in the Response-aware Graph Convolution (RGC).
