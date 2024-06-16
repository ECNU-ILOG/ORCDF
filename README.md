- # :sparkles: :sparkles: <u>ORCDF</u>-KDD 2024 :sparkles::sparkles:


:smile_cat: Welcome to ORCDF, this is a comprehensive repository specializing in ***ORCDF: An Oversmoothing-Resistant Cognitive Diagnosis
Framework for Student Learning in Online Education Systems*** published in KDD 2024.

![Sample Image](./img/cd.png)

# Abstract and Contribution :flight_arrival:

Cognitive diagnosis models (CDMs) are designed to learn students' mastery levels using their response logs. CDMs play a fundamental role in online education systems since they significantly influence downstream applications such as teachers' guidance and computerized adaptive testing. Despite the success achieved by existing CDMs, we find that they suffer from a thorny issue that the learned students' mastery levels are too similar. ***This issue, which we refer to as oversmoothing, could diminish the CDMs' effectiveness in downstream tasks.*** 

CDMs comprise two core parts: learning students' mastery levels and assessing mastery levels by fitting the response logs. 

<u>**This paper contends that the oversmoothing issue arises from that existing CDMs seldom utilize response signals on exercises in the learning part but only use them as labels in the assessing part.**</u> 

To this end, this paper proposes an oversmoothing-resistant cognitive diagnosis framework (ORCDF) to enhance existing CDMs by utilizing response signals in the learning part. 

- Specifically, ORCDF introduces a novel response graph to inherently incorporate response signals as types of edges. Then, ORCDF designs a tailored response-aware graph convolution network (RGC) that effectively captures the crucial response signals within the response graph. 
- Via ORCDF, existing CDMs are enhanced by replacing the input embeddings with the outcome of RGC, allowing for the consideration of response signals on exercises in the learning part. 
- Extensive experiments on real-world datasets show that ORCDF not only helps existing CDMs alleviate the oversmoothing issue but also significantly enhances the models' prediction and interpretability performance. Moreover, the effectiveness of ORCDF is validated in the downstream task of computerized adaptive testing.



# :small_red_triangle: ​Oversmoothing in Cognitive Diagnosis :small_red_triangle:

Despite the success, ***this paper, for the first time, identifies that existing CDMs share a potential and thorny issue that the learned Mas of students are too similar***. ***We refer to this issue as oversmoothing.*** Oversmoothing could diminish the CDMs' effectiveness in down-stream tasks.  

To support the motivation of this paper and reveal the oversmoothing issue, we conduct a pilot study on four real-world datasets collected from the online education systems, ensuring a diverse range of circumstances in the students' response logs. 

> We consider the result at the final step, which is input into the interaction function, as the student's mastery level learned by the CDMs.  For more details, you can refer to the `get_mastery_level` function in the code repository.

The learned mastery level of students by CDMs is a matrix with the number of rows equal to the number of students and the number of columns equal to the number of knowledge concepts.

We propose a metric called mean **<u>normalized difference (MND)</u>**. 

![Sample Image](./img/mnd.png)

```python
 def mean_average_distance(mastery_level, eval_func='mse'):
        n = mastery_level.shape[0]
        if eval_func == 'mse':
            row_sums = np.sum(mastery_level ** 2, axis=1)
            sum_square_diff = row_sums[:, np.newaxis] + row_sums - 2 * np.dot(mastery_level, 			              mastery_level.T)
            sum_square_diff = np.maximum(sum_square_diff, 0)
            rmse = np.sqrt(sum_square_diff / mastery_level.shape[1])
            return np.sum(rmse) / n / (n - 1)
        else:
            dot_product = np.dot(mastery_level, mastery_level.T)
            norms = np.linalg.norm(mastery_level, axis=1)
            norm_product = np.outer(norms, norms)
            cosine_similarity_matrix = dot_product / norm_product
            return np.sum(np.ones(shape=(n, n)) - cosine_similarity_matrix) / n / (n - 1)
```

Intuitively, the larger the MND value, the bigger the difference among students' Mas that learned by CDMs. 

![Sample Image](./img/oversmoothing.png)

**As shown in the above Figure, although CDMs such as NCDM, CDMFKC, KSCD and KaNCD achieve commendable prediction performance, the MND values of Mas they have learned are quite small and hard to distinguish.** 

Since CD is an upstream task, addressing this issue is urgent. For instance, if teachers rely on the outcomes of CD to assist student development, exceedingly subtle distinctions could lead to confusion. Intuitively, if MND is 0.005, it implies that the average difference in Mas for two students in a class on certain concepts is merely 0.005 (e.g., 0.51 and 0.515). Such a small margin could potentially bring difficulty to teachers to accurately assess the cognitive state of entire class. This not only fails to aid students but could also result in misguided instruction. Moreover, for downstream algorithms, a diagnosis result plagued by oversmoothing may lead to erroneous recommendations of learning materials, causing irreversible impacts on students.







# Example

Here, we provide an example for running ORCDF. The steps for running the other benchmarks are the same.

Run the code:

```
python exp_orcdf.py --method=orcdf --if_type=ncd --data_type=Assist17 --ssl_temp=0.5  --ssl_weight=1e-3  --flip_ratio=0.15  --test_size=0.2 --seed=0 --batch_size=4096 --device=cuda:0 --gcn_layers=3 --epoch=12  --keep_prob=1.0  --lr=4e-3 --weight_decay=0
```

- if_type: To control which Cognitive Diagnosis Model  to integrate into ORCDF, options like kancd, ncd, and others can be selected.
- gcn_layer: Controlling the number of layers in the Response-aware Graph Convolution (RGC).



# Metric

| Dataset  | Metric (%) | IRT OL | IRT OR    | MIRT OL | MIRT OR   | NCDM OL | NCDM OR   | CDMFKC OL | CDMFKC OR | KSCD OL | KSCD OR   | KANCD OL | KANCD OR  |
| -------- | ---------- | ------ | --------- | ------- | --------- | ------- | --------- | --------- | --------- | ------- | --------- | -------- | --------- |
| Assist17 | AUC        | 88.95  | **89.60** | 91.42   | **91.95** | 86.89   | **89.94** | 87.30     | **90.02** | 88.56   | **89.68** | 88.56    | **90.33** |
|          | ACC        | 86.11  | 86.75     | 88.15   | **88.51** | 84.56   | **87.10** | 85.15     | **87.2**  | 86.14   | **86.75** | 86.06    | **87.56** |
|          | DOA        | -      | -         | -       | -         | 51.39   | **66.76** | 54.69     | **66.67** | 65.86   | **68.05** | 62.86    | **67.01** |
|          | MND        | -      | -         | -       | -         | 1.43    | **7.57**  | 4.64      | **20.7**  | 0.05    | **2.21**  | 3.51     | **14.08** |
| EdNet-1  | AUC        | 73.18  | **74.56** | 74.41   | **74.68** | 72.86   | **74.81** | 73.05     | **74.85** | 73.74   | **74.66** | 74.42    | **75.11** |
|          | ACC        | 70.89  | **71.85** | 71.70   | **71.89** | 70.60   | **71.98** | 70.73     | **71.95** | 71.52   | **71.85** | 71.75    | **72.07** |
|          | DOA        | -      | -         | -       | -         | 59.31   | **64.29** | 60.45     | **64.01** | 64.55   | **65.07** | 63.02    | **65.47** |
|          | MND        | -      | -         | -       | -         | 1.42    | **4.29**  | 0.82      | **4.05**  | 0.05    | **2.45**  | 5.48     | **7.12**  |
| Junyi    | AUC        | 80.35  | **81.46** | 80.87   | **81.46** | 77.72   | **81.44** | 78.27     | **81.30** | OOM     | OOM       | 79.12    | **81.72** |
|          | ACC        | 76.65  | **77.52** | 77.28   | **77.54** | 74.49   | **77.59** | 74.95     | **77.28** | OOM     | OOM       | 75.57    | **77.71** |
|          | DOA        | -      | -         | -       | -         | 49.92   | **58.19** | 49.92     | **60.74** | OOM     | OOM       | 53.59    | **60.85** |
|          | MND        | -      | -         | -       | -         | 0.51    | **11.22** | 3.04      | **17.18** | OOM     | OOM       | 2.86     | **12.82** |
| XES3G5M  | AUC        | 79.18  | **80.13** | 80.43   | **80.66** | 75.46   | **80.22** | 74.15     | **79.98** | OOM     | OOM       | 79.68    | **80.41** |
|          | ACC        | 81.52  | **82.51** | 82.31   | **82.52** | 81.21   | **82.49** | 80.17     | **82.28** | OOM     | OOM       | 82.23    | **82.44** |
|          | DOA        | -      | -         | -       | -         | 68.01   | **73.93** | 69.03     | **73.89** | OOM     | OOM       | 73.50    | **73.62** |
|          | MND        | -      | -         | -       | -         | 1.04    | **19.37** | 2.83      | **35.26** | OOM     | OOM       | 6.43     | **16.67** |



# Reference :thought_balloon:

Hong Qian, Shuo Liu, Mingjia Li, Bingdong Li, Zhi liu, Aimin Zhou "ORCDF: An Oversmoothing-Resistant Cognitive Diagnosis Framework for Student Learning in Online Education Systems." In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2024.



## Bibtex

```
@inproceedings{liu2024kdd,
author = {Hong Qian, Shuo Liu, Mingjia Li, Bingdong Li, Zhi liu, Aimin Zhou},
booktitle = {Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
title = {ORCDF: An Oversmoothing-Resistant Cognitive Diagnosis Framework for Student Learning in Online Education Systems},
year = {2024},
address={Barcelona, Spain}
}
```

