# 3DCG generator evaluation

## Train losses

| track | first_loss | final_loss |
|---|---:|---:|
| track1 | 1.190891981124878 | 0.1046 |
| track2 | 0.69537353515625 | 0.1028 |
| track1_seq | 2.476442337036133 | 0.2734 |
| track2_sdf | 1.8518126010894775 | 0.3767 |

## Scorer quality on held-out synthetic (higher is better)

| track | n | quality mean±std | min | max | latency ms |
|---|---:|---:|---:|---:|---:|
| track1 | 10 | 0.8330±0.0763 | 0.7722 | 0.9829 | 0.7 |
| track2 | 10 | 0.6197±0.0399 | 0.5799 | 0.7308 | 18.4 |
| track1_trained | 10 | 0.8188±0.0884 | 0.6639 | 0.9166 | 3.1 |
| track2_trained | 10 | 0.6834±0.0197 | 0.6443 | 0.7113 | 10.1 |
| track1_sequence | 10 | 0.6279±0.0256 | 0.5674 | 0.6664 | 5.5 |
| track2_sdf | 10 | 0.7082±0.0141 | 0.6829 | 0.7293 | 138.5 |

## unit-box reference

| track | quality | chamfer | volume_iou |
|---|---:|---:|---:|
| track1 | 0.9917 | 0.0 | 1.0 |
| track1_trained | 0.9062 | 0.12028096047927508 | 0.6875 |
| track1_sequence | 0.6262 | 0.2206074343833464 | 0.4166666666666667 |
| track2 | 0.5866 | 0.33371987142445975 | 0.5222800925925926 |
| track2_trained | 0.6580 | 0.19197365564707275 | 0.8052662037037037 |
| track2_sdf | 0.6469 | 0.2647345089825205 | 0.8052662037037037 |

## Notes

- Quality is scorer composite (not training loss).
- Synthetic eval uses a different seed than training.
- track1_sequence includes UV/rig/material ops which scorer may partially score.

