| Experiment | Key result |
|---|---|
| SessionKV-DR delayed recall (delay=6,facts=4) | baseline=0.125, alc_no_write=0.125, alc=0.719 |
| SessionKV-DR overwrite | baseline=0.125, alc_no_write=0.125, alc=0.375 |
| Core adaptation | recall_mse baseline=0.8795, alc_no_write=0.3284, alc=0.6307 |
|  | recall_acc baseline=0.016, alc_no_write=0.078, alc=0.141 |
| Language-shaped benchmark | recall_mse baseline=0.8530, alc_no_write=0.3536, alc=0.7326 |
| Stability | nan_rate=0.000000, final_norm=0.635 |
| Persistence | slot_diff=0.00e+00, behavior_diff=0.00e+00 |
| Trainability | before=0.0942, after=0.0755 |
| Efficiency | baseline=0.800ms, alc=1.325ms, ratio=1.66x |
