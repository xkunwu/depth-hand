- \[iso_cube::transform\] re-center the poses is enough. Rescale poses to range [-1, 1] can produce worse results, as shown below:
![Alt text](detection_base_regre_47.png?raw=true "Optional Title")
- \[train_abc::evaluate\] using different batch size during evaluating (which is useful, especially for streaming) will mysteriously producing small clustered result:
![Alt text](detection_base_regre_14.png?raw=true "Optional Title")
- \[train_abc::evaluate\] due to unknown TF bug, 'is_training' still has to be true while evaluating, otherwise leads to junk result:
![Alt text](detection_base_regre_1.png?raw=true "Optional Title")
    - found to be related to batch_norm:
        - updates_collections=None, for in place update.
        - use slim.learning.create_train_op
- re-projection error
![Alt text](draw_dense_regre_742.png?raw=true "Optional Title")
[[-4.17668172 -6.24533534 32.56089077]
 [-4.76440158 -6.36739253 33.63058828]
 [-6.16442893 -6.86405792 67.36764189]
 [-6.29532547 -6.65877162 68.8288    ]
 [-6.29532547 -6.65877162 68.8288    ]
 [-6.8900968  -6.53940754 76.308     ]
 [-4.12563044 -6.67310609 56.66111054]
 [-4.71147409 -6.68427112 51.96      ]
 [-5.26238569 -6.59923033 48.41769644]
 [-5.19826968 -5.72312771 63.8892    ]
 [-3.68644679 -7.17109072 64.77629994]
 [-3.4436682  -7.10667396 58.52383949]
 [-5.91128772 -6.57089076 64.9392    ]
 [-3.81054307 -5.56034197 58.16207104]
 [-3.34923822 -5.95660589 62.27631551]
 [-5.95389957 -6.86265039 66.4392    ]
 [-6.41993632 -6.50252858 53.08594769]
 [-4.71147409 -6.68427112 51.96      ]
 [-4.20524721 -7.2341818  84.35567958]
 [-3.65881542 -7.05269531 85.54330899]
 [-3.69075801 -6.42141192 89.75553504]]

