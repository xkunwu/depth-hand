- \[iso_cube::transform\] re-center the poses is enough. Rescale poses to range [-1, 1] can produce worse results, as shown below:
![Alt text](detection_base_regre_47.png?raw=true "Optional Title")
- \[train_abc::evaluate\] using different batch size during evaluating (which is useful, especially for streaming) will mysteriously producing small clustered result:
![Alt text](detection_base_regre_14.png?raw=true "Optional Title")
- \[train_abc::evaluate\] due to unknown TF bug, 'is_training' still has to be true while evaluating, otherwise leads to junk result:
![Alt text](detection_base_regre_1.png?raw=true "Optional Title")
