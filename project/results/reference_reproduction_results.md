| model | train_precision | train_recall | train_f1 | test_precision | test_recall | test_f1 | class_1_train_precision | class_2_train_precision | class_3_train_precision | class_4_train_precision | class_5_train_precision | class_1_test_precision | class_2_test_precision | class_3_test_precision | class_4_test_precision | class_5_test_precision | num_trees |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| linear_svm_ovr | 0.6106 | 0.6091 | 0.586 | 0.4234 | 0.4542 | 0.4176 | 0.6303 | 0.6183 | 0.6162 | 0.125 | 0.75 | 0.4835 | 0.4572 | 0.25 | 0.0 | 0.0 | nan |
| ovo_linear_svm | 0.6247 | 0.6082 | 0.5763 | 0.4077 | 0.4819 | 0.4404 | 0.5825 | 0.6247 | 0.8818 | 0.0 | 0.75 | 0.4976 | 0.4818 | 0.0 | 0.0 | 0.0 | nan |
| ovo_rbf_svm | 0.5482 | 0.6193 | 0.5652 | 0.4027 | 0.4821 | 0.4382 | 0.6162 | 0.6218 | 0.25 | 0.0 | 0.0 | 0.4867 | 0.4806 | 0.0 | 0.0 | 0.0 | nan |
| custom_kernel_svm | 0.6618 | 0.6305 | 0.6016 | 0.39 | 0.4596 | 0.4205 | 0.6114 | 0.6374 | 0.8818 | 0.625 | 1.0 | 0.4846 | 0.4528 | 0.0 | 0.0 | 0.0 | nan |
| decision_tree | 0.984 | 0.9833 | 0.9833 | 0.4079 | 0.4151 | 0.4089 | 0.9606 | 1.0 | 1.0 | 1.0 | 1.0 | 0.4768 | 0.4524 | 0.1364 | 0.0 | 0.3333 | nan |
| bagged_decision_tree_100 | 0.9837 | 0.9833 | 0.9833 | 0.4522 | 0.4764 | 0.4557 | 0.9689 | 0.9934 | 1.0 | 0.9688 | 1.0 | 0.5113 | 0.4821 | 0.3006 | 0.0 | 0.0 | 100.0 |
