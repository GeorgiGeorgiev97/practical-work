# Multi-head CNN for OOD

Reimplementation of the multi-head CNN



## Files

```
__init__.py
mheads.py                MHeads and DenseNetMHeads classes
densenet_components.py   Bottleneck, SingleLayer, Transition, make_dense_block
base_model.py            base class
errors.py                custom exceptions

train_oct.py                 main training and evaluation script
requirements.txt             dependencies
```

---

## Quick start

### Install dependencies

```bash
pip install -r requirements.txt
```

### Download Datasets and adjust data paths in train.py

```python
oct_train_dir = '...OCT2017/train'   # Kermany dataset training folder
oct_test_dir  = '...OCT2017/test'    # Kermany dataset test folder
octid_dir     = '...OCTID'           # OCTID dataset folder (OOD)
```

### Run training and evaluation

```bash
python train.py
```

