# Building built in minutes- SfM and NeRF



## Phase 1 - Structure from Motion:

1. Change the directory to Phase 1.
2. Run
```
python Wrapper.py
```


## Phase 2 - NeRF:
Implementing the original NERF method [from this paper](https://arxiv.org/abs/2003.08934).

### Input:
Download the lego data for NeRF from the original author’s link [here](https://drive.google.com/drive/folders/1lrDkQanWtTznf48FCaW5lX9ToRdNDF1a)

#### Training:
1. Change the directory to Phase 2.
2. To train the NeRF model : Run

```
python NeRF_train.py
```


#### Testing
1. Change the directory to Phase 2.
2. To test the model: Run

```
python NeRF_test.py
```






