# crf

A pure python implementation of the Linear-Chain Conditional Random Fields

## Dependencies

- Numpy
- Scipy

## Usage

You can test this code with [CoNLL 2000 Chunking Data](https://www.clips.uantwerpen.be/conll2000/chunking/).

### Training

```sh
# format
python3 crf_train.py <train_file> <model_file>

# example
python3 crf_train.py data/chunking_small/small_train.data small_model.json
```

### Test

```sh
# format
python3 crf_test.py <test_file> <trained_model_file>

# example
python3 crf_test.py data/chunking_small/small_test.data small_model.json
```

## Benchmark Result

- Data: CoNLL corpus
    - [data/chunking_full](https://github.com/lancifollia/tiny_crf/tree/master/data/chunking_full): original data (8936 sentences)
    - [data/chunking_small](https://github.com/lancifollia/tiny_crf/tree/master/data/chunking_small): sampled data (77 sentences)
- Compared with [CRF++](http://taku910.github.io/crfpp/)
- Use feature set

**Accuracy**

|                | crf |  CRF++   |
|--------------- | -------- | -------- |
| chunking_full  | 0.960128 | 0.960128 |
| chunking_small | 0.899072 | 0.889474 |

## License
MIT

## Reference
An Introduction to Conditional Random Fields / Charles Sutton, Andrew McCallum/ 2010
