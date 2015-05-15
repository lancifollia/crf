# laon_crf
Python implementation of Linear-Chain Conditional Random Fields

## 사용방법
소스코드에 포함된 코퍼스(CoNLL 2000 Chunking Data)를 이용해서 바로 테스트해 볼 수 있습니다.

Numpy, Scipy가 필요합니다.

#### (당분간의) 유의사항
* python3에서만 동작합니다.
* CoNLL 코퍼스와 동일한 포맷의 데이터만 다룰 수 있습니다.
* 속도가 느립니다.
* 메모리를 많이 사용합니다.

#### 훈련
```sh
$python3 crf_train.py train_file model_file
```
예를 들어, 아래의 명령어를 입력하시면 첨부된 CoNLL 코퍼스로 CRF 모델을 훈련할 수 있습니다.
```sh
$python3 crf_train.py data/chunking_small/small_train.data small_model.json
```

#### 테스트
```sh
$python3 crf_train.py train_file model_file
```
예를 들어, 아래의 명령어를 입력하시면 위에서 훈련시킨 CRF 모델의 성능을 테스트할 수 있습니다.
```sh
$python3 crf_test.py data/chunking_small/small_test.data small_model.json
```

## 성능
CoNLL 코퍼스(data/chunking_small, data/chunking_full)로 CRF++( http://taku910.github.io/crfpp/ )와 비교해 본 결과, 비슷한 성능이 나왔습니다.

(동일한 데이터, 동일한 Feature set을 사용했습니다.)

* data/chunking_full: CoNLL 2000 Chunking Data 원본 (8936 문장)
* data/chunking_small: CoNLL 2000 Chunking Data의 일부로 만든 소량의 코퍼스 (77 문장)

정확도(정답레이블수/전체레이블수)

|                | laon-crf |  CRF++   |
|--------------- | -------- | -------- |
| chunking_full  | 0.960128 | 0.960128 |
| chunking_small | 0.899072 | 0.889474 |

* 유의사항: 아직 laon-crf의 속도가 많이 느립니다. chunking_full로 훈련할 경우, Intel Core i7(2.9GHz) 기준으로 24시간 걸렸습니다.

## License
MIT
