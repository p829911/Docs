### Isolation Forest

[link](https://en.wikipedia.org/wiki/Isolation_forest)

`Isolation forest` 는 대부분에 정상 데이터에 집중하는 기법들과는 다르게 격리된 이상치들에 집중하는 `anomaly detection` 알고리즘이다.   

통계적으로 `anomaly (outlier)` 는 다른 관측치 또는 이벤트와는 다른 평균으로 부터 생성되었다고 생각이 될 만큼 멀리 떨어진 관측치 또는 이벤트이다.  
![fig1](https://user-images.githubusercontent.com/17154958/76717459-c5c44e80-6776-11ea-8946-dd928786412c.png)

예를 들면, 위의 그래프는 한 달 동안 세시간 간격으로 웹 서버에 들어가려는 접속 요청 수를 추적한 그래프이다. 빨간색으로 표시된 몇개의 포인트들은 한눈에 보기에도 비 정상적으로 보여지고, 저 시간에 웹서버가 공격을 당하고 있다고 생각할 수 있는 의심을 갖게 해줄 수 있다. 반면에 빨간색 화살표가 가리키는 평평한 구간도 정상처럼 보이지는 않는데, 저 시간에 웹서버가 다운되었다는 신호일 가능성이 있다.

빅데이터 셋에서의 `Anomalies` 는 매우 복잡한 패턴을 따를 수 있다. 그렇기 때문에 거의 대부분의 케이스가 눈으로 이상치들을 잡아낼 수 없다. 이것이 바로 `anomaly detection`이란 영역에 머신러닝 기법을 쓰는 큰 이유이기도 하다.  

대부분의 많은 이상 탐지의 기법들은 `normal` 이 무엇인지 정의하는데 초점을 맞추고 있다. 이상치는 데이터셋 내에서 정상이 아닌 것으로 분류된 것들이다. `Isolation Forest`는 다른 접근방식을 가지고 있다. 정상 관측치에 맞춰진 모델을 생성하려고 노력하는 것이 아니고, 데이터 셋 내에서 명백하게 고립된 이상 포인트들을 가지고 모델을 생성한다.  `Isolation Forest` 기법은 `normal` 데이터에 초점을 맞춘 다른 기법들과는 다르게 `sampling` 기법을 활용할 수 있다는게 가장 큰 장점인데, 이 때문에 적은 메모리로 빠른 알고리즘을 생성할 수 있게 된다.

#### History

`Isolation Forest` 알고리즘은 2008년 `Fei Tony Liu, Kai Ming Ting and Zhi-Hua Zhou` 에 의해 [논문](https://ieeexplore.ieee.org/document/4781136)에서 처음 발표된 알고리즘이다. 저자들은 샘플 내에서 이상 데이터 포인트의 두가지 양적인 특성을 이용하였다.

1. Few - 이상치는 매우 소수이며
2. Different - 이상치는 속성이나 밸류가 정상 데이터와는 매우 다르다.

이상치는 `few and different` 하기 때문에 그것들은 쉽게 정상데이터와 구별이 가능하다.  

`Isolation Forest` 는 `Isolation Trees` 의 ensemble 모델이다. `anomalies` 라는 것은 `iTrees`의 평균 깊이가 짧은 것으로 정의된다.  

2012년에 발표된 동일 저자의 논문에서는 `iForest` 가 다음을 증명하는 일련의 실험들을 설명했다.

- 적은 메모리 요구와 선형시간 복잡도.
- 관련 없는 속성들을 가지고 있는 고차원 데이터 처리 가능
- 이상치의 유무와 관계없는 모델 훈련 가능
- 재훈련 없이 다양한 수준의 탐지 결과 제공

2013년에는 `Zhiguo Ding, Minrui Fei`가 `iForest(Isolation Forest)` 기반 Streaming data 내의 이상 데이터 탐지 문제를 해결하는 프레임워크를 제시하였다. 이에 관해 더 나아간 `Tan et al, Susto et al, Weng et al` 의 논문이 있다.  

이상탐지에 있어서 `iForest`의 주요한 문제는 모델 자체에 있는 것이 아니라 `anomaly score`를 구하는 부분에 있었다. 이 문제는 2018년 `Sahand Hariri, Matias Carrasco Kind, Roberg J. Brunner`이 제기하였고, 이것을 보완한 `Extended Isolation Forest (EIF)` 를 제안했다. 또한 저자들은 이 논문에서 원래 모델의 개선 사항과 주어진 데이터 포인트에 대해 생성된 `anomaly score` 의 일관성과 신뢰성을 향상시킬 수 있는 방법도 설명한다.

#### Algorithm

`Isolation Forest algorithm` 은 정상데이터 보다 이상데이터가 나머지 데이터와 나누는 것이 더 쉽다는것을 기반으로 한다. 데이터 포인트를 분리하기 위해 알고리즘은 속성을 무작위로 선택한 다음 그 속성의 최대값과 최소값 사이에서 속성에 대한 분할 값을 임의로 선택하여 샘플에서 파티션을 반복적으로 생성한다.

![fig2](https://user-images.githubusercontent.com/17154958/76719663-be547380-677d-11ea-96b6-c10921e43eb1.png)

![fig3](/home/p829911/.config/Typora/typora-user-images/image-20200316120145725.png)

위의 그래프는 정규분포로 생성된 포인트들에서 정상 포인트(fig2) 를 나눌 때 보다 비정상 포인트(fig3)를 나눌 때 랜덤 파티션이 더 작다는 것을 보여 준다.  

수학적 관점에서 `Isolation Tree` 로 데이터 포인트를 유일하게 분리할 수 있도록 만들어진 파티션의 수는 트리의 루트에서 종료 노드까지의 거리이다. 위의 그림에서 fig2가 fig3 보다 트리 depth가 더 크다.  

`iTree`를 구축하기 위해 알고리즘은 노드에 하나의 인스턴스만 있거나 노드의 모든 데이터가 동일한 값을 가질 때까지 속성 q와 분할 값 p를 임의로 선택하여 트리를 재귀적으로 나눈다.

#### Properties of Isolation Forest

- **Sub-sampling**: `iForest` 는 모든 정상 인스턴스를 분리 할 필요가 없으므로 대부분의 학습 샘플을 무시할 수 있다. 결과적으로, `iForest` 는 샘플링 크기가 작게 유지 될 때 잘 작동한다. 이는 일반적으로 큰 샘플링 크기를 선호하는 다른 대부분의 알고리즘과는 구별된다.
- **Swamping**: 정상 인스턴스가 이상치에 매우 가까우면 이상치를 분류하기 위해 파티션이 증가하는데 이 현상을 `swamping` 이라고 한다. 정상과 이상 구분을 어렵게 만드는 요인이다. `swamping` 의 주요한 이유 중 하나는 `anomaly detection` 하기 위한 데이터가 많을 때 생기는데 이때 스이는 기법이 `sub-sampling` 이다. `iForest` 는 성능의 관점에서 `sub-sampling` 에 잘 반응하므로 데이터 포인트를 줄이는 것이 `swamping` 의 효과를 줄이는 좋은 방법이다.
- **Masking**: 이상치의 숫자가 많을 때 일부 집단이 밀도가 높고 큰 군집을 이루고 있다면, 단일 이상을 분리하고 이상치를 찾기 더 어려워질 수 있다. `swamping`과 마찬가지로 샘플의 수가 많을 때 나타날 가능성이 높으며 `sub-sampling` 을 통해 완화할 수 있다.
- **High Dimensional Data**: 거리 기반의 알고리즘들은 높은 차원의 데이터셋을 다룰 때 비효율적이다. 이것의 주된 이유는 높은 차원에서 모든 포인트들은 희박하기 때문이다. 불행하게도 `iForest`에서도 높은 차원에 데이터가 이러한 문제를 주지만, `Kurtosis`와 같은 샘플 공간의 차원을 줄이는 feature selection test를 추가하면 성능을 크게 향상시킬 수 있다.
- **Normal Instance Only**: `iForest` 는 훈련 데이터 셋에 어떠한 이상치가 들어있지 않아도 잘 작동한다. `iForest` 는 데이터의 분포를 $h(x_i)$ (root node부터 leaf node까지) 의 거리(degree)로 설명하기 때문이다.

