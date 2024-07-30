from sklearn.datasets import load_iris
import pandas as pd

# iris 데이터셋 불러오기
iris = load_iris()

# 데이터 프레임 변환
iris_df = pd.DataFrame(data = iris.data, columns = iris.feature_names)
iris_df['species'] = iris.target

print(iris_df)

# species 컬럼을 문자열로 변환 (0: setosa, 1: versicolor, 2: virginica)
species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
iris_df['species'] = iris_df['species'].map(species_map)

# 데이터프레임 출력
print(iris_df.head())