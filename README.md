## Digit recognition with CNN

#### Step 1: Download dataset from Kaggle to Google Colab 
File: Import_dataset_from_kaggle.ipynb 

```python
!pip install kaggle 

Requirement already satisfied: kaggle in /usr/local/lib/python3.6/dist-packages (1.5.10)
Requirement already satisfied: six>=1.10 in /usr/local/lib/python3.6/dist-packages (from kaggle) (1.15.0)
Requirement already satisfied: certifi in /usr/local/lib/python3.6/dist-packages (from kaggle) (2020.12.5)
Requirement already satisfied: python-slugify in /usr/local/lib/python3.6/dist-packages (from kaggle) (4.0.1)
Requirement already satisfied: tqdm in /usr/local/lib/python3.6/dist-packages (from kaggle) (4.41.1)
Requirement already satisfied: requests in /usr/local/lib/python3.6/dist-packages (from kaggle) (2.23.0)
Requirement already satisfied: python-dateutil in /usr/local/lib/python3.6/dist-packages (from kaggle) (2.8.1)
Requirement already satisfied: urllib3 in /usr/local/lib/python3.6/dist-packages (from kaggle) (1.24.3)
Requirement already satisfied: text-unidecode>=1.3 in /usr/local/lib/python3.6/dist-packages (from python-slugify->kaggle) (1.3)
Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.6/dist-packages (from requests->kaggle) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.6/dist-packages (from requests->kaggle) (2.10)

```
Upload kaggle.json file which is download from the kaggle personal page: 

```python
from google.colab import files
files.upload()

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/

# change the permission
!chmod 600 ~/.kaggle/kaggle.json

```
Then, download the dataset by the command: 

```python 
!kaggle competitions download -c digit-recognizer
```
The dataset contains two files: train.csv and test.csv

Read the data:

```python 
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.head()
 ```
 ![upload data](https://user-images.githubusercontent.com/69978820/106286831-6237f800-6246-11eb-8100-2669b2f7b37a.png)
Visualize data: 
```python 
plt.figure(figsize = (15,8))
g = sns.countplot(y_train)
plt.title('Number of digits classes',fontsize = 20, color = 'b')
```
![visualize](https://user-images.githubusercontent.com/69978820/106287350-f86c1e00-6246-11eb-8565-026d9e464511.png)


#### Step 2: Build CNN model to recognize images 
File: Digit recognition code.ipynb 
 - Two models (with and without using Batch normalizations) are built.
 - Optimizer method: Adam with learning rate lr = 0.001
 - Metric: accuracy
 - Batch size: 128
 - Number of epoches: 30
 
 Results: 
 - The accuracy corresponding to the first model (with batch normalization) is up to 99.17 % in the validation set
 - The accuracy corresponding to the second model (without batch normalization) is up to 99.11 % in the validation set, which a less than the one in the first case. 
 
 ![accuracy](https://user-images.githubusercontent.com/69978820/106288760-bc39bd00-6248-11eb-942b-dbe9f9b4b2f3.png)

#### Step 3: Load and apply the pre-trained model to recognize the new images in the test set
File: Load model and predict.ipynb

```python 
import pandas as pd 
import numpy as np
```
Read the test data: 
```python
test = pd.read_csv('test.csv')
```
Normalize data: 

```python
test = test/255
```
Reshape to size (28,28)

```python 
X_test = test.values.reshape(-1, 28, 28, 1)
```
Load model and weights which are saved in file ``Digit recognition code.ipynb```

```python 
from keras.models import load_model

model_1 = load_model('model_case1.h5')
model_1.load_weights('weights_case1.hdf5')

y_test_pred = model_1.predict(X_test)
classes_test_pred = np.argmax(y_test_pred, axis = 1)
```

Compare randomly the predicted results to the true images:
```python 
import matplotlib.pyplot as plt 
import random

ind = random.randint(0, len(X_test))
print('At the index: ', ind)
print('Predicted value is: ', classes_test_pred[ind])

print('The real value is: ')
plt.imshow(X_test[ind].reshape(28,28))
```
![predict](https://user-images.githubusercontent.com/69978820/106289854-0bccb880-624a-11eb-995f-205e9bd9adf1.png)
