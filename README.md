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
```python 
from sklearn.metrics import confusion_matrix
import itertools

from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Dense, Flatten, BatchNormalization
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

model = Sequential()

# layer 1
model.add(Conv2D(filters= 8, kernel_size = (5,5), padding = 'Same', activation = 'relu', input_shape = (28,28,1)))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Dropout(0.25))

# layer 2
model.add(Conv2D(filters = 16, kernel_size = (3,3), padding = 'Same', activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Dropout(0.2))

# layer 3
model.add(Conv2D(filters = 32, kernel_size= (3,3), padding = 'Same', activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Dropout(0.2))

# fully connected layer 
model.add(Flatten())
model.add(Dense(256, activation= 'relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation = 'softmax'))




