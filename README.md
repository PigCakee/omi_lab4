# Лабораторная работа #4
## 2. С использованием техники обучения Transfer Learning и оптимальной политики изменения темпа обучения обучить нейронную сеть EfficientNet-B0 (предварительно обученную на базе изображений imagenet) для решения задачи классификации изображений Food-101 с использованием следующих техник аугментации данных:
a. Случайное горизонтальное и вертикальное отображение
b. Использование случайной части изображения
c. Поворот на случайный угол

В данной работе используется фиксированный темп обучения 0.001

```
optimizer=tf.optimizers.Adam(lr=0.001)
```

### 2.a. Случайное горизонтальное и вертикальное отображение
https://tensorboard.dev/experiment/lw3Z3V11Tju2IebnN4PIrg/

```
data_augmentation = tf.keras.Sequential(
    [
        preprocessing.RandomFlip("horizontal_and_vertical", seed=1)
    ]
)
def build_model():
  inputs = tf.keras.Input(shape=(RESIZE_TO, RESIZE_TO, 3))
  x = data_augmentation(inputs)
  x = EfficientNetB0(include_top=False, weights='imagenet', input_tensor = x)
  x.trainable = False
  x = tf.keras.layers.GlobalAveragePooling2D()(x.output)
  outputs = tf.keras.layers.Dense(NUM_CLASSES, activation=tf.keras.activations.softmax)(x)
  return tf.keras.Model(inputs=inputs, outputs=outputs)
```

#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/PigCakee/omi_lab4/main/epoch_categorical_accuracy_2_a.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/PigCakee/omi_lab4/main/epoch_loss_2_a.svg">

Максимальная точность на валидационной выборке (15 эпох) = 66%

### 2.b. Использование случайной части изображения
https://tensorboard.dev/experiment/XjBBluVTQbKAOP0Tx01TlQ/#scalars

```
def parse_proto_example(proto):
  keys_to_features = {
    'image/encoded': tf.io.FixedLenFeature((), tf.string, default_value=''),
    'image/label': tf.io.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64))
  }
  example = tf.io.parse_single_example(proto, keys_to_features)
  example['image'] = tf.image.decode_jpeg(example['image/encoded'], channels=3)
  example['image'] = tf.image.convert_image_dtype(example['image'], dtype=tf.uint8)
  example['image'] = tf.image.resize(example['image'], tf.constant([250, 250]))
  return example['image'], tf.one_hot(example['image/label'], depth=NUM_CLASSES)

data_augmentation = tf.keras.Sequential(
    [
        preprocessing.RandomCrop(112, 112, seed=1)
    ]
)

def build_model():
  inputs = tf.keras.Input(shape=(250, 250, 3))
  x = data_augmentation(inputs)
  x = EfficientNetB0(include_top=False, weights='imagenet', input_tensor = x)
  x.trainable = False
  x = tf.keras.layers.GlobalAveragePooling2D()(x.output)
  outputs = tf.keras.layers.Dense(NUM_CLASSES, activation=tf.keras.activations.softmax)(x)
  return tf.keras.Model(inputs=inputs, outputs=outputs)
```

#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/PigCakee/omi_lab4/main/epoch_categorical_accuracy_2_b.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/PigCakee/omi_lab4/main/epoch_loss_2_b.svg">

Максимальная точность на валидационной выборке (15 эпох) = 43%

### 2.c. Поворот на случайный угол
https://tensorboard.dev/experiment/BepaM6f8Sv2dc6HzeVXD8g/#scalars

```
data_augmentation = tf.keras.Sequential(
    [
        preprocessing.RandomRotation(1, fill_mode='reflect', interpolation='bilinear', seed=1, name=None,fill_value=0.0)
    ]
)
def build_model():
  inputs = tf.keras.Input(shape=(RESIZE_TO, RESIZE_TO, 3))
  x = data_augmentation(inputs)
  x = EfficientNetB0(include_top=False, weights='imagenet', input_tensor = x)
  x.trainable = False
  x = tf.keras.layers.GlobalAveragePooling2D()(x.output)
  outputs = tf.keras.layers.Dense(NUM_CLASSES, activation=tf.keras.activations.softmax)(x)
  return tf.keras.Model(inputs=inputs, outputs=outputs)
```

#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/PigCakee/omi_lab4/main/epoch_categorical_accuracy_2_c.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/PigCakee/omi_lab4/main/epoch_loss_2_c.svg">

Максимальная точность на валидационной выборке (15 эпох) = 65%

### Анализ результатов:

При выборе случайных параметров мы на данном этапе получили следующие результаты максимальной точности: 
1. Случайное горизонтальное и вертикальное отображение - 66%
2. Использование случайной части изображения - 43%
3. Поворот на случайный угол - 65%
Пока выводы делать рано, но уже видно, что 15 эпох достаточно для каждого метода, чтобы график потерь пошел вверх (сеть начала переобучаться)

## 3. Для каждой индивидуальной техники аугментации определить оптимальный набор параметров
### 3.a. Случайное горизонтальное и вертикальное отображение
https://tensorboard.dev/experiment/lw3Z3V11Tju2IebnN4PIrg/

Так как единственным параметром при заданном условии является параметр seed, а его исследовать необходимости нет, то этот эксперимент просто дублирует эксперимент 2.а.

#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/PigCakee/omi_lab4/main/epoch_categorical_accuracy_3_a.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/PigCakee/omi_lab4/main/epoch_loss_3_a.svg">

Максимальная точность на валидационной выборке (15 эпох) = 66%

### 3.b. Использование случайной части изображения

Параметры, доступные для исследования:
```
height
width
```

1. height = 224, width = 224
https://tensorboard.dev/experiment/2M3BReq0SxWhEmbu9gevTA/#scalars&runSelectionState=eyJmMTAxLTE2MTk3MTY0MzkuNDYyODc2Ni90cmFpbiI6ZmFsc2UsImYxMDEtMTYxOTcxNjQzOS40NjI4NzY2L3ZhbGlkYXRpb24iOmZhbHNlfQ%3D%3D
```
data_augmentation = tf.keras.Sequential(
    [
        preprocessing.RandomCrop(224, 224)
    ]
)
```

#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/PigCakee/omi_lab4/main/epoch_categorical_accuracy_3_b_224.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/PigCakee/omi_lab4/main/epoch_loss_3_b_224.svg">

Максимальная точность на валидационной выборке (15 эпох) = 57%

2. height = 150, width = 150
https://tensorboard.dev/experiment/jqDr8EwIRfW6PsRMT4OIbw/#scalars
```
data_augmentation = tf.keras.Sequential(
    [
        preprocessing.RandomCrop(150, 150)
    ]
)
```

#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/PigCakee/omi_lab4/main/epoch_categorical_accuracy_3_b_150.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/PigCakee/omi_lab4/main/epoch_loss_3_b_150.svg">

3. height = 112, width = 112
https://tensorboard.dev/experiment/60kYqx19R1u2sAkvov1yHQ/#scalars&runSelectionState=eyJmMTAxLTE2MTk3MDQwOTQuNTA1MDQzL3RyYWluIjp0cnVlLCJmMTAxLTE2MTk3MDEzMDguMzU4NDA4L3RyYWluIjpmYWxzZSwiZjEwMS0xNjE5NzAxMzA4LjM1ODQwOC92YWxpZGF0aW9uIjpmYWxzZX0%3D
```
data_augmentation = tf.keras.Sequential(
    [
        preprocessing.RandomCrop(112, 112)
    ]
)
```

#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/PigCakee/omi_lab4/main/epoch_categorical_accuracy_3_b_112.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/PigCakee/omi_lab4/main/epoch_loss_3_b_112.svg">

Максимальная точность на валидационной выборке (15 эпох) = 44%

### Анализ результатов
В течение данного эсперимента наблюдается снижение максимальной точности сети с уменьшением размера обрезанного изображения, из чего делаем вывод, что самые подходящие значения для параметров height & width - 224.

### 3.c. Поворот на случайный угол

Параметры, доступные для исследования:
```
factor
fill_mode
interpolation
fill_value
```

1. factor = 1.0
https://tensorboard.dev/experiment/KD50sbiGSZie2UeP0znKfw/#scalars&runSelectionState=eyJmMTAxLTE2MTk2Mjg5NzMuMTQwODMzMS90cmFpbiI6ZmFsc2UsImYxMDEtMTYxOTYyNjkwOC42MTg4NTUvdmFsaWRhdGlvbiI6dHJ1ZSwiZjEwMS0xNjE5NjI2OTA4LjYxODg1NS90cmFpbiI6dHJ1ZSwiZjEwMS0xNjE5NjI5MDEwLjczNjAzMTMvdHJhaW4iOmZhbHNlLCJmMTAxLTE2MTk2MjkwMTAuNzM2MDMxMy92YWxpZGF0aW9uIjpmYWxzZX0%3D

```
data_augmentation = tf.keras.Sequential(
    [
     preprocessing.RandomRotation(1.0, fill_mode='reflect', interpolation='bilinear', seed=1, name=None, fill_value=0.0)
    ]
)
```

#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/PigCakee/omi_lab4/main/epoch_categorical_accuracy_3_c_factor_1.0.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/PigCakee/omi_lab4/main/epoch_loss_3_c_factor_1.0.svg">

Максимальная точность на валидационной выборке (15 эпох) = 65%

2. factor = 0.63
https://tensorboard.dev/experiment/KD50sbiGSZie2UeP0znKfw/#scalars&runSelectionState=eyJmMTAxLTE2MTk2Mjg5NzMuMTQwODMzMS90cmFpbiI6ZmFsc2UsImYxMDEtMTYxOTYyNjkwOC42MTg4NTUvdmFsaWRhdGlvbiI6ZmFsc2UsImYxMDEtMTYxOTYyNjkwOC42MTg4NTUvdHJhaW4iOmZhbHNlfQ%3D%3D

```
data_augmentation = tf.keras.Sequential(
    [
     preprocessing.RandomRotation(0.66, fill_mode='reflect', interpolation='bilinear', seed=1, name=None, fill_value=0.0)
    ]
)
```

#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/PigCakee/omi_lab4/main/epoch_categorical_accuracy_3_c_factor_0.66.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/PigCakee/omi_lab4/main/epoch_loss_3_c_factor_0.66.svg">

Максимальная точность на валидационной выборке (15 эпох) = 64%

3. factor = 0.33
https://tensorboard.dev/experiment/mtvxQKXwR3iMW3W8GMVWlw/#scalars&runSelectionState=eyJmMTAxLTE2MTk2MjkwMTAuNzM2MDMxMy92YWxpZGF0aW9uIjpmYWxzZSwiZjEwMS0xNjE5NjI5MDEwLjczNjAzMTMvdHJhaW4iOmZhbHNlLCJmMTAxLTE2MTk2Mjg5NzMuMTQwODMzMS90cmFpbiI6ZmFsc2UsImYxMDEtMTYxOTYyNjkwOC42MTg4NTUvdmFsaWRhdGlvbiI6ZmFsc2UsImYxMDEtMTYxOTYyNjkwOC42MTg4NTUvdHJhaW4iOmZhbHNlLCJmMTAxLTE2MTk2MzA1ODkuNzM3MzUwNy90cmFpbiI6dHJ1ZX0%3D

```
data_augmentation = tf.keras.Sequential(
    [
     preprocessing.RandomRotation(0.33, fill_mode='reflect', interpolation='bilinear', seed=1, name=None, fill_value=0.0)
    ]
)
```

#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/PigCakee/omi_lab4/main/epoch_categorical_accuracy_3_c_factor_0.33.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/PigCakee/omi_lab4/main/epoch_loss_3_c_factor_0.33.svg">

Максимальная точность на валидационной выборке (15 эпох) = 65,5%

4. factor = 0.1
https://tensorboard.dev/experiment/qnpUrukZR8WRx5mwsxdXJg/#scalars&runSelectionState=eyJmMTAxLTE2MTk2MzA1ODkuNzM3MzUwNy92YWxpZGF0aW9uIjpmYWxzZSwiZjEwMS0xNjE5NjMwNTg5LjczNzM1MDcvdHJhaW4iOmZhbHNlLCJmMTAxLTE2MTk2MjkwMTAuNzM2MDMxMy92YWxpZGF0aW9uIjpmYWxzZSwiZjEwMS0xNjE5NjI5MDEwLjczNjAzMTMvdHJhaW4iOmZhbHNlLCJmMTAxLTE2MTk2Mjg5NzMuMTQwODMzMS90cmFpbiI6ZmFsc2UsImYxMDEtMTYxOTYyNjkwOC42MTg4NTUvdmFsaWRhdGlvbiI6ZmFsc2UsImYxMDEtMTYxOTYyNjkwOC42MTg4NTUvdHJhaW4iOmZhbHNlLCJmMTAxLTE2MTk2MzIxNDAuNzU0MzIwOS90cmFpbiI6dHJ1ZSwiZjEwMS0xNjE5NjMyMTQwLjc1NDMyMDkvdmFsaWRhdGlvbiI6dHJ1ZX0%3D

```
data_augmentation = tf.keras.Sequential(
    [
     preprocessing.RandomRotation(0.1, fill_mode='reflect', interpolation='bilinear', seed=1, name=None, fill_value=0.0)
    ]
)
```

#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/PigCakee/omi_lab4/main/epoch_categorical_accuracy_3_c_factor_0.1.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/PigCakee/omi_lab4/main/epoch_loss_3_c_factor_0.1.svg">

Максимальная точность на валидационной выборке (15 эпох) = 66,7%

### Анализ результатов
По ходу выполнения данного эксперимента сильной разбежки максимальной точности сети не наблюдается, однако лучшие результаты получаются при использовании параметра factor = 0.1

1. fill_mode = constant
https://tensorboard.dev/experiment/mqKhVoi5Tje5E56yXxmRRg/#scalars

```
data_augmentation = tf.keras.Sequential(
    [
     preprocessing.RandomRotation(0.1, fill_mode='constant', interpolation='bilinear', seed=1, name=None, fill_value=0.0)
    ]
)
```

#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/PigCakee/omi_lab4/main/epoch_categorical_accuracy_3_c_fill_mode_constant.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/PigCakee/omi_lab4/main/epoch_loss_3_c_fill_mode_constant.svg">

Максимальная точность на валидационной выборке (15 эпох) = 66,5%

2. fill_mode = reflect
https://tensorboard.dev/experiment/qnpUrukZR8WRx5mwsxdXJg/#scalars&runSelectionState=eyJmMTAxLTE2MTk2MzA1ODkuNzM3MzUwNy92YWxpZGF0aW9uIjpmYWxzZSwiZjEwMS0xNjE5NjMwNTg5LjczNzM1MDcvdHJhaW4iOmZhbHNlLCJmMTAxLTE2MTk2MjkwMTAuNzM2MDMxMy92YWxpZGF0aW9uIjpmYWxzZSwiZjEwMS0xNjE5NjI5MDEwLjczNjAzMTMvdHJhaW4iOmZhbHNlLCJmMTAxLTE2MTk2Mjg5NzMuMTQwODMzMS90cmFpbiI6ZmFsc2UsImYxMDEtMTYxOTYyNjkwOC42MTg4NTUvdmFsaWRhdGlvbiI6ZmFsc2UsImYxMDEtMTYxOTYyNjkwOC42MTg4NTUvdHJhaW4iOmZhbHNlLCJmMTAxLTE2MTk2MzIxNDAuNzU0MzIwOS90cmFpbiI6dHJ1ZSwiZjEwMS0xNjE5NjMyMTQwLjc1NDMyMDkvdmFsaWRhdGlvbiI6dHJ1ZX0%3D

```
data_augmentation = tf.keras.Sequential(
    [
     preprocessing.RandomRotation(0.1, fill_mode='reflect', interpolation='bilinear', seed=1, name=None, fill_value=0.0)
    ]
)
```

#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/PigCakee/omi_lab4/main/epoch_categorical_accuracy_3_c_fill_mode_reflect.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/PigCakee/omi_lab4/main/epoch_loss_3_c_fill_mode_reflect.svg">

Максимальная точность на валидационной выборке (15 эпох) = 66,7%

3. fill_mode = wrap
https://tensorboard.dev/experiment/kZO0vexOQwuGi5Gca09LKQ/#scalars&runSelectionState=eyJmMTAxLTE2MTk2MzM4MDAuMzk4ODY3NC90cmFpbiI6ZmFsc2UsImYxMDEtMTYxOTYzNTM4Mi4wMjc2MTM2L3RyYWluIjp0cnVlLCJmMTAxLTE2MTk2MzM4MDAuMzk4ODY3NC92YWxpZGF0aW9uIjpmYWxzZX0%3D

```
data_augmentation = tf.keras.Sequential(
    [
     preprocessing.RandomRotation(0.1, fill_mode='wrap', interpolation='bilinear', seed=1, name=None, fill_value=0.0)
    ]
)
```

#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/PigCakee/omi_lab4/main/epoch_categorical_accuracy_3_c_fill_mode_wrap.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/PigCakee/omi_lab4/main/epoch_loss_3_c_fill_mode_wrap.svg">

Максимальная точность на валидационной выборке (15 эпох) = 66,3%

4. fill_mode = nearest
https://tensorboard.dev/experiment/vvCqHN97RXqbAVmNiZK0Og/#scalars&runSelectionState=eyJmMTAxLTE2MTk2MzM4MDAuMzk4ODY3NC90cmFpbiI6ZmFsc2UsImYxMDEtMTYxOTYzMzgwMC4zOTg4Njc0L3ZhbGlkYXRpb24iOmZhbHNlLCJmMTAxLTE2MTk2MzUzODIuMDI3NjEzNi90cmFpbiI6ZmFsc2UsImYxMDEtMTYxOTYzNTM4Mi4wMjc2MTM2L3ZhbGlkYXRpb24iOmZhbHNlLCJmMTAxLTE2MTk2MzY4NDQuMDA4MzkyOC90cmFpbiI6dHJ1ZSwiZjEwMS0xNjE5NjM2ODQ0LjAwODM5MjgvdmFsaWRhdGlvbiI6dHJ1ZX0%3D

```
data_augmentation = tf.keras.Sequential(
    [
     preprocessing.RandomRotation(0.1, fill_mode='nearest', interpolation='bilinear', seed=1, name=None, fill_value=0.0)
    ]
)
```

#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/PigCakee/omi_lab4/main/epoch_categorical_accuracy_3_c_fill_mode_nearest.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/PigCakee/omi_lab4/main/epoch_loss_3_c_fill_mode_nearest.svg">

Максимальная точность на валидационной выборке (15 эпох) = 66,9%

### Анализ результатов
Точно так же, как и в прошлом эксперименте, сильной разбежки нет, но лучшим показал себя параметр fill_mode = nearest.

1. interpolation = bilineal
https://tensorboard.dev/experiment/vvCqHN97RXqbAVmNiZK0Og/#scalars&runSelectionState=eyJmMTAxLTE2MTk2MzM4MDAuMzk4ODY3NC90cmFpbiI6ZmFsc2UsImYxMDEtMTYxOTYzMzgwMC4zOTg4Njc0L3ZhbGlkYXRpb24iOmZhbHNlLCJmMTAxLTE2MTk2MzUzODIuMDI3NjEzNi90cmFpbiI6ZmFsc2UsImYxMDEtMTYxOTYzNTM4Mi4wMjc2MTM2L3ZhbGlkYXRpb24iOmZhbHNlLCJmMTAxLTE2MTk2MzY4NDQuMDA4MzkyOC90cmFpbiI6dHJ1ZSwiZjEwMS0xNjE5NjM2ODQ0LjAwODM5MjgvdmFsaWRhdGlvbiI6dHJ1ZX0%3D

```
data_augmentation = tf.keras.Sequential(
    [
     preprocessing.RandomRotation(0.1, fill_mode='nearest', interpolation='bilinear', seed=1, name=None, fill_value=0.0)
    ]
)
```

#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/PigCakee/omi_lab4/main/epoch_categorical_accuracy_3_c_interpolation_bilineal.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/PigCakee/omi_lab4/main/epoch_loss_3_c_interpolation_bilineal.svg">

Максимальная точность на валидационной выборке (15 эпох) = 66,9%

2. interpolation = nearest
https://tensorboard.dev/experiment/6LfvHxYOQyWredF1sMtA5Q/#scalars

```
data_augmentation = tf.keras.Sequential(
    [
     preprocessing.RandomRotation(0.1, fill_mode='nearest', interpolation='nearest', seed=1, name=None, fill_value=0.0)
    ]
)
```

#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/PigCakee/omi_lab4/main/epoch_categorical_accuracy_3_c_interpolation_nearest.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/PigCakee/omi_lab4/main/epoch_loss_3_c_interpolation_nearest.svg">

Максимальная точность на валидационной выборке (15 эпох) = 65,8%

### Анализ результатов
Параметр interpolation = bilineal показал лучший результат за 15 эпох - 66.9%, в обоих случаях сеть начала переобучаться примерно после 13 эпохи. Параметр fill_value можно изменять только при использовании параметра fill_mode = constant, но учитывая, что он не показал хороших результатов, исследование параметра fill_value не несет в себе необходимости.

## 4. Обучить нейронную сеть с использованием оптимальных техник аугментации данных 2a-с совместно
https://tensorboard.dev/experiment/kNhQy1ZXR7KJ3e60koDW0g/#scalars&runSelectionState=eyJmMTAxLTE2MTk3MTY0MzkuNDYyODc2Ni90cmFpbiI6ZmFsc2UsImYxMDEtMTYxOTcxNjQzOS40NjI4NzY2L3ZhbGlkYXRpb24iOmZhbHNlLCJmMTAxLTE2MTk5NDI4NzYuODQzMDE2L3RyYWluIjpmYWxzZSwiZjEwMS0xNjE5OTQyODc2Ljg0MzAxNi92YWxpZGF0aW9uIjpmYWxzZSwiZjEwMS0xNjIwMDI3NjI4LjUwODQ5MzcvdHJhaW4iOnRydWUsImYxMDEtMTYyMDAyNzYyOC41MDg0OTM3L3ZhbGlkYXRpb24iOnRydWV9

```
data_augmentation = tf.keras.Sequential(
    [
        preprocessing.RandomCrop(224, 224),
        preprocessing.RandomRotation(0.1, fill_mode='nearest', interpolation='bilinear', seed=1, name=None, fill_value=0.0),
        preprocessing.RandomFlip("horizontal_and_vertical", seed=1)
    ]
)
```

#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/PigCakee/omi_lab4/main/epoch_categorical_accuracy_4.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/PigCakee/omi_lab4/main/epoch_loss_4.svg">

Максимальная точность на валидационной выборке (15 эпох) = 64,4%

### Анализ результатов
В ходе данной лабораторной работы путем исследований следующие параметры оказались наиболее оптимальными: 
1. Использование случайной части изображения: width = 224, height = 224 (upscale до 250)
2. Поворот на случайный угол: factor = 0.1, fill_mode='nearest', interpolation='bilinear'
Важно отметить, что несмотря на использование оптимальных параметров, применение сразу нескольких методов аугментации данных не улучшает максимальную точность сети по сравнению с использованием какого-либо одного метода.

## Дополнение: изображения, полученные при использовании различных техник аугментации данных:

