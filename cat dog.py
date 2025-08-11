from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os

DATASET_PATH = 'dataset/'
num_classes = len(os.listdir(DATASET_PATH))
class_mode = 'binary' if num_classes == 2 else "categorical"

# Улучшенная аугментация данных для увеличения разнообразия
train_datagen = ImageDataGenerator(
    rescale=1 / 255,
    validation_split=0.2,
    rotation_range=30,  # Поворот изображений
    width_shift_range=0.2,  # Горизонтальный сдвиг
    height_shift_range=0.2,  # Вертикальный сдвиг
    shear_range=0.2,  # Скос
    zoom_range=0.2,  # Масштабирование
    horizontal_flip=True,  # Горизонтальное отражение
    fill_mode='nearest'  # Заполнение пустых пикселей
)

# Для валидации используем только нормализацию
val_datagen = ImageDataGenerator(rescale=1 / 255, validation_split=0.2)

train_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(128, 128),
    batch_size=32,
    class_mode=class_mode,
    subset="training"
)

val_data = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(128, 128),
    batch_size=32,
    class_mode=class_mode,
    subset="validation"
)

# Улучшенная архитектура модели
model = Sequential([
    Input(shape=(128, 128, 3)),

    # Первый блок сверточных слоев
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    # Второй блок сверточных слоев
    Conv2D(64, (3, 3), activation="relu", padding='same'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation="relu", padding='same'),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    # Третий блок сверточных слоев
    Conv2D(128, (3, 3), activation="relu", padding='same'),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation="relu", padding='same'),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    # Полносвязные слои
    Flatten(),
    Dense(512, activation="relu"),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid") if class_mode == "binary" else Dense(num_classes, activation="softmax")
])

# Callback'ы для улучшения обучения
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=0.0001,
    verbose=1
)

loss_function = "binary_crossentropy" if class_mode == "binary" else "categorical_crossentropy"
model.compile(optimizer="adam", loss=loss_function, metrics=["accuracy"])

# Обучение с callback'ами и меньшим количеством эпох
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=100,  # Уменьшили количество эпох
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

test_loss, test_accuracy = model.evaluate(val_data)
print(f"Точность модели на валидационных данных: {test_accuracy:.2f}")
model.save("image_classifier.h5")