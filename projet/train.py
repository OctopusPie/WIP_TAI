import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np

from kafka import KafkaConsumer

from tensorflow import keras
from keras import layers
from keras import Sequential
from keras import layers

def plot_loss(history):
   plt.plot(history.history['loss'], label='loss')
   plt.plot(history.history['val_loss'], label='val_loss')
   plt.ylim([0, 10])
   plt.xlabel('Epoch')
   plt.ylabel('Error [MPG]')
   plt.legend()
   plt.grid(True)

if __name__ == "__main__":
    """
    Etape 1 : Constituer le Dataset (récupérer le dataset depuis le cluster Hadoop)
    Etape 2 : Diviser le dataset à 90 % en entrainement, 10% en validation
    Etape 3 : Construire le modèle de régression modèle
    Etape 4 : Ecrire la fonction pour lancer l'entrainement
    Etape 6 : Sauvegarder le modèle.
    """

    # Initialiser le consommateur Kafka
    consumer = KafkaConsumer('capteur',
                            bootstrap_servers=['127.0.0.1:9092'],
                            value_deserializer=lambda m: json.loads(m.decode('utf-8')))

    data_list = []
    # Boucle infinie pour lire les données Kafka
    for message in consumer:
        data = message.value
        data_list.append(data)
    
    train_ratio = 0.9
    train_size = int(len(data_list) * train_ratio)

    train_dataset = data_list[:train_size]
    test_dataset = data_list[train_size:]

    
    regression_model = tf.keras.Sequential([
        layers.Dense(units=1)
    ])

    regression_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
        loss='mean_absolute_error')

    input_features = train_dataset[['sensor_id', 'temperature', 'humidity', 'timestamp']].values
    target_label = train_dataset[['timestamp']]

    train_features = np.array(input_features)
    train_labels = np.array(target_label)

    history = regression_model.fit(
        train_features[''],
        train_labels,
        epochs=50,
        # Suppress logging.
        verbose=0,
        # Calculate validation results on 20% of the training data.
        validation_split = 0.1)
    
    regression_model.save('saved_model/my_model')

    # Define your model architecture
    model = Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_features,)))
    model.add(layers.Dense(1))  # Assuming a single output for regression

    # Compile the model with an appropriate loss function and optimizer
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Train the model
    history = model.fit(train_features, train_labels, epochs=50, validation_split=0.1, verbose=1)

    #save the model
    model.save('saved_model/training')

