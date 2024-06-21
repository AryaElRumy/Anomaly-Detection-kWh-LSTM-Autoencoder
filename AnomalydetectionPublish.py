import paho.mqtt.client as mqtt
import json
import pandas as pd
import numpy as np
from keras.models import load_model
import pickle
from collections import deque
import time

# Load the model from the file using pickle
with open('lstm_autoencoder_model_V1.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the scaler from the file using pickle
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Buffer to store received data
data_buffer = deque(maxlen=10)  # Buffer to store the latest 10 data points

# MQTT Configuration
local_mqtt_broker = "localhost"
local_mqtt_port = 1883
local_mqtt_topic = "power_meter/data"
anomaly_topic = "power_meter/anomaly"  # Topic for anomaly messages

def on_message(client, userdata, msg):
    """Callback function to be called when a message is received"""
    try:
        data = json.loads(msg.payload)
        data_buffer.append(data)
        if len(data_buffer) == 10:  # Process data when buffer is full
            detect_anomalies()
            data_buffer.clear()  # Clear the buffer after processing
    except Exception as e:
        print(f"Error processing message: {e}")

def detect_anomalies():
    """Detect anomalies in the received data"""
    try:
        df = pd.DataFrame(list(data_buffer))
        df['kWh'] = df['kWh'].astype(float)

        # Print the dataframe to verify the structure
        print("DataFrame head:\n", df.head(10))
        
        # Normalize the kWh values
        data_normalized = scaler.transform(df[['kWh']])
        
        # Prepare the data for the model
        X_test = np.array(data_normalized).reshape(len(data_normalized), 1, 1)

        # Print the shape of X_test to verify the input shape
        print("X_test shape:", X_test.shape)
        
        # Make predictions (reconstruction)
        values_pred_scaled = model.predict(X_test)
        
        # Inverse transform the predicted values
        values_pred = scaler.inverse_transform(values_pred_scaled.reshape(-1, values_pred_scaled.shape[-1])).reshape(values_pred_scaled.shape)
        
        # Calculate the mean squared error (MSE)
        test_reconstruction_error = np.mean(np.abs(values_pred_scaled - X_test), axis=(1, 2))
        
        # Set threshold for MSE to identify anomalies
        threshold = 0.027
        
        # Identify anomalies
        anomalies = test_reconstruction_error > threshold
        
        # Print anomalies and their timestamps
        for i, is_anomaly in enumerate(anomalies):
            if is_anomaly:
                timestamp = df['datetime'].iloc[i]
                kWh_value = df['kWh'].iloc[i]
                print(f"Anomaly detected at {timestamp}: kWh={kWh_value}")
                
                # Publish a message to the MQTT broker
                anomaly_message = {
                    'timestamp': timestamp,
                    'kWh': kWh_value
                }
                client.publish(anomaly_topic, json.dumps(anomaly_message))
    except Exception as e:
        print(f"Error detecting anomalies: {e}")

# Create an MQTT client instance
client = mqtt.Client()

# Set the callback function for receiving messages
client.on_message = on_message

# Connect to the MQTT broker and subscribe to the topic
client.connect(local_mqtt_broker, local_mqtt_port)
client.subscribe(local_mqtt_topic)

# Start the MQTT client loop
client.loop_start()

# Keep the script running
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Exiting...")
finally:
    client.loop_stop()
    client.disconnect()
