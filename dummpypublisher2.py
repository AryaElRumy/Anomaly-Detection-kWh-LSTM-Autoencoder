import paho.mqtt.client as mqtt
import json
import time
import pandas as pd

# Local MQTT Configuration
local_mqtt_broker = "localhost"
local_mqtt_port = 1883
local_mqtt_topic = "power_meter/data"

# Read the CSV file
df = pd.read_csv('datapublish.csv')
df = df[['datetime', 'kWh']]
# Ensure the datetime column is in datetime format
df['datetime'] = pd.to_datetime(df['datetime'])

# Convert the DataFrame to a list of dictionaries
data_list = df.to_dict('records')

# Callback function to be called when the message is published
def on_publish(client, userdata, mid):
    print("Message published")

# Create an MQTT client instance
client = mqtt.Client()

# Set the callback function for publishing
client.on_publish = on_publish

# Connect to the MQTT broker
client.connect(local_mqtt_broker, local_mqtt_port)

try:
    while True:
        for data in data_list:
            # Convert each row to JSON string
            payload = json.dumps(data, default=str)
            
            # Publish the message
            client.publish(local_mqtt_topic, payload)
            print(f"Published message: {payload}")
            
            # Wait for 1 second before publishing the next message
            time.sleep(1)

except KeyboardInterrupt:
    print("Exiting...")
finally:
    # Disconnect from the MQTT broker
    client.disconnect()