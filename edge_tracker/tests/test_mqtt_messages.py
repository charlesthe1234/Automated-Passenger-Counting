#!/usr/bin/env python3
"""
Simple MQTT subscriber to diagnose what messages are being published.
Run this in a separate terminal while the detection script is running.
"""

import json
import paho.mqtt.client as mqtt
import sys


def on_connect(client, userdata, connect_flags, reason_code, properties=None):
    print(f"✓ Connected to MQTT broker (code: {reason_code})")
    # Subscribe to both tactical and metrics topics
    client.subscribe("cag/tactical", qos=1)
    client.subscribe("cag/metrics", qos=1)
    print("  Subscribed to: cag/tactical, cag/metrics")


def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        print(f"\n📩 Message on {msg.topic}:")
        print(f"   camera_id: {payload.get('camera_id', 'N/A')}")
        print(f"   people_count: {payload.get('people_count', 'N/A')}")
        print(f"   run_id: {payload.get('run_id', 'N/A')}")
        if "positions_cm" in payload:
            print(f"   positions_cm: {len(payload['positions_cm'])} people detected")
        if "passenger_count" in payload:
            print(f"   passenger_count: {payload['passenger_count']}")
    except json.JSONDecodeError:
        print(f"\n❌ Could not decode message on {msg.topic}: {msg.payload}")
    except Exception as e:
        print(f"\n❌ Error processing message: {e}")


def on_disconnect(client, userdata, disconnect_flags, reason_code, properties=None):
    print(f"\n✗ Disconnected from MQTT broker (code: {reason_code})")


if __name__ == "__main__":
    broker_host = sys.argv[1] if len(sys.argv) > 1 else "localhost"
    broker_port = int(sys.argv[2]) if len(sys.argv) > 2 else 1883

    print(f"🔍 Connecting to MQTT broker at {broker_host}:{broker_port}...")
    print("   Listening for messages on: cag/tactical, cag/metrics")
    print("   Press Ctrl+C to stop.\n")

    try:
        try:
            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION_2, client_id="mqtt-test-subscriber")
        except (AttributeError, TypeError):
            client = mqtt.Client(client_id="mqtt-test-subscriber")

        client.on_connect = on_connect
        client.on_message = on_message
        client.on_disconnect = on_disconnect

        client.connect(broker_host, broker_port, keepalive=60)
        client.loop_forever()
    except KeyboardInterrupt:
        print("\n\nStopped.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
