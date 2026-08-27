"""Selecting which fused people reach the dashboard, and shipping them to MQTT/HTTP."""

import json
import paho.mqtt.client as mqtt
import time
import urllib.request

from datetime import (
    datetime,
    timezone,
)

from tactical_render import encode_image_to_base64


def create_mqtt_client(host, port, client_id="tactical-publisher", username=None, password=None):
    try:
        try:
            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION_2, client_id=client_id)
        except (AttributeError, TypeError):
            client = mqtt.Client(client_id=client_id)
        if username is not None and password is not None:
            client.username_pw_set(username, password)
        client.connect(host, port, keepalive=60)
        client.loop_start()
        print(f"MQTT connected to {host}:{port}")
        return client
    except Exception as e:
        print(f"MQTT connection failed: {e}. Continuing without MQTT.")
        return None

def post_json(url, payload, timeout=5):
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, response.read().decode("utf-8")

def dashboard_eligible_people(fused_people):
    """Return confirmed, role-classified masters for the public dashboard."""
    eligible = []
    for person in fused_people:
        if person.get("identity_id") is None or person.get("identity_state") != "confirmed":
            continue
        role = person.get("role")
        if role is None:
            observation_roles = {
                str(observation.get("role")).strip().lower()
                for observation in person.get("observations", ())
                if observation.get("role") is not None
            }
            role = observation_roles.pop() if len(observation_roles) == 1 else None
        role = str(role).strip().lower() if role is not None else None
        if role not in {"evacuee", "cag", "scdf"}:
            continue
        dashboard_person = dict(person)
        dashboard_person["role"] = role
        eligible.append(dashboard_person)
    return eligible

def dashboard_tactical_people(fused_people):
    """Return every person whose position should appear on the tactical map.

    Identity and role analysis must not erase a physical presence. Confirmed
    people retain their classified role, while every unresolved fused presence
    remains visible with an explicitly unknown role until analysis completes.
    """
    confirmed_roles = {
        person["identity_id"]: person["role"]
        for person in dashboard_eligible_people(fused_people)
    }
    tactical_people = []
    for person in fused_people:
        tactical_person = dict(person)
        identity_id = person.get("identity_id")
        role = confirmed_roles.get(identity_id)
        tactical_person["role"] = role
        if role is None and tactical_person.get("identity_state") in {None, "confirmed"}:
            # This includes early multi-crop intake and role classification:
            # both are physically present but not ready for a role colour yet.
            tactical_person["identity_state"] = "analyzing"
        tactical_people.append(tactical_person)
    return tactical_people

def build_payloads(contexts, args, fused_people, combined_map=None):
    dashboard_people = dashboard_eligible_people(fused_people)
    tactical_people = dashboard_tactical_people(fused_people)
    evacuees = [person for person in dashboard_people if person["role"] == "evacuee"]
    passenger_count = len(evacuees)
    positions = []
    zone_counts = {context.camera_id: 0 for context in contexts}

    for person in evacuees:
        for camera_id in set(person.get("sources", ())):
            if camera_id in zone_counts:
                zone_counts[camera_id] += 1

    for person in tactical_people:
        center = person.get("center")
        if center is None:
            # Still counted upstream, but the dashboard plots coordinates and
            # there are none to plot without inventing them.
            continue
        x, y = center
        stable_id = person.get("identity_id")
        temporary_group_id = person.get("temporary_group_id")
        source_track_key = "+".join(
            f"{observation.get('camera_id')}:{observation.get('local_track_id')}"
            for observation in person.get("observations", ())
        )
        if stable_id is not None:
            person_id = f"ID_{stable_id}"
        elif temporary_group_id is not None:
            person_id = f"TMP_{temporary_group_id}"
        else:
            person_id = f"TRACK_{source_track_key or len(positions)}"
        positions.append({
            "person_id": person_id,
            "master_id": stable_id,
            "role": person.get("role"),
            "identity_state": person.get("identity_state"),
            "sources": person["sources"],
            "source_tracks": [
                {
                    "camera_id": observation.get("camera_id"),
                    "local_track_id": observation.get("local_track_id"),
                }
                for observation in person.get("observations", ())
            ],
            "x": round(float(x), 1),
            "y": round(float(y), 1),
        })

    tactical_payload = {
        "timestamp": int(time.time()),
        "camera_id": args.camera_id,
        "run_id": args.run_id,
        "people_count": passenger_count,
        "positions_cm": positions,
        "map_size_cm": args.map_size_cm,
        "zone_counts": zone_counts,
        "camera_online_count": sum(1 for context in contexts if context.cap.is_opened()),
    }

    if combined_map is not None and args.mqtt_send_map_image:
        image_b64 = encode_image_to_base64(combined_map, args.mqtt_image_quality)
        if image_b64 is not None:
            tactical_payload["tactical_map_image"] = image_b64

    metric_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": args.run_id,
        "passenger_count": passenger_count,
        "zone_counts": zone_counts,
        "camera_online_count": tactical_payload["camera_online_count"],
    }

    return tactical_payload, metric_payload
