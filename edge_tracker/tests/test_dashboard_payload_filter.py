import unittest
from types import SimpleNamespace

from dashboard_payload import build_payloads


class _Capture:
    def is_opened(self):
        return True


def person(
    identity_id,
    role,
    center,
    state="confirmed",
    sources=("cam_1", "cam_2"),
    temporary_group_id=None,
):
    return {
        "center": center,
        "sources": list(sources),
        "observations": [],
        "identity_id": identity_id,
        "temporary_group_id": temporary_group_id,
        "identity_state": state,
        "role": role,
    }


class DashboardPayloadFilterTests(unittest.TestCase):
    def test_unresolved_people_are_mapped_but_only_confirmed_evacuees_are_counted(self):
        contexts = [
            SimpleNamespace(camera_id="cam_1", tactical_points=[], cap=_Capture()),
            SimpleNamespace(camera_id="cam_2", tactical_points=[], cap=_Capture()),
        ]
        args = SimpleNamespace(
            camera_id="fused",
            run_id="test",
            map_size_cm=300,
            mqtt_send_map_image=False,
            mqtt_image_quality=80,
        )
        fused_people = [
            person(1, "evacuee", (100, 100)),
            person(2, "cag", (120, 120)),
            person(3, "scdf", (350, 120), sources=("cam_2",)),
            person(None, None, (140, 140), state=None, sources=("cam_1",)),
            person(4, "evacuee", (160, 160), state="provisional"),
            person(
                None,
                None,
                (180, 180),
                state="analyzing",
                sources=("cam_1",),
                temporary_group_id="tmp_1",
            ),
        ]

        tactical, metrics = build_payloads(contexts, args, fused_people)

        self.assertEqual(tactical["people_count"], 1)
        self.assertEqual(metrics["passenger_count"], 1)
        self.assertEqual(len(tactical["positions_cm"]), 6)
        self.assertEqual(
            {position["role"] for position in tactical["positions_cm"]},
            {"evacuee", "cag", "scdf", None},
        )
        analyzing = next(
            position
            for position in tactical["positions_cm"]
            if position["person_id"] == "TMP_tmp_1"
        )
        self.assertEqual(analyzing["identity_state"], "analyzing")
        self.assertEqual(analyzing["person_id"], "TMP_tmp_1")
        self.assertIsNone(analyzing["master_id"])
        self.assertIsNone(analyzing["role"])
        unresolved = [
            position
            for position in tactical["positions_cm"]
            if position["role"] is None
        ]
        self.assertEqual(len(unresolved), 3)
        self.assertEqual(
            {position["identity_state"] for position in unresolved},
            {"analyzing", "provisional"},
        )
        self.assertNotIn("temporary_group_id", tactical["positions_cm"][0])
        self.assertEqual(tactical["zone_counts"], {"cam_1": 1, "cam_2": 1})


if __name__ == "__main__":
    unittest.main()
