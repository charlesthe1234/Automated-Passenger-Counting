import unittest

from models import TacticalStateCreate
from tactical_state import TacticalStateStore


class TacticalStateRoleTests(unittest.TestCase):
    def test_staff_remain_visible_but_do_not_inflate_evacuee_counts(self):
        store = TacticalStateStore()
        payload = TacticalStateCreate(
            camera_id="fused",
            people_count=1,
            map_size_cm=300,
            positions_cm=[
                {
                    "x": 100,
                    "y": 100,
                    "person_id": "ID_1",
                    "master_id": 1,
                    "role": "evacuee",
                    "identity_state": "confirmed",
                },
                {
                    "x": 120,
                    "y": 120,
                    "person_id": "ID_2",
                    "master_id": 2,
                    "role": "cag",
                    "identity_state": "confirmed",
                },
                {
                    "x": 350,
                    "y": 100,
                    "person_id": "ID_3",
                    "master_id": 3,
                    "role": "scdf",
                    "identity_state": "confirmed",
                },
                {
                    "x": 140,
                    "y": 140,
                    "person_id": "TMP_tmp_1",
                    "role": None,
                    "identity_state": "analyzing",
                },
                {
                    "x": 160,
                    "y": 160,
                    "person_id": "LEGACY_PENDING",
                    "role": "evacuee",
                },
            ],
        )

        state = store.update(payload)

        self.assertEqual(len(state.positions_cm), 5)
        self.assertEqual(state.inside_count, 1)
        self.assertEqual(state.outside_visible_count, 0)
        self.assertEqual(state.people_count, 1)
        self.assertEqual(state.total_visible_count, 5)
        self.assertEqual(state.analyzing_count, 2)
        self.assertEqual(
            {position.role for position in state.positions_cm},
            {"evacuee", "cag", "scdf", None},
        )
        analyzing = next(
            position
            for position in state.positions_cm
            if position.identity_state == "analyzing"
        )
        self.assertEqual(analyzing.person_id, "TMP_tmp_1")
        legacy_pending = next(
            position
            for position in state.positions_cm
            if position.person_id == "LEGACY_PENDING"
        )
        self.assertIsNone(legacy_pending.identity_state)


if __name__ == "__main__":
    unittest.main()
