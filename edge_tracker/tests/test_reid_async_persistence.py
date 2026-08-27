import copy
import threading
import time
import unittest

import numpy as np

from reid_memory import AppearanceIdentityMemory


class BlockingPersistenceStore:
    def __init__(self):
        self.calls = []
        self.started = threading.Event()
        self.release = threading.Event()
        self._lock = threading.Lock()

    @staticmethod
    def load_payload():
        return {
            "schema_version": AppearanceIdentityMemory.SCHEMA_VERSION,
            "identities": {},
        }

    def save_identity(self, identity_id, record):
        with self._lock:
            self.calls.append((int(identity_id), copy.deepcopy(record)))
            call_number = len(self.calls)
        if call_number == 1:
            self.started.set()
            self.release.wait(timeout=2.0)


class IncompletePersistenceStore:
    @staticmethod
    def load_payload():
        return {
            "schema_version": AppearanceIdentityMemory.SCHEMA_VERSION,
            "identities": {
                2: {
                    "gallery": AppearanceIdentityMemory._empty_gallery(),
                },
            },
        }

    @staticmethod
    def save_identity(_identity_id, _record):
        return None


class AsyncPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.store = BlockingPersistenceStore()
        self.memory = AppearanceIdentityMemory(
            persistence_store=self.store,
            enable_role_classification=False,
            enable_demographics=False,
            start_worker=False,
        )
        record = self.memory._new_record()
        record["gallery"]["baseline"] = {
            "feature": np.asarray([1.0, 0.0], dtype=np.float32),
        }
        self.memory.identities[1] = record

    def tearDown(self):
        self.store.release.set()
        self.memory.close(drain=True)

    def test_fastapi_save_is_non_blocking_and_coalesces_latest_snapshot(self):
        self.memory.identities[1]["hits"] = 1
        started_at = time.monotonic()
        self.memory.save_database(1)
        elapsed = time.monotonic() - started_at

        self.assertLess(elapsed, 0.25)
        self.assertTrue(self.store.started.wait(timeout=1.0))

        self.memory.identities[1]["hits"] = 2
        self.memory.save_database(1)
        self.memory.identities[1]["hits"] = 3
        self.memory.save_database(1)

        self.store.release.set()
        self.assertTrue(self.memory.wait_for_idle(timeout=2.0))
        self.assertEqual(len(self.store.calls), 2)
        self.assertEqual(self.store.calls[0][1]["hits"], 1)
        self.assertEqual(self.store.calls[1][1]["hits"], 3)

    def test_incomplete_backend_identity_does_not_block_startup(self):
        memory = AppearanceIdentityMemory(
            persistence_store=IncompletePersistenceStore(),
            enable_role_classification=False,
            enable_demographics=False,
            start_worker=False,
        )
        try:
            self.assertEqual(memory.identities, {})
            self.assertEqual(memory.next_identity_id, 3)
        finally:
            memory.close(drain=True)


if __name__ == "__main__":
    unittest.main()
