"""Test that FollowerRegistry is thread-safe under concurrent access.

This test exploits the race condition where HTTP handler threads call register()
(adding new keys) while the main loop calls has_live_followers() (iterating
and deleting keys). Without a lock, this produces:
    RuntimeError: dictionary changed size during iteration
"""

import threading
import time

import pytest

from kiro_ception.engine_main import FollowerRegistry


class TestFollowerRegistryThreadSafety:
    def test_concurrent_register_and_liveness_check(self):
        """Concurrent register() + has_live_followers() must not raise RuntimeError.

        This simulates the real scenario:
        - Multiple HTTP handler threads calling register(pid) with new PIDs
        - Main loop calling has_live_followers() which iterates + prunes dead PIDs

        Without thread safety, this reliably produces:
            RuntimeError: dictionary changed size during iteration

        We run multiple attempts because the race is non-deterministic on any
        single run — but with enough pressure it's near-certain to trigger.
        """
        for attempt in range(5):
            registry = FollowerRegistry()
            errors: list[Exception] = []
            stop_event = threading.Event()

            def register_loop():
                """Simulate HTTP handler threads registering new follower PIDs."""
                pid = 10000
                while not stop_event.is_set():
                    try:
                        registry.register(pid)
                    except Exception as e:
                        errors.append(e)
                    pid += 1
                    # Wrap around to keep adding genuinely new keys
                    if pid > 60000:
                        pid = 10000

            def liveness_check_loop():
                """Simulate main loop calling has_live_followers() repeatedly."""
                while not stop_event.is_set():
                    try:
                        registry.has_live_followers()
                    except Exception as e:
                        errors.append(e)

            # Launch many threads to maximize contention
            threads = []
            for _ in range(8):
                t = threading.Thread(target=register_loop)
                threads.append(t)
            for _ in range(4):
                t = threading.Thread(target=liveness_check_loop)
                threads.append(t)

            for t in threads:
                t.start()

            # Run under contention for 3 seconds
            time.sleep(3)
            stop_event.set()

            for t in threads:
                t.join(timeout=5)

            # If FollowerRegistry is not thread-safe, we'll see RuntimeError here
            runtime_errors = [e for e in errors if isinstance(e, RuntimeError)]
            if runtime_errors:
                pytest.fail(
                    f"Attempt {attempt + 1}: Got {len(runtime_errors)} RuntimeError(s) "
                    f"from concurrent access. First: {runtime_errors[0]}"
                )
                return

        # If we got here without any errors across all attempts, the test passes.
        # This shouldn't happen on unfixed code — the race triggers reliably.

    def test_concurrent_register_during_iteration_with_count(self):
        """The .count property must also be safe under concurrent mutation."""
        registry = FollowerRegistry()
        errors: list[Exception] = []
        stop_event = threading.Event()

        def register_loop():
            pid = 30000
            while not stop_event.is_set():
                try:
                    registry.register(pid)
                except Exception as e:
                    errors.append(e)
                pid += 1
                if pid > 40000:
                    pid = 30000

        def count_loop():
            while not stop_event.is_set():
                try:
                    _ = registry.count
                except Exception as e:
                    errors.append(e)

        threads = [
            threading.Thread(target=register_loop),
            threading.Thread(target=register_loop),
            threading.Thread(target=count_loop),
            threading.Thread(target=count_loop),
        ]

        for t in threads:
            t.start()
        time.sleep(1)
        stop_event.set()
        for t in threads:
            t.join(timeout=5)

        runtime_errors = [e for e in errors if isinstance(e, RuntimeError)]
        assert len(runtime_errors) == 0, (
            f"Got {len(runtime_errors)} RuntimeError(s). First: {runtime_errors[0]}"
        )
