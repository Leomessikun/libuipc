import socket
import struct
import threading
import unittest

from policy_service import MAX_FRAME, frame, read_frame


class PolicyTransportTest(unittest.TestCase):
    def test_fragmented_message_is_reassembled(self):
        payload = bytes(range(256)) * 30
        receiver, sender = socket.socketpair()

        def send():
            with sender:
                message = frame(payload)
                for i in range(0, len(message), 3):
                    sender.sendall(message[i:i + 3])

        thread = threading.Thread(target=send)
        thread.start()
        with receiver:
            receiver.settimeout(2)
            self.assertEqual(read_frame(receiver), payload)
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())

    def test_incomplete_message_fails_instead_of_returning_partial_arrays(self):
        receiver, sender = socket.socketpair()
        with sender:
            sender.sendall(struct.pack('<Q', 24) + b'only part')
        with receiver:
            with self.assertRaises(EOFError):
                read_frame(receiver)

    def test_oversized_frame_is_rejected_before_reading_payload(self):
        receiver, sender = socket.socketpair()
        with sender:
            sender.sendall(struct.pack('<Q', MAX_FRAME + 1))
        with receiver:
            with self.assertRaises(ValueError):
                read_frame(receiver)


if __name__ == '__main__':
    unittest.main()
