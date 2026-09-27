"""Only the dispatcher stops; its child finishes, and all exits restore it."""
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from collection_eval_lease import lease, process


@pytest.mark.parametrize("complete", [True, False])
def test_worker_continues_and_dispatcher_is_restored(tmp_path, complete):
    marker, status = tmp_path / "child_done", tmp_path / "ipc.json"
    status.write_text(json.dumps(dict(state="waiting_for_gpu")))
    child = tmp_path / "collector.py"
    child.write_text("from pathlib import Path\nimport sys,time\ntime.sleep(.3)\nPath(sys.argv[1]).write_text('done')\n")
    dispatcher = tmp_path / "collect_scaled_batched.py"
    dispatcher.write_text("import subprocess,sys,time\n"
                          "subprocess.Popen([sys.executable,sys.argv[1],sys.argv[2]])\n"
                          "while True: time.sleep(.1)\n")
    evaluator = tmp_path / "eval_static_flow.py"
    evaluator.write_text("from pathlib import Path\nimport sys,time,json\n"
                         "while not Path(sys.argv[1]).exists(): time.sleep(.01)\n"
                         "state=Path('/proc/'+sys.argv[3]+'/stat').read_text().rsplit(')',1)[1].split()[0]\n"
                         "if sys.argv[4]=='yes':\n"
                         " p=Path(sys.argv[2]); t=p.with_suffix('.tmp')\n"
                         " t.write_text(json.dumps(dict(state='complete',dispatcher_state=state))); t.replace(p)\n"
                         "else:\n"
                         " while True: time.sleep(.1)\n")
    collection = subprocess.Popen([sys.executable, str(dispatcher), str(child), str(marker)])
    evaluation = subprocess.Popen([sys.executable, str(evaluator), str(marker), str(status),
                                   str(collection.pid), "yes" if complete else "no"])
    try:
        # The dispatcher must have spawned its worker before we stop it.
        from collection_eval_lease import children
        deadline = time.monotonic() + 5
        while not children(collection.pid):
            assert time.monotonic() < deadline
            time.sleep(.01)
        if complete:
            result = lease(collection.pid, evaluation.pid, status, tmp_path / "lease.json", max_wait=3, poll=.01)
            assert result["state"] == "evaluation_complete"
            assert json.loads(status.read_text())["dispatcher_state"] == "T"
        else:
            with pytest.raises(TimeoutError):
                lease(collection.pid, evaluation.pid, status, tmp_path / "lease.json", max_wait=.5, poll=.01)
            assert evaluation.wait(timeout=3) != 0
        deadline = time.monotonic() + 2
        while process(collection.pid)[0] == "T":
            assert time.monotonic() < deadline
            time.sleep(.01)
        assert marker.exists()
        assert json.loads((tmp_path / "lease.json").read_text())["collector_resumed"]
    finally:
        for proc in (collection, evaluation):
            if proc.poll() is None:
                proc.send_signal(signal.SIGCONT)
                proc.terminate()
            proc.wait(timeout=3)
