"""Run the test suite in an open Roblox Studio session, through Studio's MCP server.

Studio must have a place open whose ReplicatedStorage.Citrate holds src and tests (as
tests.project.json lays them out; tools/roblox_test.py saves such a version of a place)
and "Enable Studio as MCP server" turned on in the Assistant settings. The script starts a
playtest, requires each test module in the chosen data model (one call each, since code
run through MCP blocks Studio until it returns), prints the failures and stops the
playtest.

    python tools/studio_test.py [Client | Server | Edit]
"""
import json
import os
import pathlib
import queue
import re
import subprocess
import sys
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
BAT = os.path.expandvars(r"%LOCALAPPDATA%\Roblox\mcp.bat")


class Client:
    def __init__(self):
        self.p = subprocess.Popen(["cmd.exe", "/c", BAT], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1)
        self.q = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()
        self.next_id = 1

    def _read(self):
        for line in self.p.stdout:
            line = line.strip()
            if line:
                try:
                    self.q.put(json.loads(line))
                except json.JSONDecodeError:
                    pass

    def send(self, method, params=None, notify=False):
        msg = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        if not notify:
            msg["id"] = self.next_id
            self.next_id += 1
        self.p.stdin.write(json.dumps(msg) + "\n")
        self.p.stdin.flush()
        if notify:
            return None
        while True:
            reply = self.q.get(timeout=600)
            if reply.get("id") == msg["id"]:
                if "error" in reply:
                    raise RuntimeError(json.dumps(reply["error"]))
                return reply["result"]

    def start(self):
        self.send("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                 "clientInfo": {"name": "citrate-tests", "version": "1"}})
        self.send("notifications/initialized", notify=True)



def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "Client"
    run = (ROOT / "tests" / "run.luau").read_text(encoding="utf-8")
    modules = re.findall(r'require\("\./(\w+)"\)', run)
    modules = [m for m in modules if m != "harness"]

    c = Client()
    c.start()


    def call(name, **args):
        for _ in range(5):
            res = c.send("tools/call", {"name": name, "arguments": args})
            text = "".join(p.get("text", "") for p in res.get("content", []))
            if "No Roblox Studio instances are connected" in text:
                time.sleep(2)
                continue
            return text, res.get("isError", False)
        return text, True


    sid = None
    for _ in range(15):
        studios = json.loads(call("list_roblox_studios")[0])["studios"]
        if studios:
            sid = studios[0]["id"]
            break
        time.sleep(2)
    assert sid, "no Studio connected"
    if target != "Edit":
        print(call("start_stop_play", studio_id=sid, is_start=True)[0])
        for _ in range(30):
            state = call("get_studio_state", studio_id=sid)[0]
            if target in state:
                break
            time.sleep(2)
        print(state[:300])

    pre = "local tests = game:GetService('ReplicatedStorage'):WaitForChild('Citrate'):WaitForChild('tests')\n"
    total = 0.0
    for m in modules:
        code = pre + f"local t = os.clock()\nrequire(tests['{m}'])\nreturn string.format('%.2f', os.clock() - t)"
        out, err = call("execute_luau", studio_id=sid, datamodel_type=target, code=code)
        print(f"{m}: {out.strip()[:300]}{' ERROR' if err else ''}")
        try:
            total += float(out.strip().split()[-1])
        except ValueError:
            pass
    code = pre + "local h = require(tests.harness)\nlocal ok, err = pcall(h.finish)\nreturn if ok then 'all passed' else tostring(err)"
    print("finish:", call("execute_luau", studio_id=sid, datamodel_type=target, code=code)[0])
    print(f"suite time {total:.2f} s")
    console = call("get_console_output", studio_id=sid)[0]
    fails = [l for l in console.splitlines() if "FAIL" in l or "checks passed" in l]
    print("\n".join(fails[-60:]))
    if target != "Edit":
        print(call("start_stop_play", studio_id=sid, is_start=False)[0])
    c.p.terminate()


if __name__ == "__main__":
    main()
