"""Run the test suite inside Roblox through Open Cloud.

Builds the test place with Rojo (tests.project.json), saves it as a new version of an
existing place, runs tests/run.luau there with the Luau Execution API, and prints the
output. Needs the environment variables ROBLOX_API_KEY (an Open Cloud key with
universe-places write and luau-execution-sessions read/write on the experience),
ROBLOX_UNIVERSE_ID and ROBLOX_PLACE_ID. Saving a version replaces what is in the place.

    python tools/roblox_test.py [--script FILE] [--timeout SECONDS]

--script runs another Luau file instead of the test suite (it can require
game.ReplicatedStorage.Citrate.src).
"""

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
API = "https://apis.roblox.com"

TEST_SCRIPT = """
local tests = game:GetService("ReplicatedStorage").Citrate.tests
-- yield before each test: a script may only run so long without yielding
require(tests.harness).hooks.yield = function() task.wait() end
local start = os.clock()
require(tests.run)
print(string.format("suite time: %.2f s", os.clock() - start))
"""


def env(name):
    value = os.environ.get(name)
    if not value and sys.platform == "win32":
        # A user variable set after this shell started is still in the registry.
        import winreg

        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                value = winreg.QueryValueEx(key, name)[0]
        except OSError:
            value = None
    if not value:
        sys.exit(f"missing environment variable {name}")
    return value


def request(method, url, key, body=None, content_type="application/json"):
    data = body
    if isinstance(body, (dict, list)):
        data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("x-api-key", key)
    if data is not None:
        req.add_header("Content-Type", content_type)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as err:
        sys.exit(f"{method} {url.split('?')[0]} failed: HTTP {err.code}: {err.read().decode(errors='replace')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", help="a Luau file to run instead of the test suite")
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    key = env("ROBLOX_API_KEY")
    universe = env("ROBLOX_UNIVERSE_ID")
    place = env("ROBLOX_PLACE_ID")

    rojo = os.environ.get("ROJO", "rojo")
    out = ROOT / "build" / "tests.rbxl"
    out.parent.mkdir(exist_ok=True)
    subprocess.run([rojo, "build", str(ROOT / "tests.project.json"), "-o", str(out)], check=True)

    saved = request(
        "POST",
        f"{API}/universes/v1/{universe}/places/{place}/versions?versionType=Saved",
        key,
        out.read_bytes(),
        "application/octet-stream",
    )
    version = saved["versionNumber"]
    print(f"saved place version {version}")

    script = pathlib.Path(args.script).read_text(encoding="utf-8") if args.script else TEST_SCRIPT
    task = request(
        "POST",
        f"{API}/cloud/v2/universes/{universe}/places/{place}/versions/{version}/luau-execution-session-tasks",
        key,
        {"script": script, "timeout": f"{args.timeout}s"},
    )
    path = task["path"]
    while task.get("state") not in ("COMPLETE", "FAILED", "CANCELLED"):
        time.sleep(3)
        task = request("GET", f"{API}/cloud/v2/{path}", key)

    token = ""
    while True:
        logs = request("GET", f"{API}/cloud/v2/{path}/logs?maxPageSize=10000&pageToken={token}", key)
        for page in logs.get("luauExecutionSessionTaskLogs", []):
            for line in page.get("messages", []):
                print(line)
        token = logs.get("nextPageToken")
        if not token:
            break

    print(f"task {task['state']}")
    if task["state"] != "COMPLETE":
        print(json.dumps(task.get("error", {}), indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
