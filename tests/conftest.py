import os
import tempfile

os.environ["NESICODE_NO_DOTENV"] = "1"  # never read the developer's real .env (keys, models)
for _k in [k for k in os.environ if k.endswith("_API_KEY")]:
    del os.environ[_k]

# Must run before any nesicode import: settings and the DB are created at import time.
os.environ["NESICODE_DATA_DIR"] = tempfile.mkdtemp(prefix="nesicode-test-")
os.environ["NESICODE_SANDBOX"] = "local"
os.environ["NESICODE_AUTH"] = "true"  # the API tests exercise multi-user mode
os.environ["NESICODE_EXECUTOR_MODEL"] = "mock/demo"
os.environ["NESICODE_IMPROVER_MODEL"] = "mock/demo"
# Tests never touch the real ~/.nesicode/vault.key
import base64 as _b64

os.environ["NESICODE_VAULT_KEY"] = _b64.b64encode(b"t" * 32).decode()

os.environ["NESICODE_NO_TELEGRAM"] = "1"  # tests drive the bot with a fake Telegram
