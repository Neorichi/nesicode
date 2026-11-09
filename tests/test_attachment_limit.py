"""Files dropped into the chat may be up to 50 MB (an APK of 42 MB was refused at 25 MB); bigger ones are refused with
the limit in the message. The project upload keeps its own 200 MB."""

from nesicode.server import app as appmod

from test_api import alice, project  # noqa: F401  (fixtures)


def test_a_42_mb_apk_can_be_attached_and_51_mb_cannot(alice, project, monkeypatch):
    assert appmod.MAX_ATTACHMENT == 50 * 1024 * 1024
    t = alice.post(f"/api/projects/{project['id']}/threads", json={}).json()
    apk = b"PK\x03\x04" + b"\0" * (42 * 1024 * 1024)
    up = alice.post(f"/api/threads/{t['id']}/attachments", files=[("files", ("claude.apks", apk, "application/zip"))])
    assert up.status_code == 200, up.text
    assert up.json()[0]["path"] == ".attachments/claude.apks"
    monkeypatch.setattr(appmod, "MAX_ATTACHMENT", 1024)  # (the refusal, without a 51-MB upload)
    big = alice.post(f"/api/threads/{t['id']}/attachments", files=[("files", ("big.bin", b"x" * 2048, "application/octet-stream"))])
    assert big.status_code == 413 and big.json()["detail"] == "big.bin: máximo 0 MB por archivo"
