"""«Chrome» means the user's own Chrome (the extension), never the separate DevTools Chrome.

Seen in the Connectors page: the gallery's separate Chrome (chrome-devtools-mcp, id `chrome`) was installed with the
early label «Chrome» and listed `chrome` among the words that switch it on — right under the «Tu Chrome (extensión)»
card, it read as the same browser twice, and «abre chrome» started a second, empty Chrome."""

import time

from nesicode.connectors import mentioned, rename_legacy_labels, trigger_words
from nesicode.server.db import DB


def test_the_word_chrome_belongs_to_the_users_own_chrome():
    words = trigger_words(name="chrome", label="Chrome", template="chrome")
    assert "chrome" not in words
    assert {"chrome devtools", "devtools", "chrome aparte"} <= set(words)
    sep = dict(name="chrome", label="Chrome aparte (DevTools)", template="chrome")
    assert not mentioned("abre chrome y entra en mi correo", **sep)
    assert mentioned("usa chrome devtools para medir la web", **sep)
    assert mentioned("hazlo en el chrome aparte", **sep)
    # the extension keeps the word
    assert "chrome" in trigger_words(name="browser", label="Tu Chrome", template="browser")
    # a second copy of the separate Chrome keeps its own distinct name
    assert "chrome 2" in trigger_words(name="chrome-2", label="Chrome aparte (DevTools)", template="chrome")


def test_old_gallery_labels_are_renamed_once_and_chosen_labels_kept(tmp_path):
    db = DB(tmp_path / "n.db")
    uid = db.insert("INSERT INTO users (email, name, password_hash, created_at) VALUES ('a@b.c', 'a', '', ?)", (time.time(),))

    def add(name, label, template):
        db.insert("INSERT INTO user_connectors (user_id, name, label, template, kind, created_at) VALUES (?, ?, ?, ?, 'stdio', ?)",
                  (uid, name, label, template, time.time()))

    add("chrome", "Chrome", "chrome")
    add("chrome-2", "Mi Chrome de pruebas", "chrome")  # chosen by the user
    add("word", "Word (.docx)", "word")
    assert rename_legacy_labels(db) == 1
    labels = {r["name"]: r["label"] for r in db.all("SELECT name, label FROM user_connectors")}
    assert labels == {"chrome": "Chrome aparte (DevTools)", "chrome-2": "Mi Chrome de pruebas", "word": "Word (.docx)"}
    assert rename_legacy_labels(db) == 0  # idempotent
