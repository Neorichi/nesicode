"""Speech to text for the microphone button.

Engines, in order of preference (see server/app.py transcribe_audio):
  1. The project's "Transcripción" model (any OpenAI-compatible /audio/transcriptions: Whisper,
     gpt-4o-transcribe, Groq whisper-large-v3...).
  2. macOS: Apple's own speech recognizer (Speech framework), on-device when the language supports it.
     Nothing to install and no key. Only inside the Nesicode.app bundle: macOS requires the app's
     Info.plist usage descriptions, and a bare python process would be killed by the privacy system.
"""

from __future__ import annotations

import os
import sys
import threading

LOCALES = {"es": "es-ES", "en": "en-US"}


def native_available() -> bool:
    if sys.platform != "darwin" or not (getattr(sys, "frozen", False) or os.environ.get("NESICODE_NATIVE_SPEECH")):
        return False
    try:
        import Speech  # noqa: F401  (pyobjc-framework-Speech, bundled with the app)
    except Exception:
        return False
    return True


class SpeechError(RuntimeError):
    pass


def _authorize(timeout: float = 120) -> None:
    import Speech

    status = Speech.SFSpeechRecognizer.authorizationStatus()
    if status == Speech.SFSpeechRecognizerAuthorizationStatusAuthorized:
        return
    if status in (Speech.SFSpeechRecognizerAuthorizationStatusDenied, Speech.SFSpeechRecognizerAuthorizationStatusRestricted):
        raise SpeechError("macOS no deja a Nesicode usar el reconocimiento de voz: actívalo en Ajustes del Sistema → "
                          "Privacidad y seguridad → Reconocimiento de voz")
    done, box = threading.Event(), {}

    def cb(new_status):
        box["s"] = new_status
        done.set()

    Speech.SFSpeechRecognizer.requestAuthorization_(cb)  # macOS asks the user once
    done.wait(timeout)
    if box.get("s") != Speech.SFSpeechRecognizerAuthorizationStatusAuthorized:
        raise SpeechError("Sin permiso de reconocimiento de voz (Ajustes del Sistema → Privacidad y seguridad → "
                          "Reconocimiento de voz)")


def transcribe_native(path: str, lang: str = "es", timeout: float = 180) -> str:
    """Transcribe an audio file (WAV/M4A) with Apple's recognizer. Returns '' when no speech was heard."""
    import Speech
    from Foundation import NSLocale, NSOperationQueue, NSURL

    _authorize()
    locale = LOCALES.get(lang, lang if "-" in lang else "es-ES")
    rec = Speech.SFSpeechRecognizer.alloc().initWithLocale_(NSLocale.localeWithLocaleIdentifier_(locale))
    if rec is None:
        raise SpeechError(f"macOS no tiene reconocimiento de voz para {locale}")
    if not rec.isAvailable():
        raise SpeechError("El reconocimiento de voz de macOS no está disponible ahora (¿sin conexión y sin el idioma descargado?)")
    rec.setQueue_(NSOperationQueue.alloc().init())  # callbacks on a background queue: no main run loop needed
    req = Speech.SFSpeechURLRecognitionRequest.alloc().initWithURL_(NSURL.fileURLWithPath_(path))
    req.setShouldReportPartialResults_(False)
    if rec.supportsOnDeviceRecognition():
        req.setRequiresOnDeviceRecognition_(True)  # private: the audio never leaves the Mac
    if hasattr(req, "setAddsPunctuation_"):
        req.setAddsPunctuation_(True)
    done, box = threading.Event(), {}

    def handler(result, error):
        if error is not None:
            box["error"] = error
            done.set()
        elif result is not None and result.isFinal():
            box["text"] = str(result.bestTranscription().formattedString())
            done.set()

    task = rec.recognitionTaskWithRequest_resultHandler_(req, handler)
    if not done.wait(timeout):
        task.cancel()
        raise SpeechError("La transcripción tardó demasiado")
    if "error" in box and "text" not in box:
        err = box["error"]
        code = err.code() if hasattr(err, "code") else 0
        if code in (1110, 203):  # no speech detected
            return ""
        raise SpeechError(f"Reconocimiento de voz de macOS: {err.localizedDescription()}")
    return box.get("text", "")
