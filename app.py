import os
import tempfile
import logging
import re
import collections
import json
import secrets
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    session,
    redirect,
    url_for,
)
from dotenv import load_dotenv
from pathlib import Path
from faster_whisper import WhisperModel
import openai
import random
from epitran.flite import FliteT2P

load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

# Load the faster-whisper model once at startup
whisper_model = WhisperModel("base.en", device="cpu", compute_type="int8")

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret")

# Configure logging to file for debugging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)

# Persistent storage for struggled words per user
USER_COUNTS_FILE = "user_counts.json"
if os.path.exists(USER_COUNTS_FILE):
    with open(USER_COUNTS_FILE, "r", encoding="utf-8") as f:
        USER_COUNTS = json.load(f)
else:
    USER_COUNTS = {}


def save_user_counts() -> None:
    with open(USER_COUNTS_FILE, "w", encoding="utf-8") as f:
        json.dump(USER_COUNTS, f)


def get_user_id() -> str:
    if "uid" not in session:
        session["uid"] = secrets.token_hex(16)
    return session["uid"]


class SentenceGenerator:
    """Generate GPT sentences in batches."""

    def __init__(self, batch_size: int = 10, history_limit: int = 50) -> None:
        self.batch_size = batch_size
        self.history: collections.deque[str] = collections.deque(maxlen=history_limit)

    def fetch_batch(self, top_words: list[str] | None = None) -> list[str]:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set; using fallback sentences")
            return []

        client = openai.OpenAI(api_key=api_key)
        prompt = (
            f"Provide {self.batch_size} short English sentences for pronunciation practice. "
            "Avoid rhymes or nonsense. Do not number them or add introductions. "
            "Each sentence must end with a period. Return only the sentences separated by new lines. "
            "The sentences shouldn't be like rhymes, they should be like from real world dialogues. "
        )
        if top_words:
            prompt += (
                "Across the set, try to include the following words at least once: "
                + ", ".join(top_words)
                + "."
            )

        attempts = 0
        lines: list[str] = []
        while attempts < 3 and len(lines) < self.batch_size:
            attempts += 1
            try:
                full_prompt = prompt + f" Seed: {random.random()}"
                resp = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "system", "content": full_prompt}],
                    max_tokens=self.batch_size * 25,
                    temperature=0.9,
                    presence_penalty=1.0,
                    frequency_penalty=0.5,
                )
                text = resp.choices[0].message.content
                new_lines = [
                    re.sub(r"^\s*\d+[.)-]?\s*", "", line).strip("- \t")
                    for line in text.splitlines()
                    if line.strip() and re.search(r"[.!?]\s*$", line.strip())
                ]
                unique = [
                    l for l in new_lines if l not in self.history and l not in lines
                ]
                lines.extend(unique)
            except Exception:
                logger.exception("Failed to generate sentences with OpenAI")
                return []

        if len(lines) > self.batch_size:
            lines = lines[: self.batch_size]
        if len(lines) < self.batch_size:
            logger.warning(
                "Expected %d sentences but received %d from OpenAI",
                self.batch_size,
                len(lines),
            )
        self.history.extend(lines)
        logger.info("Generated %d sentences with GPT", len(lines))
        return lines


sentence_generator = SentenceGenerator()


# Initialize Epitran G2P for English
g2p = FliteT2P()

# Confidence threshold below which a word is considered "struggled"
CONF_THRESHOLD = 0.6


def transcribe_audio(audio_path: str):
    """Transcribe the full audio file and return words with probabilities."""
    logger.info("Transcribing with faster-whisper")
    segments, _ = whisper_model.transcribe(
        audio_path,
        word_timestamps=True,
        beam_size=5,
    )

    words = []
    for segment in segments:
        for word in segment.words:
            prob = getattr(word, "probability", None)
            clean = re.sub(r"^[^\w']+|[^\w']+$", "", word.word).lower()
            words.append({"word": word.word, "clean": clean, "prob": prob})

    text = " ".join(w["word"] for w in words)
    return text, words


def phonemize(sentence: str) -> list[str]:
    """Return IPA transcription for each word in the sentence."""
    tokens = re.findall(r"\b[\w']+\b", sentence)
    return [g2p.transliterate(t.lower()) for t in tokens]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/profile")
def profile():
    uid = get_user_id()
    counts = session.get("struggle_counts")
    if counts is None:
        counts = USER_COUNTS.get(uid, {})
        session["struggle_counts"] = counts
    words = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]
    return render_template("profile.html", words=words)


@app.route("/transcribe", methods=["POST"])
def transcribe():
    uid = get_user_id()
    file = request.files["audio"]
    sentence = request.form.get("sentence", "")
    logger.info("Received transcription request")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        file.save(tmp.name)
        text, words = transcribe_audio(tmp.name)
        os.remove(tmp.name)

    # Track struggled words based on expected sentence
    expected = re.findall(r"\b[\w']+\b", sentence.lower())
    counts = session.get("struggle_counts")
    if counts is None:
        counts = USER_COUNTS.get(uid, {})
    for i, orig in enumerate(expected):
        if i >= len(words):
            counts[orig] = counts.get(orig, 0) + 1
            continue
        w = words[i]
        clean = w.get("clean", "")
        prob = w.get("prob")
        if clean != orig or prob is None or prob < CONF_THRESHOLD:
            counts[orig] = counts.get(orig, 0) + 1
        else:
            counts[orig] = max(0, counts.get(orig, 0) - 1)
    session["struggle_counts"] = counts
    USER_COUNTS[uid] = counts
    save_user_counts()

    logger.info("Transcription completed")

    return jsonify({"text": text, "words": words})


@app.route("/random-sentence")
def random_sentence():
    uid = get_user_id()
    queue = session.get("sentence_queue", [])
    if not queue:
        counts = session.get("struggle_counts")
        if counts is None:
            counts = USER_COUNTS.get(uid, {})
            session["struggle_counts"] = counts
        top_words = [
            w for w, _ in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]
        ]
        queue = sentence_generator.fetch_batch(top_words)
        if not queue:
            with open("static/sentences.txt", "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            queue = random.sample(lines, min(len(lines), sentence_generator.batch_size))
        session["sentence_queue"] = queue
    sentence = queue.pop(0)
    session["sentence_queue"] = queue
    tokens = re.findall(r"\b[\w']+\b", sentence)
    ipa = phonemize(sentence)
    return jsonify({"sentence": sentence, "words": tokens, "ipa": ipa})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
