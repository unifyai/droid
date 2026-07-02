"""Fast-brain reply selection.

The fast brain (voice agent) does not compose the real answer — the slow brain
does, moments later. On each user turn the fast brain gives ONE brief, natural
acknowledgement to cover that gap — not a substantive reply, just a quick sign
the caller was heard.

It is no longer a fixed phrase bank: it is prompted with patterns/templates that
it adapts and fills in (e.g. "Yes, I'll let you know once {X}"), held to a hard
rule that it never actually answers or gives real information. A length backstop
and a safe default keep a mis-fire harmless. The deterministic "still waiting"
path is folded in here too (via ``already_deferred``) so repeated deferrals stay
context-aware — e.g. answering "take your time" with "Thanks", not a lookup.
"""

from __future__ import annotations

import random

from unify.common.llm_client import new_llm_client
from unify.logger import LOGGER
from unify.settings import SETTINGS

# Returned on empty input or any failure - safe after almost any utterance.
_DEFAULT_PHRASE = "One moment."

# Hard ceiling on a fast reply. A reaction / ack / brief defer is short; anything
# longer means the model overstepped into a substantive answer (the slow brain's
# job), so we drop it for the safe default.
_MAX_FAST_REPLY_CHARS = 160

_FAST_REPLY_PROMPT = """\
You give a brief, natural acknowledgement on a live voice call. The real
answer is being composed and will be spoken right after you — your only job is to
acknowledge what the caller just said in that gap, briefly and naturally.

STAY ONE PERSON — to the caller you are a single person. Never disown your own
words or split yourself into separate agents: never say a word "wasn't you",
never mention any other system, model, agent, or "smarter" component, never call
yourself "filler", and never say an answer is "still being composed". If the
caller questions what you said, just own it naturally as yourself.

HARD RULE — you NEVER actually answer or give real information: no facts, data,
names, numbers, results, instructions, or next steps. That is handled moments
from now. You only acknowledge what they said, agree, or briefly defer (say
you're getting something). If a real reply would need anything you were not just
told, you do NOT attempt it — you briefly defer instead.

MUST CONTEXTUALIZE — your reply ALWAYS refers back to what the caller just said:
their topic, the action they took, or what they asked for. NEVER reply with a
bare, standalone phrase like "Got it.", "One moment.", "Nice.", "Perfect.", or
"Will do." on its own — those sound robotic and ambiguous (a bare "Got it" sounds
like you received a thing). Always attach the specific thing, e.g. "Got it —
looking into your email now." instead of "Got it."

Acknowledge what they just said — name the topic or action, not your availability.
Adapt these patterns to the moment, fill in the specifics in {braces}, keep it to
a few words, and never recite them verbatim:
- They tell you something or confirm an action → a quick contextual ack: "Got it — I'll check on that now." / "Nice, that's the {thing} sorted."
- They give you space ("take your time", "no rush", "whenever you're ready") → thank them and take the pause, anchored to it: "Thanks — I'll keep at it." / "No problem, I'll stay on it." NEVER say you're checking or looking anything up here.
- They ask you to do or relay something ("let me know when it's done") → confirm you will, naming it: "Will do — I'll let you know once {the thing} is ready."
- They ask a question or for something you'd have to look up → a brief, honest, anchored defer: "One sec — pulling up your {thing} now." / "Let me check on {the thing}."
- You're acknowledging what you're fetching or clarifying → name it: "Sure — getting your {reply / number / details} now."
- A greeting → greet back warmly: "Hey there!" / "Hi — good to hear you."

One short line only, always tied to what they just said."""

_ALREADY_DEFERRED_NOTE = """\
You have already deferred to this caller and are STILL waiting for the real reply
to land. Do not start a fresh lookup or re-explain why — just briefly acknowledge
you're still on it, anchored to what they're waiting on ("Bear with me, almost
there with your {thing}.") or, if they just gave you space, simply thank them."""

# Present ONLY when the smarter system bundled a note for this exact moment. It
# is the single, deliberate exception to the no-real-information HARD RULE. Kept
# general (no domain-specific concepts) — the note itself carries any specifics.
_GUIDANCE_NOTE = """\
The smarter system has handed you a short note to help with THIS moment:

{guidance}

Use it ONLY to directly respond to what the caller JUST said (e.g. confirm or
answer the specific thing they just asked). This is the one case where you may
give that piece of real information. Follow any instruction in the note exactly —
especially any "do not reveal / only confirm if…" constraint. NEVER volunteer it,
bring it up unprompted, or use it for anything they did not just ask about. If
their message is unrelated to the note, ignore the note and reply as normal. Keep
it to one short line."""


def _clean(raw: object) -> str:
    """Normalize whitespace and strip wrapping quotes from the model output."""
    return " ".join(str(raw).split()).strip().strip("\"'\u201c\u201d\u2018\u2019")


# Resumption (fast brain picking up an interrupted slow-brain line) ----------

# Sentinels the continuation classifier returns. CONTINUE = resume the unheard
# remainder; DEFER = the barge-in redirected, so leave it to the slow brain.
_DEFER_SENTINEL = "DEFER"
_CONTINUE_SENTINEL = "CONTINUE"

# Fixed bridge phrases prepended to a resumed line. Authored HERE, never by the
# model, so they can never overlap/duplicate the resumed content (the weak model
# used to copy the remainder's first sentence as its "lead-in", producing
# back-to-back duplication). Rotated for variety so resumes don't sound robotic.
_RESUME_LEAD_INS = (
    "Sorry — as I was saying,",
    "Right, where was I —",
    "Okay, picking up where I left off —",
    "So, to finish that thought —",
    "Anyway, as I was saying —",
    "Right, continuing —",
    "Sorry about that — so,",
)


def pick_resume_lead_in() -> str:
    """A fixed, non-overlapping bridge phrase to prepend to a resumed line.

    Authored here (never by the model) so it can never duplicate the resumed
    content; rotated for variety. Shared by the classifier (CONTINUE) and the
    speechless auto-resume path.
    """
    return random.choice(_RESUME_LEAD_INS)


_CONTINUATION_PROMPT = """\
You were speaking on a live call and the caller cut you off mid-sentence. The
EXACT words you had left to say will be resumed automatically — you do NOT write
them. Your ONLY job: decide whether resuming them now is right, given what the
caller just said.

Reply with exactly one word. CONTINUE is the strong default — resume unless there
is a clear, explicit reason not to.

- CONTINUE — the remaining words are almost always exactly what the caller needs.
  Choose this for anything that is not an explicit redirect: greetings, "go on",
  filler, thinking aloud, agreeing, partial overlap, talking over you, or simply
  continuing to speak. If unsure, CONTINUE.
- DEFER — ONLY when continuing would plainly be wrong because the caller
  EXPLICITLY: changed the subject to something else, declined/objected, told you
  to stop or wait, asked a specific question the remaining words clearly do not
  answer, OR indicated they have ALREADY done (or are already doing) the very
  thing your remaining words instruct ("I already clicked it", "just did that",
  "I already replied", "I've got it") — re-delivering a completed instruction is
  wrong, so DEFER.

It is fine if a genuine redirect later needs its own turn — the caller keeps
talking anyway. The costly mistake is failing to resume content that was right
there, forcing a long silence. So lean hard toward CONTINUE.

CRUCIAL — these are NOT reasons to defer; they are the caller inviting you to say
your piece, so reply CONTINUE:
- They just answered with a greeting: "Hello?", "Hello, can you hear me?", "Hi".
- They asked what this is / why you're contacting them: "Why are you calling?",
  "What's this about?", "What did you need?".
Answering a call and asking "why are you calling?" is the single most obvious case
to CONTINUE — do not treat it as a new question.

Output ONLY the single word CONTINUE or DEFER — nothing else."""


def compute_resume_text(full: str, spoken: str) -> str:
    """Return the unheard tail of ``full``, backed up to a clean resume point.

    ``spoken`` is the prefix the caller actually heard before barging in (usually
    cutting a sentence mid-way). We return the remainder, but rewound to the start
    of the sentence that was cut, so resuming repeats the partial sentence cleanly
    rather than starting mid-word. Falls back to the raw remainder when ``spoken``
    is not a prefix of ``full`` or no sentence boundary precedes the cut.
    """
    full = (full or "").strip()
    spoken = (spoken or "").strip()
    if not full:
        return ""
    if not spoken or not full.startswith(spoken):
        return full
    # Rewind to just after the last sentence terminator inside the spoken prefix,
    # so the partially-spoken sentence is resumed from its beginning.
    boundary = max(spoken.rfind(c) for c in ".!?")
    if boundary == -1:
        return full[len(spoken) :].strip()
    return full[boundary + 1 :].strip()


async def select_continuation(resume_text: str, user_text: str) -> str | None:
    """Return a fixed lead-in to resume an interrupted line, or ``None`` to defer.

    The model ONLY classifies CONTINUE vs DEFER given the caller's barge-in; the
    lead-in itself is drawn from a fixed bank (`_RESUME_LEAD_INS`) so it can never
    duplicate the resumed content — the caller then hears "{fixed lead-in}
    {verbatim remainder}". ``resume_text`` is the slow brain's own remaining
    content (appended by the caller, never composed here); ``user_text`` is the
    barge-in. Returns ``None`` (let the slow brain handle it) when the model
    classifies a redirect, on empty input, or any error.
    """
    if not (resume_text or "").strip() or not (user_text or "").strip():
        return None
    messages = [
        {"role": "system", "content": _CONTINUATION_PROMPT},
        {
            "role": "user",
            "content": (
                f"The words you still have left to say: {resume_text.strip()}\n\n"
                f"What the caller just said: {user_text.strip()}"
            ),
        },
    ]
    try:
        client = new_llm_client(
            SETTINGS.conversation.FAST_BRAIN_MODEL,
            origin="FastBrain.continuation",
            reasoning_effort="low",
        )
        raw = await client.generate(messages=messages)
        decision = _clean(raw).upper()
        # Defer on an explicit DEFER, empty output, or anything unexpected that is
        # not a clear CONTINUE; otherwise resume with a fixed, non-overlapping
        # lead-in. (Default-to-continue is preserved for a clear CONTINUE.)
        if not decision or decision == _DEFER_SENTINEL:
            return None
        return pick_resume_lead_in()
    except Exception as e:  # never let continuation selection break the turn
        LOGGER.warning(f"Continuation selection failed; deferring: {e}")
    return None


async def select_fast_reply(
    user_text: str,
    recent_assistant_text: str = "",
    already_deferred: bool = False,
    guidance: str = "",
) -> str:
    """Return the fast brain's brief, in-the-moment reply for the utterance.

    The reply is freeform but template-guided (see the prompt) and must never be
    a substantive answer. ``recent_assistant_text`` is the assistant's previous
    spoken line (given as context + an anti-repeat nudge). ``already_deferred``
    marks a repeated deferral in the same wait streak, so the reply acknowledges
    the wait rather than starting a fresh lookup. ``guidance`` is an optional short note
    the smarter system bundled for this moment; when present it adds a scoped
    block letting the reply directly answer the caller's just-said point (the one
    exception to the no-real-information rule).

    Fail-safe: returns the default phrase on empty input, an over-long reply
    (model overreaching into a real answer), or any error.
    """
    if not (user_text or "").strip():
        return _DEFAULT_PHRASE
    messages = [{"role": "system", "content": _FAST_REPLY_PROMPT}]
    note = (guidance or "").strip()
    if note:
        messages.append(
            {"role": "system", "content": _GUIDANCE_NOTE.format(guidance=note)},
        )
    if already_deferred:
        messages.append({"role": "system", "content": _ALREADY_DEFERRED_NOTE})
    prev = (recent_assistant_text or "").strip()
    if prev:
        messages.append({"role": "assistant", "content": prev})
        messages.append(
            {
                "role": "system",
                "content": "That was your previous line — do not repeat it; "
                "say something different.",
            },
        )
    messages.append({"role": "user", "content": user_text.strip()})
    try:
        client = new_llm_client(
            SETTINGS.conversation.FAST_BRAIN_MODEL,
            origin="FastBrain.reply",
            reasoning_effort="low",
        )
        raw = await client.generate(messages=messages)
        text = _clean(raw)
        if not text:
            return _DEFAULT_PHRASE
        # Backstop: a genuine reaction/defer is short. A long reply means the
        # model tried to answer substantively — drop it for the safe default.
        if len(text) > _MAX_FAST_REPLY_CHARS:
            return _DEFAULT_PHRASE
        return text
    except Exception as e:  # never let fast-reply selection break the turn
        LOGGER.warning(f"Fast reply selection failed; using default: {e}")
    return _DEFAULT_PHRASE
