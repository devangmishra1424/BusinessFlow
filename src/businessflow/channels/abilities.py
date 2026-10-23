"""The "what I can help with" buttons, shared by the browser chat and the Telegram bot so both always say the same thing.

A borrower who does not know what to ask sees a short list of things to tap instead of a blank box. Tapping one is the same
as typing its `prompt`: it goes through the normal assistant turn, with the same checks and the same tools. An entry with a
`reply` instead is answered as it stands, with no assistant turn (how to verify, or the "type your question" nudge).

Deliberately modest, and checked by tests/test_abilities.py:
  * Each entry names something the assistant really has a tool for (balance and due date, a payment link, payment history,
    reviewing a request for more time, recording a dispute, a closure check, the policy documents, passing the chat to a
    person). The wording says "ask about" where the tool only reviews or checks, because a review is not a promise.
  * Nothing says "everything" or "anything". The last button is always "More questions or doubts", which is the honest way
    to say the list is not the limit, and its reply says the assistant will say so when it cannot answer.
  * What is shown depends on who is asking. An unverified chat cannot see account figures (the assistant refuses account
    tools for it), so it gets only the general entries and how to verify, never a button that would end in a refusal.

No imports from the voice, database or Telegram stacks, so it loads anywhere and is cheap to test. The Hindi wording avoids
verb forms that depend on the speaker's gender; a Hindi speaker should still read it before it is relied on.
"""

from dataclasses import dataclass

LANGUAGES = ("en", "hi")

# Who sees an entry: a verified chat, an unverified one, or both.
VERIFIED, ANONYMOUS, EVERYONE = "verified", "anonymous", "everyone"


@dataclass(frozen=True)
class Ability:
    id: str
    audience: str
    label: dict[str, str]
    # Exactly one of these is set. `prompt` is sent to the assistant as if the borrower had typed it; `reply` is shown as is.
    prompt: dict[str, str] | None = None
    reply: dict[str, str] | None = None


_INTRO = {
    "en": "Here are a few things I can help with. Tap one, or just type your question.",
    "hi": "ये कुछ चीज़ें हैं जिनमें मदद मिल सकती है। किसी एक को चुनें, या अपना सवाल सीधे लिखें।",
}

# The order here is the order shown. "more" is last, always.
_ABILITIES: tuple[Ability, ...] = (
    Ability(
        "status", VERIFIED,
        label={"en": "Check my balance and due date", "hi": "बकाया और अगली तारीख़ देखें"},
        prompt={"en": "What's my balance and when is my next EMI due?", "hi": "मेरा बकाया कितना है और अगली EMI कब है?"},
    ),
    Ability(
        "pay", VERIFIED,
        label={"en": "Get a payment link", "hi": "भुगतान का लिंक पाएँ"},
        prompt={"en": "I'd like a payment link for what's due now.", "hi": "मुझे अभी जो बकाया है उसके लिए भुगतान का लिंक चाहिए।"},
    ),
    Ability(
        "history", VERIFIED,
        label={"en": "See my recent payments", "hi": "हाल के भुगतान देखें"},
        prompt={"en": "Show me my recent payments.", "hi": "मेरे हाल के भुगतान दिखाइए।"},
    ),
    Ability(
        "promise", VERIFIED,
        label={"en": "Make or check a promise to pay", "hi": "भुगतान का वादा करें या देखें"},
        prompt={"en": "What promise to pay do I have on record, and how do I make one?",
                "hi": "मेरा भुगतान का कौन-सा वादा दर्ज है, और नया वादा कैसे करूँ?"},
    ),
    Ability(
        "requests", VERIFIED,
        label={"en": "My requests and your replies", "hi": "मेरे अनुरोध और आपके जवाब"},
        prompt={"en": "What has happened to my requests, and has your team replied?",
                "hi": "मेरे अनुरोधों का क्या हुआ, और क्या आपकी टीम ने जवाब दिया?"},
    ),
    Ability(
        "more_time", VERIFIED,
        label={"en": "Ask about more time to pay", "hi": "भुगतान के लिए और समय के बारे में पूछें"},
        prompt={"en": "I need more time to pay my EMI. What are my options?", "hi": "मुझे EMI भरने के लिए और समय चाहिए। मेरे पास क्या विकल्प हैं?"},
    ),
    Ability(
        "dispute", VERIFIED,
        label={"en": "Report a problem or dispute", "hi": "कोई समस्या या विवाद बताएँ"},
        prompt={"en": "I want to report a problem with a charge on my account.", "hi": "मुझे अपने अकाउंट में एक चार्ज को लेकर समस्या बतानी है।"},
    ),
    Ability(
        "fee_waiver", VERIFIED,
        label={"en": "Ask to remove a late fee", "hi": "लेट फ़ीस हटाने का अनुरोध करें"},
        prompt={"en": "Can you remove the late fee on my account? I have a reason.",
                "hi": "क्या आप मेरे अकाउंट की लेट फ़ीस हटा सकते हैं? मेरे पास कारण है।"},
    ),
    Ability(
        "closure", VERIFIED,
        label={"en": "Ask about loan closure", "hi": "लोन बंद करने के बारे में पूछें"},
        prompt={"en": "Can I get a loan closure certificate?", "hi": "क्या मुझे लोन बंद करने का प्रमाणपत्र मिल सकता है?"},
    ),
    Ability(
        "policy", EVERYONE,
        label={"en": "Late-fee and payment rules", "hi": "लेट फ़ीस और भुगतान के नियम"},
        prompt={"en": "What are the late-fee and payment rules?", "hi": "लेट फ़ीस और भुगतान के नियम क्या हैं?"},
    ),
    Ability(
        "person", VERIFIED,
        label={"en": "Talk to a person", "hi": "हमारी टीम के किसी व्यक्ति से बात करें"},
        prompt={"en": "I'd like to talk to a person on your team.", "hi": "मुझे आपकी टीम के किसी व्यक्ति से बात करनी है।"},
    ),
    Ability(
        "verify", ANONYMOUS,
        label={"en": "How do I verify my account?", "hi": "अकाउंट वेरिफ़ाई कैसे करें"},
        reply={
            "en": 'To look at your own account, send your account ID and 6-digit access key together in one message, e.g. "BF-1001 482913".',
            "hi": 'अपना अकाउंट देखने के लिए अकाउंट ID और 6 अंकों की एक्सेस की एक ही संदेश में भेजें, जैसे "BF-1001 482913"।',
        },
    ),
    Ability(
        "more", EVERYONE,
        label={"en": "More questions or doubts", "hi": "और सवाल या शंकाएँ"},
        reply={
            "en": "Of course. Type your question or doubt below (or send a voice message). If I can't answer it, I'll say so.",
            "hi": "ज़रूर। अपना सवाल या शंका नीचे लिखें (या आवाज़ में भेजें)। अगर उसका जवाब नहीं मिल पाया, तो यह भी साफ़ बता दिया जाएगा।",
        },
    ),
)


def _language(language: str) -> str:
    return language if language in LANGUAGES else "en"


def _shown_to(ability: Ability, verified: bool) -> bool:
    return ability.audience == EVERYONE or (ability.audience == VERIFIED) == verified


def intro(language: str) -> str:
    return _INTRO[_language(language)]


def abilities_for(language: str, verified: bool) -> list[dict]:
    """The buttons for a chat, in display order: [{id, label, prompt, reply}] with the unused one of prompt/reply as None."""
    lang = _language(language)
    return [
        {
            "id": a.id,
            "label": a.label[lang],
            "prompt": a.prompt[lang] if a.prompt else None,
            "reply": a.reply[lang] if a.reply else None,
        }
        for a in _ABILITIES
        if _shown_to(a, verified)
    ]


def find_ability(language: str, verified: bool, ability_id: str) -> dict | None:
    """One button by id, but only if this chat is allowed to see it (a button left over from before verification or a reset,
    or an id made up by a client, finds nothing). Telegram looks the text up here by id instead of trusting what the tap carries."""
    return next((a for a in abilities_for(language, verified) if a["id"] == ability_id), None)
