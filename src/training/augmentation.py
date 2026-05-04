"""Turkish text augmentation — synonym replacement + random token swap."""
import random
import re


TURKISH_SYNONYMS = {
    "iyi": ["güzel", "harika", "kaliteli", "mükemmel"],
    "güzel": ["iyi", "harika", "şahane", "kaliteli"],
    "harika": ["mükemmel", "süper", "iyi", "güzel"],
    "kötü": ["berbat", "rezalet", "kalitesiz", "yetersiz"],
    "berbat": ["kötü", "rezalet", "korkunç", "berbat"],
    "hızlı": ["çabuk", "süratli", "acele"],
    "yavaş": ["geç", "ağır", "bekletici"],
    "pahalı": ["masraflı", "fahiş", "yüksek fiyatlı"],
    "ucuz": ["ekonomik", "uygun fiyatlı", "makul"],
    "memnun": ["mutlu", "tatmin", "hoşnut"],
    "sağlam": ["dayanıklı", "güçlü", "kaliteli"],
    "bozuk": ["arızalı", "kırık", "hasarlı"],
    "tavsiye": ["öneri", "salık"],
    "teşekkür": ["sağ ol", "minnettarım"],
}


def synonym_replace(text: str, p: float = 0.15) -> str:
    words = text.split()
    result = []
    for w in words:
        w_low = w.lower()
        if w_low in TURKISH_SYNONYMS and random.random() < p:
            syn = random.choice(TURKISH_SYNONYMS[w_low])
            # preserve capitalisation
            result.append(syn.capitalize() if w[0].isupper() else syn)
        else:
            result.append(w)
    return " ".join(result)


def random_swap(text: str, n: int = 1) -> str:
    words = text.split()
    if len(words) < 2:
        return text
    for _ in range(n):
        i, j = random.sample(range(len(words)), 2)
        words[i], words[j] = words[j], words[i]
    return " ".join(words)


def random_delete(text: str, p: float = 0.1) -> str:
    words = text.split()
    if len(words) == 1:
        return text
    return " ".join(w for w in words if random.random() > p) or words[0]


def augment(text: str, label: int) -> list[tuple[str, int]]:
    """Returns list of (augmented_text, label) pairs including original."""
    results = [(text, label)]
    results.append((synonym_replace(text), label))
    results.append((random_swap(text, n=2), label))
    return results
