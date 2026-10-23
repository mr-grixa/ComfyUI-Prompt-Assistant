"""Splitting a prompt into the tags a heat map can be read per word.

The cross attention maps of a prompt have one row per token, so the prompt has
to be turned into {tag text -> token rows}. ComfyUI tokenizes a BREAK separated
prompt in 77 token chunks and concatenates the conditioning, so the same is done
here and the token indices stay aligned with the map rows.

The decoded text of a token is used to find the comma boundaries and to label
the tags; the token indices are what the maps are actually sliced with, so a
tag whose label decodes imperfectly still gets the right rows.
"""

from __future__ import annotations

BREAK = "BREAK"
# Key no tokenizer stream uses; keeps metadata out of the stream lookup.
METADATA_KEY = "_daam_prompt_text"
END_MARKERS = ("</w>", "▁")


def tokenize_prompt(clip, text):
    """Tokenize like CLIPTextEncode does, one 77 token chunk per BREAK part."""
    chunks = [part.strip() for part in text.split(BREAK)]
    chunks = [part for part in chunks if part] or [""]

    streams = {}
    for chunk in chunks:
        for key, chunk_tokens in clip.tokenize(chunk).items():
            streams.setdefault(key, []).extend(chunk_tokens)
    return streams


def stream_key(tokens):
    """The stream the heat maps are aligned to: clip_l for SDXL, else the first."""
    for key in ("l", "g", "t5xxl"):
        if key in tokens:
            return key
    for key in tokens:
        if not str(key).startswith("_"):
            return key
    raise RuntimeError("[DAAM] the prompt tokenizer returned no stream")


def flatten(tokens, key):
    return [token for chunk in tokens[key] for token in chunk]


def _tokenizer_for(clip, key):
    attribute = {"l": "clip_l", "g": "clip_g"}.get(key, key)
    return getattr(clip.tokenizer, attribute, clip.tokenizer)


def special_ids(clip, key):
    """Start/end/padding ids, learned by tokenizing an empty prompt."""
    ids = {0}
    try:
        empty = clip.tokenize("")[key][0]
        for position in (0, 1, -1):
            try:
                ids.add(empty[position][0])
            except (IndexError, TypeError):
                pass
    except Exception:
        pass
    return ids


def decoded_pieces(clip, key, ids):
    """One (token id, text) pair per token, as the tokenizer writes it."""
    tokenizer = _tokenizer_for(clip, key)
    untokenize = getattr(tokenizer, "untokenize", None)
    if callable(untokenize):
        try:
            return [(token_id, text) for token_id, text in untokenize(ids)]
        except Exception:
            pass
    vocab = getattr(tokenizer, "inv_vocab", None) or {}
    return [(token_id, vocab.get(token_id, "")) for token_id in ids]


def _clean_label(text):
    for marker in END_MARKERS:
        text = text.replace(marker, " ")
    return " ".join(text.split())


def split_tags(clip, text):
    """[{label, indices}] for a prompt, split on commas and BREAK.

    ``indices`` address the token axis of the cross attention maps.
    """
    tokens = tokenize_prompt(clip, text)
    key = stream_key(tokens)
    ids = flatten(tokens, key)
    pieces = decoded_pieces(clip, key, ids)
    special = special_ids(clip, key)

    tags = []
    indices, words = [], []

    def flush():
        if indices:
            label = _clean_label(" ".join(words)) or f"tokens {indices[0]}-{indices[-1]}"
            tags.append({"label": label, "indices": list(indices)})
        indices.clear()
        words.clear()

    for position, (token_id, piece) in enumerate(pieces):
        if token_id in special or piece.startswith("<|"):
            flush()
            continue

        if "," not in piece:
            indices.append(position)
            words.append(piece)
            continue

        # A piece can hold a comma, and sometimes more than one.
        parts = piece.split(",")
        for part_index, part in enumerate(parts):
            if part.strip():
                indices.append(position)
                words.append(part)
            if part_index != len(parts) - 1:
                flush()
    flush()
    return tags


def tag_maps(tags, heat_map, labels=None):
    """One averaged map (h, w) per tag whose rows the heat map holds.

    ``labels`` limits the result to the tags the user asked for; an empty list
    means every tag of the prompt.
    """
    rows_total = heat_map.shape[0]
    wanted = {label.strip().lower() for label in labels or [] if label.strip()}

    used, maps = [], []
    for tag in tags:
        if wanted and tag["label"].strip().lower() not in wanted:
            continue
        valid = [index for index in tag["indices"] if index < rows_total]
        if not valid:
            continue
        used.append(tag["label"])
        maps.append(heat_map[valid].mean(0))
    return used, maps


def pick_heat_map(heatmaps, side, token_count=None, batch_index=0):
    """The map tensor for one batch entry, matched by its token count."""
    by_batch = (heatmaps or {}).get(side) or {}
    if not by_batch:
        return None, None

    entry = by_batch.get(batch_index)
    if entry is None:
        entry = next(iter(by_batch.values()))
    if token_count is not None and token_count not in entry:
        available = ", ".join(str(count) for count in sorted(entry))
        raise RuntimeError(
            f"[DAAM] the prompt is {token_count} tokens but the heat maps were "
            f"collected for {available}. Connect the text that was actually "
            "encoded for sampling."
        )
    row = entry.get(token_count) if token_count is not None else max(entry, key=lambda k: entry[k].shape[0])
    return row, token_count if token_count is not None else row.shape[0]
