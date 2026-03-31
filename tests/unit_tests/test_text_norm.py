from torchtitan.tools.text_norm import extract_language_token, normalize_text_for_eval


def test_normalize_text_for_eval_preserves_native_script_letters():
    text = "नमस्ते, दुनिया! यह एक TEST है."
    assert normalize_text_for_eval(text) == "नमस्ते दुनिया यह एक test है"


def test_normalize_text_for_eval_collapses_punctuation_and_whitespace():
    text = "Hello,\n\nWorld!!  2026?"
    assert normalize_text_for_eval(text) == "hello world 2026"


def test_extract_language_token_from_prompt_text():
    text = "<lang_hi>नमस्ते<audio><1_0></audio>"
    assert extract_language_token(text) == "hi"
