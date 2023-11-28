# Don't forget to support cases when target_text == ''
import editdistance


def calc_cer(target_text, predicted_text) -> float:
    target_chars = list(target_text)
    pred_chars = list(predicted_text)
    if len(target_chars) == 0:
        return 0 if len(pred_chars) == 0 else 1
    return editdistance.eval(target_chars, pred_chars) / len(target_chars)


def calc_wer(target_text, predicted_text) -> float:
    target_words = target_text.split()
    pred_words = predicted_text.split()
    if len(target_words) == 0:
        return 0 if len(pred_words) == 0 else 1
    return editdistance.eval(target_words, pred_words) / len(target_words)
