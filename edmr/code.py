"""Channel coding of a burst: whitening, the (24,12) column code and the CRC-4."""
import numpy as np

# Symbols are written as four levels (-3, -1, +1, +3; soft values use 10 per unit). A burst is a preamble
# (symbols 0-15), a sync pattern (16-23), a header codeword (24-35) and 13 payload codewords written column
# by column into a 12 x 13 grid (symbols 36-191). Every symbol is whitened with a fixed 4-symbol pattern.

_PARITY_COLUMNS = ["110101010101", "101100110111", "011101001111", "111011000110", "010100111110", "001100011001",
                   "010011111001", "110001100100", "101111100110", "000110010010", "011110011000", "111001001011"]
_P = np.array([[int(_PARITY_COLUMNS[k][i]) for k in range(12)] for i in range(12)], np.uint8)
_DATA = np.array([[(m >> (11 - i)) & 1 for i in range(12)] for m in range(4096)], np.uint8)
_BITS = np.concatenate([_DATA, (_DATA @ _P.T) % 2], axis=1)
CODEWORDS = _BITS[:, 0::2] * 2 + _BITS[:, 1::2]          # 4096 codewords as 12 four-level digits

# expected soft level of digit d at symbol position p: LEVELS[p % 4][d]
_WHITEN = [3, 2, 0, 1]                                   # whitening symbol, repeating every 4 positions
_LEVEL = {0: 30, 1: -10, 2: 10, 3: -30}
_GRAY = {0: 0, 1: 1, 2: 3, 3: 2}
LEVELS = np.array([[_LEVEL[_GRAY[d] ^ _WHITEN[(p + 1) % 4]] for d in range(4)] for p in range(4)], float)

# one-hot form of the code book: one matrix product gives the distance of all codewords for all columns
ONE_HOT = np.zeros((4096, 48), np.float32)
for _r in range(12):
    ONE_HOT[np.arange(4096), _r * 4 + CODEWORDS[:, _r]] = 1
ONE_HOT_T = np.ascontiguousarray(ONE_HOT.T)


def _crc_matrix():
    a = np.array([[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1], [1, 1, 0, 0]], np.uint8)
    eye = np.eye(4, dtype=np.uint8)
    inv = None
    for b in range(1 << 16):
        x = np.array([(b >> k) & 1 for k in range(16)], np.uint8).reshape(4, 4)
        if ((a @ x) % 2 == eye).all():
            inv = x
            break
    c = [None] * 152
    c[61] = np.array([0, 0, 0, 1], np.uint8)
    for i in range(60, -1, -1):
        c[i] = (a @ c[i + 1]) % 2
    for i in range(62, 152):
        c[i] = (inv @ c[i - 1]) % 2
    return np.array(c)


_CRC = _crc_matrix()


def raw_bits(digits):
    """Digit string (values 0-3) -> bits as sent."""
    out = []
    for ch in digits:
        x = int(ch)
        out += [(x >> 1) ^ (x & 1), x & 1]
    return out


def crc_ok(digits):
    """digits: the 78 data digits of a burst (13 codewords x 6)."""
    b = np.array(raw_bits(digits[:76]), np.uint8)
    x = [int(c) for c in digits[76:]]
    check = np.array([x[0] >> 1, x[0] & 1, x[1] >> 1, x[1] & 1], np.uint8)
    return tuple(((b @ _CRC) % 2) ^ check) == (1, 0, 1, 0)


def message_bits(digits):
    """The 156 message bits (true polarity) of a burst."""
    return [1 - v for v in raw_bits(digits)]


def words_hex(bits):
    """156 bits -> 13 words of 12 bits as hex strings."""
    return ["%03x" % int("".join(map(str, bits[12 * i:12 * i + 12])), 2) for i in range(13)]


def digits_of(bits):
    """156 message bits -> 13 words of 6 digits (inverse of message_bits)."""
    inv = [1 - b for b in bits]
    ds = [((inv[i] ^ inv[i + 1]) << 1) | inv[i + 1] for i in range(0, 156, 2)]
    return ["".join(str(v) for v in ds[6 * i:6 * i + 6]) for i in range(13)]


def with_crc(bits152):
    """Append the 4 CRC bits that make 152 message bits a valid burst. Returns 13 digit words or None."""
    for c in range(16):
        ws = digits_of(list(bits152) + [(c >> 3) & 1, (c >> 2) & 1, (c >> 1) & 1, c & 1])
        if crc_ok("".join(ws)):
            return ws
    return None
