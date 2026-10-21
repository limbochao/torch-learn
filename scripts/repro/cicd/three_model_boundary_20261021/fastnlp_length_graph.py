"""fastNLP BERT word-piece length graph; no model or dataset required."""
import torch
import torch_npu

torch.manual_seed(0)
torch.npu.set_device(0)


def word_piece_lengths(words, table):
    mask = words.ne(0)
    seq_len = mask.sum(dim=-1)
    lengths = table[words].masked_fill(mask.eq(False), 0)
    total = lengths.sum(dim=-1)
    maximum = lengths.sum(dim=-1).max()
    return maximum, mask, seq_len, lengths, total


words = (torch.arange(473, dtype=torch.int64) % 71).reshape(1, 473).npu()
table = (torch.arange(2869, dtype=torch.int64) % 4 + 1).npu()
expected = word_piece_lengths(words, table)
compiled = torch.compile(word_piece_lengths, backend="inductor", dynamic=True, fullgraph=True)
actual = compiled(words, table)
torch.npu.synchronize()
print("INPUT", words.shape, words.stride(), words.dtype, "dynamic=True", flush=True)
print("EAGER", expected[0].item(), expected[2].tolist(), expected[4].tolist(), flush=True)
print("COMPILED", actual[0].item(), actual[2].tolist(), actual[4].tolist(), flush=True)
for name, eager, result in zip(("max", "mask", "seq_len", "lengths", "total"), expected, actual):
    print("CHECK", name, bool(torch.equal(eager, result)), flush=True)
    if name != "max":
        torch.testing.assert_close(result, eager, rtol=0, atol=0)
torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
