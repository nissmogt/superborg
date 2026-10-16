# The `.borg` model file (version 1)

One file holds everything the engine needs: architecture config, tokenizer and
weights. The engine memory-maps it, so loading costs no copies and several
processes share the same pages.

All integers are little-endian.

```
offset 0    char[4]  magic = "BORG"
offset 4    u32      version = 1
offset 8    u64      header_len (bytes of UTF-8 JSON that follow)
offset 16   JSON header (header_len bytes)
            zero padding up to the next multiple of 64  -> data_start
data_start  tensor data; every tensor starts at data_start + offset, offset % 64 == 0
```

## JSON header

```json
{
  "format": "borg",
  "version": 1,
  "arch": "llama",                // informational: llama | mistral | qwen2 | qwen3
  "source": "HuggingFaceTB/SmolLM2-135M-Instruct",
  "quant": "q8_0",                // quantization recipe, informational ("q4_1+imatrix" when calibrated)
  "config": {
    "dim": 576,                   // hidden size
    "n_layers": 30,
    "n_heads": 9,
    "n_kv_heads": 3,
    "head_dim": 64,               // may differ from dim / n_heads (Qwen3)
    "hidden_dim": 1536,           // FFN intermediate size
    "vocab_size": 49152,
    "max_seq_len": 8192,
    "norm_eps": 1e-5,
    "rope_theta": 100000.0,
    "rope_scaling": null,         // or {"type": "llama3", "factor": 32.0, "low_freq_factor": 1.0,
                                  //      "high_freq_factor": 4.0, "original_max_position_embeddings": 8192}
                                  // or {"type": "linear", "factor": 2.0}
    "attn_bias": false,           // q/k/v projections carry a bias (Qwen2)
    "qk_norm": false,             // per-head RMSNorm on q and k before RoPE (Qwen3)
    "tie_embeddings": true,       // output projection reuses tok_embd
    "act": "silu"                 // silu (SwiGLU) | gelu_tanh (GeGLU)
  },
  "tokenizer": { ... see below ... },
  "tensors": {
    "tok_embd": {"dtype": "q8_0", "shape": [49152, 576], "offset": 0, "nbytes": 30081024},
    ...
  }
}
```

### Tensor names

Shapes follow PyTorch `nn.Linear.weight`: `[out_features, in_features]`, row-major.
Quantized blocks run along the last dimension.

| name                     | shape                         | notes                         |
|--------------------------|-------------------------------|-------------------------------|
| `tok_embd`               | `[vocab, dim]`                |                               |
| `output_norm`            | `[dim]`                       | f32                           |
| `output`                 | `[vocab, dim]`                | absent when `tie_embeddings`  |
| `blk.N.attn_norm`        | `[dim]`                       | f32                           |
| `blk.N.attn_q`           | `[n_heads*head_dim, dim]`     |                               |
| `blk.N.attn_k`           | `[n_kv_heads*head_dim, dim]`  |                               |
| `blk.N.attn_v`           | `[n_kv_heads*head_dim, dim]`  |                               |
| `blk.N.attn_q.bias` etc. | `[rows]`                      | f32, only when `attn_bias`    |
| `blk.N.attn_q_norm`      | `[head_dim]`                  | f32, only when `qk_norm`      |
| `blk.N.attn_k_norm`      | `[head_dim]`                  | f32, only when `qk_norm`      |
| `blk.N.attn_o`           | `[dim, n_heads*head_dim]`     |                               |
| `blk.N.ffn_norm`         | `[dim]`                       | f32                           |
| `blk.N.ffn_gate`         | `[hidden_dim, dim]`           |                               |
| `blk.N.ffn_up`           | `[hidden_dim, dim]`           |                               |
| `blk.N.ffn_down`         | `[dim, hidden_dim]`           |                               |

Q and K rows keep the Hugging Face order, so RoPE is the "rotate half" (NeoX)
form: dimension `i` pairs with `i + head_dim/2`.

### Data types

| dtype  | layout                                                                                 |
|--------|----------------------------------------------------------------------------------------|
| `f32`  | IEEE float32                                                                           |
| `f16`  | IEEE float16                                                                           |
| `bf16` | bfloat16 (upper 16 bits of a float32)                                                  |
| `q8_0` | blocks of 32 values: `{f16 d; int8 q[32]}` (34 bytes); `x = d * q`                      |
| `q4_0` | blocks of 32 values: `{f16 d; u8 qs[16]}` (18 bytes); element `j<16` is the low nibble  |
|        | of `qs[j]`, element `j+16` the high nibble; `x = d * (nibble - 8)`                      |
| `q4_1` | blocks of 32 values: `{f16 d; f16 m; u8 qs[16]}` (20 bytes); nibbles as in `q4_0`;     |
|        | `x = d * nibble + m`                                                                   |
| `u8`, `u32`, `i32` | plain integers (tokenizer tables)                                          |

`q8_0`, `q4_0` and `q4_1` match the ggml layouts of the same names. A quantized tensor's
last dimension must be a multiple of 32; the converter stores other tensors as
f16/f32.

## Tokenizer

```json
"tokenizer": {
  "byte_level": true,          // GPT-2 byte-level BPE (true) or SentencePiece-style BPE (false)
  "pretokenizer": "smollm",    // gpt2 | llama3 | qwen2 | smollm | none
  "nfc": false,                // HF normalizer applies NFC; callers normalize before tokenizing
  "spm_add_prefix_space": false,  // SentencePiece: prepend "▁" to the text
  "byte_fallback": false,      // SentencePiece: unknown chars become <0xXX> byte tokens
  "ignore_merges": false,      // emit a pre-token directly when it is a whole vocab entry
  "add_bos": false,
  "add_eos": false,
  "bos_id": 1,                 // -1 when absent
  "eos_id": 2,
  "eos_ids": [2],              // every id that ends generation
  "pad_id": 2,
  "unk_id": 0,
  "added_tokens": [0, 1, 2],   // ids matched verbatim in text when parse_special is on
  "chat_template": "...",      // raw Jinja template from tokenizer_config.json, or null
  "chat_format": "chatml",     // chatml | llama3 | zephyr | qwen3 | plain (built-in fallback)
  "n_vocab": 49152,            // informational: rows of tokenizer.vocab_offsets minus one
  "n_merges": 48900            // informational: rows of tokenizer.merges
}
```

Tokenizer tables live in the tensor area:

| tensor                    | dtype | shape            | content                                                  |
|---------------------------|-------|------------------|----------------------------------------------------------|
| `tokenizer.vocab`         | u8    | `[total_bytes]`  | concatenated token byte strings                          |
| `tokenizer.vocab_offsets` | u32   | `[n_vocab + 1]`  | token `i` is `vocab[off[i]:off[i+1]]`                     |
| `tokenizer.token_type`    | u8    | `[n_vocab]`      | 0 normal, 1 byte, 2 special, 3 added (non-special), 4 unused |
| `tokenizer.merges`        | u32   | `[n_merges, 3]`  | `(left_id, right_id, merged_id)`; row index = merge rank |

Token byte strings:

* byte-level BPE: the raw bytes the token stands for (the GPT-2 byte-to-unicode
  mapping is undone by the converter). Added tokens hold their literal UTF-8 text.
* SentencePiece BPE: the UTF-8 piece with `▁` (U+2581) kept as is. Byte tokens
  (`<0x0A>`) hold the single byte and have type 1. Decoding turns `▁` into a space.

`n_vocab` may be smaller than `config.vocab_size` (padded embedding rows); ids at
or above `n_vocab` never decode to text.
