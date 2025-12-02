/// Constants for Qwen/Qwen3-0.6B (https://huggingface.co/Qwen/Qwen3-0.6B/raw/main/config.json)
// {
//   "architectures": [
//     "Qwen3ForCausalLM"
//   ],
//   "attention_bias": false,
//   "attention_dropout": 0.0,
//   "bos_token_id": 151643,
//   "eos_token_id": 151645,
//   "head_dim": 128,
//   "hidden_act": "silu",
//   "hidden_size": 1024,
//   "initializer_range": 0.02,
//   "intermediate_size": 3072,
//   "max_position_embeddings": 40960,
//   "max_window_layers": 28,
//   "model_type": "qwen3",
//   "num_attention_heads": 16,
//   "num_hidden_layers": 28,
//   "num_key_value_heads": 8,
//   "rms_norm_eps": 1e-06,
//   "rope_scaling": null,
//   "rope_theta": 1000000,
//   "sliding_window": null,
//   "tie_word_embeddings": true,
//   "torch_dtype": "bfloat16",
//   "transformers_version": "4.51.0",
//   "use_cache": true,
//   "use_sliding_window": false,
//   "vocab_size": 151936
// }

// qwen 0.6b
// pub(crate) const SEQ_LEN: usize = 8192;
// pub(crate) const HEAD_DIM: usize = 128;
// pub(crate) const HIDDEN_SIZE: usize = 1024;
// pub(crate) const NUM_ATTN_HEADS: usize = 16;
// pub(crate) const NUM_HIDDEN_LAYERS: usize = 28;
// pub(crate) const NUM_KV_HEADS: usize = 8;
// pub(crate) const INTERMEDIATE_SIZE: usize = 3072;
// pub(crate) const VOCAB_SIZE: usize = 151936;


// qwen 8b
pub(crate) const SEQ_LEN: usize = 1024;
pub(crate) const HEAD_DIM: usize = 128;
pub(crate) const HIDDEN_SIZE: usize = 4096;
pub(crate) const NUM_ATTN_HEADS: usize = 32;
pub(crate) const NUM_HIDDEN_LAYERS: usize = 36;
pub(crate) const NUM_KV_HEADS: usize = 8;
pub(crate) const INTERMEDIATE_SIZE: usize = 12288;
pub(crate) const VOCAB_SIZE: usize = 151936;

pub(crate) const KV_DIM: usize = NUM_KV_HEADS * HEAD_DIM;
pub(crate) const Q_DIM: usize = NUM_ATTN_HEADS * HEAD_DIM;