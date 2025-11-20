#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

pub(crate) mod constants;
pub(crate) mod tokenizer;

use std::ffi::CString;

use anyhow::Result;
use clap::Parser;
use env_logger;
use log::{debug, info};

#[derive(Parser, Debug)]
struct Args {
    #[arg(short, long)]
    tokenizer_path: String,

    #[arg(short, long)]
    model_path: String,

    #[arg(short, long)]
    prompt: String,
}

fn main() -> Result<()> {
    env_logger::init();

    let args = Args::parse();

    info!("Loading tokenizer...");
    let tokenizer = tokenizer::Tokenizer::new(args.tokenizer_path)?;

    info!("Loading weights...");
    let mut weights = unsafe { crate::load_weights(CString::new(args.model_path)?.as_ptr()) };

    info!("Initiating run state...");
    let mut run_state = unsafe { crate::init_run_state() };

    info!("Initiating CUBLAS handle...");
    let handle = unsafe { crate::init_cublas_handle() };

    let processed_prompt = format!("<|im_start|>user{}<|im_end|><|im_start|>assistant<think>
    
    </think>", args.prompt);
    let mut pos: usize = 0;
    let mut next_token_id = 0;

    let encoded_token_id = tokenizer.encode(processed_prompt)?;
    debug!("encoded_token_id: {:?}", encoded_token_id);

    info!("Generating tokens...");
    loop {
        let token = if pos < encoded_token_id.len() { encoded_token_id[pos] } else { next_token_id };

        // 1. copy embedding of new token
        unsafe {
            crate::copy_embedding(&mut weights as *mut Weights, &mut run_state as *mut RunState, token);
        }

        for l in 0..constants::NUM_HIDDEN_LAYERS {
            let layer = weights.transformer_blocks[l];

            // 2. input rms norm
            unsafe {
                rmsnorm(run_state.xb, run_state.x, layer.input_layernorm, constants::HIDDEN_SIZE);
            }

            // 3. KVQ
            let k_cache_pos: *mut __nv_bfloat16_raw = unsafe { run_state.key_cache.offset((l * constants::SEQ_LEN * constants::KV_DIM + pos * constants::KV_DIM).try_into()?) };
            let v_cache_pos: *mut __nv_bfloat16_raw = unsafe { run_state.value_cache.offset((l * constants::SEQ_LEN * constants::KV_DIM + pos * constants::KV_DIM).try_into()?) };
            unsafe {
                crate::matmul_cublas(handle, run_state.q, layer.attention.q_proj, run_state.xb, constants::Q_DIM, constants::HIDDEN_SIZE, 1.0, 0.0);
                crate::matmul_cublas(handle, k_cache_pos, layer.attention.k_proj, run_state.xb, constants::KV_DIM, constants::HIDDEN_SIZE, 1.0, 0.0);
                crate::matmul_cublas(handle, v_cache_pos, layer.attention.v_proj, run_state.xb, constants::KV_DIM, constants::HIDDEN_SIZE, 1.0, 0.0);
            }

            // 4. QK-Norm
            unsafe {
                qk_norm_fused(run_state.q, k_cache_pos, layer.attention.q_norm, layer.attention.k_norm);
            };

            // 5. RoPE
            unsafe {
                rope(run_state.q, k_cache_pos, pos);
            };

            // 6. mha
            unsafe {
                mha(&mut run_state as *mut RunState, l, pos);
            };

            // 7. final attention output projection and residual connection
            unsafe {
                matmul_cublas(handle, run_state.x, layer.attention.o_proj, run_state.q, constants::HIDDEN_SIZE, constants::Q_DIM, 1.0, 1.0);
            };

            // 8. post-attention RMSNorm
            unsafe {
                rmsnorm(run_state.xb, run_state.x, layer.post_attention_layernorm, constants::HIDDEN_SIZE);
            };

            // 9. FFN projections (Gate and Up)
            // output of w1 matmul is s->hb. output of w3 matmul is s->hb2.
            unsafe {
                matmul_cublas(handle, run_state.hb, layer.ffn.gate_proj, run_state.xb, constants::INTERMEDIATE_SIZE, constants::HIDDEN_SIZE, 1.0, 0.0);
            };
            unsafe {
                matmul_cublas(handle, run_state.hb2, layer.ffn.up_proj, run_state.xb, constants::INTERMEDIATE_SIZE, constants::HIDDEN_SIZE, 1.0, 0.0);
            };

            // 10. SwiGLU
            unsafe {
                swiglu(run_state.hb, run_state.hb2, constants::INTERMEDIATE_SIZE);
            };

            // 11. final FFN Down Projection matmul and residual connection (fused)
            unsafe {
                matmul_cublas(handle, run_state.x, layer.ffn.down_proj, run_state.hb, constants::HIDDEN_SIZE, constants::INTERMEDIATE_SIZE, 1.0, 1.0);
            };
        }

        // 12. final RMSNorm
        // in-place operation on s->x
        unsafe {
            rmsnorm(run_state.x, run_state.x, weights.norm, constants::HIDDEN_SIZE);
        };

        // 13. classifier Matmul
        unsafe {
            matmul_cublas(handle, run_state.logits, weights.lm_head, run_state.x, constants::VOCAB_SIZE, constants::HIDDEN_SIZE, 1.0, 0.0);
        };

        // 14. get logits
        let logits = unsafe { get_logits(&mut run_state as *mut RunState) };

        next_token_id = logits._inner.iter()
            .enumerate()
            .max_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        let next_token = tokenizer.decode(vec![next_token_id])?;
        if next_token_id == 151645 {
            break;
        }
        if pos > encoded_token_id.len() {
            print!("{next_token}");
        }

        pos += 1;
    }

    Ok(())
}
