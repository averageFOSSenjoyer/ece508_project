#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

pub(crate) mod constants;
pub(crate) mod tokenizer;

use std::{collections::HashMap, ffi::CString, thread, time::SystemTime, io::{self, Write}};
use colored::Colorize;

use anyhow::Result;
use clap::Parser;
use env_logger;
use log::{debug, info};
use crossbeam_channel::unbounded;

#[derive(Parser, Debug)]
struct Args {
    #[arg(short, long)]
    tokenizer_path: String,

    #[arg(short, long)]
    model_path: String,

    #[arg(short, long)]
    prompt: String,
}

#[derive(Hash, Eq, PartialEq, Debug)]
enum TimeCategory {
    RMS,
    MATMUL,
    QK_NORM,
    ROPE,
    MHA,
    SWIGLU,
}

fn main() -> Result<()> {
    env_logger::init();

    let args = Args::parse();
    let (tx, rx) = unbounded();

    info!("Loading tokenizer...");
    let tokenizer = tokenizer::Tokenizer::new(args.tokenizer_path)?;

    let processed_prompt = format!("<|im_start|>user
    {}<|im_end|>
    <|im_start|>assistant
    <think>

    </think>", args.prompt);

    let encoded_token_id = tokenizer.encode(processed_prompt)?;
    debug!("encoded_token_id: {:?}", encoded_token_id);

    let prompt_token_len = encoded_token_id.len();

    let th = thread::spawn(move || -> Result<(HashMap::<TimeCategory, u128>, i32)> {
        let mut total_token_generated = 0;
        let mut time_map = HashMap::<TimeCategory, u128>::new();
        time_map.insert(TimeCategory::RMS, 0);
        time_map.insert(TimeCategory::MATMUL, 0);
        time_map.insert(TimeCategory::QK_NORM, 0);
        time_map.insert(TimeCategory::ROPE, 0);
        time_map.insert(TimeCategory::MHA, 0);
        time_map.insert(TimeCategory::SWIGLU, 0);

        info!("Loading weights...");
        let mut weights = unsafe { crate::load_weights(CString::new(args.model_path)?.as_ptr()) };

        info!("Initiating run state...");
        let mut run_state = unsafe { crate::init_run_state() };

        info!("Initiating CUBLAS handle...");
        let handle = unsafe { crate::init_cublas_handle() };

        let mut pos: usize = 0;
        let mut next_token_id = 0;

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
                let now = SystemTime::now();
                unsafe {
                    rmsnorm(run_state.xb, run_state.x, layer.input_layernorm, constants::HIDDEN_SIZE);
                }
                time_map.entry(TimeCategory::RMS).and_modify(|v| *v += now.elapsed().unwrap().as_micros());

                // 3. KVQ
                let now = SystemTime::now();
                let k_cache_pos: *mut __nv_bfloat16_raw = unsafe { run_state.key_cache.offset((l * constants::SEQ_LEN * constants::KV_DIM + pos * constants::KV_DIM).try_into()?) };
                let v_cache_pos: *mut __nv_bfloat16_raw = unsafe { run_state.value_cache.offset((l * constants::SEQ_LEN * constants::KV_DIM + pos * constants::KV_DIM).try_into()?) };
                unsafe {
                    crate::matmul_cublas(handle, run_state.q, layer.attention.q_proj, run_state.xb, constants::Q_DIM, constants::HIDDEN_SIZE, 1.0, 0.0);
                    crate::matmul_cublas(handle, k_cache_pos, layer.attention.k_proj, run_state.xb, constants::KV_DIM, constants::HIDDEN_SIZE, 1.0, 0.0);
                    crate::matmul_cublas(handle, v_cache_pos, layer.attention.v_proj, run_state.xb, constants::KV_DIM, constants::HIDDEN_SIZE, 1.0, 0.0);
                }
                time_map.entry(TimeCategory::MATMUL).and_modify(|v| *v += now.elapsed().unwrap().as_micros());

                // 4. QK-Norm
                let now = SystemTime::now();
                unsafe {
                    qk_norm_fused(run_state.q, k_cache_pos, layer.attention.q_norm, layer.attention.k_norm);
                };
                time_map.entry(TimeCategory::QK_NORM).and_modify(|v| *v += now.elapsed().unwrap().as_micros());

                // 5. RoPE
                let now = SystemTime::now();
                unsafe {
                    rope(run_state.q, k_cache_pos, pos);
                };
                time_map.entry(TimeCategory::ROPE).and_modify(|v| *v += now.elapsed().unwrap().as_micros());

                // 6. mha
                let now = SystemTime::now();
                unsafe {
                    mha(&mut run_state as *mut RunState, l, pos);
                };
                time_map.entry(TimeCategory::MHA).and_modify(|v| *v += now.elapsed().unwrap().as_micros());

                // 7. final attention output projection and residual connection
                let now = SystemTime::now();
                unsafe {
                    matmul_cublas(handle, run_state.x, layer.attention.o_proj, run_state.q, constants::HIDDEN_SIZE, constants::Q_DIM, 1.0, 1.0);
                };
                time_map.entry(TimeCategory::MATMUL).and_modify(|v| *v += now.elapsed().unwrap().as_micros());

                // 8. post-attention RMSNorm
                let now = SystemTime::now();
                unsafe {
                    rmsnorm(run_state.xb, run_state.x, layer.post_attention_layernorm, constants::HIDDEN_SIZE);
                };
                time_map.entry(TimeCategory::RMS).and_modify(|v| *v += now.elapsed().unwrap().as_micros());

                // 9. FFN projections (Gate and Up)
                // output of w1 matmul is s->hb. output of w3 matmul is s->hb2.
                let now = SystemTime::now();
                unsafe {
                    matmul_cublas(handle, run_state.hb, layer.ffn.gate_proj, run_state.xb, constants::INTERMEDIATE_SIZE, constants::HIDDEN_SIZE, 1.0, 0.0);
                };
                unsafe {
                    matmul_cublas(handle, run_state.hb2, layer.ffn.up_proj, run_state.xb, constants::INTERMEDIATE_SIZE, constants::HIDDEN_SIZE, 1.0, 0.0);
                };
                time_map.entry(TimeCategory::MATMUL).and_modify(|v| *v += now.elapsed().unwrap().as_micros());

                // 10. SwiGLU
                let now = SystemTime::now();
                unsafe {
                    swiglu(run_state.hb, run_state.hb2, constants::INTERMEDIATE_SIZE);
                };
                time_map.entry(TimeCategory::SWIGLU).and_modify(|v| *v += now.elapsed().unwrap().as_micros());

                // 11. final FFN Down Projection matmul and residual connection (fused)
                let now = SystemTime::now();
                unsafe {
                    matmul_cublas(handle, run_state.x, layer.ffn.down_proj, run_state.hb, constants::HIDDEN_SIZE, constants::INTERMEDIATE_SIZE, 1.0, 1.0);
                };
                time_map.entry(TimeCategory::MATMUL).and_modify(|v| *v += now.elapsed().unwrap().as_micros());
            }

            // 12. final RMSNorm
            // in-place operation on s->x
            let now = SystemTime::now();
            unsafe {
                rmsnorm(run_state.x, run_state.x, weights.norm, constants::HIDDEN_SIZE);
            };
            time_map.entry(TimeCategory::RMS).and_modify(|v| *v += now.elapsed().unwrap().as_micros());

            // 13. classifier Matmul
            let now = SystemTime::now();
            unsafe {
                matmul_cublas(handle, run_state.logits, weights.lm_head, run_state.x, constants::VOCAB_SIZE, constants::HIDDEN_SIZE, 1.0, 0.0);
            };
            time_map.entry(TimeCategory::MATMUL).and_modify(|v| *v += now.elapsed().unwrap().as_micros());

            // 14. get logits
            let logits = unsafe { get_logits(&mut run_state as *mut RunState) };

            total_token_generated += 1;

            next_token_id = logits._inner.iter()
                .enumerate()
                .max_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap())
                .map(|(i, _)| i)
                .unwrap();

            tx.try_send((next_token_id, pos))?;
            if next_token_id == 151645 || pos > constants::SEQ_LEN {
                break;
            }

            pos += 1;
        }

        Ok((time_map, total_token_generated))
    });

    let now = SystemTime::now();
    loop {
        let (next_token_id, pos) = rx.recv()?;
        let next_token = tokenizer.decode(vec![next_token_id])?;
        if next_token_id == 151645 || pos > constants::SEQ_LEN  {
            break;
        }
        if pos > prompt_token_len {
            print!("{}", next_token.green().bold());
            io::stdout().flush()?;
        }
    }
    println!("");
    let o_now = now.elapsed().unwrap().as_micros();

    let (time_map, total_token_generated) = th.join().unwrap()?;
    let total_time_passed = time_map.iter().map(|(_, v)| *v).collect::<Vec<u128>>().into_iter().sum::<u128>();
    let mut summary_time_map = HashMap::new();
    for (k, v) in time_map {
        summary_time_map.insert(k, v as f64 / total_time_passed as f64);
    }
    info!("(strict) token/s: {}", total_token_generated as f64 / (total_time_passed as f64 / 1e6));
    info!("(loose) token/s: {}", total_token_generated as f64 / (o_now as f64 / 1e6));
    info!("summary_time_map: {:?}", summary_time_map);
    
    Ok(())
}
