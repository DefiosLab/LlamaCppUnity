#nullable enable
using System;
using System.Runtime.InteropServices;
using LlamaCppUnity.Internals;

namespace LlamaCppUnity
{
  namespace LlamaCppWrappers
  {
    public enum LlamaKvOverrideType
    {
      Int = 0,
      Float = 1,
      Bool = 2
    }
    public enum LlamaSplitMode
    {
      None = 0,
      Layer = 1,
      Row = 2
    }
    public enum GgmlNumaStrategy
    {
      Disabled = 0,
      Distribute = 1,
      Isolate = 2,
      Numactl = 3,
      Mirror = 4,
      Count = 5
    }
    public enum LlamaRopeScalingType
    {
      Unspecified = -1,
      None = 0,
      Linear = 1,
      Yarn = 2,
      MaxValue = 2
    }
    public delegate bool LlamaProgressCallback(float progress, IntPtr userData);

    public delegate bool GgmlAbortCallback(IntPtr userData);

    // GgmlBackendSchedEvalCallback Reference : https://github.com/ggerganov/llama.cpp/blob/5c4d767ac028c0f9c31cba3fceaf765c6097abfc/ggml-backend.h#L176
    public delegate bool GgmlBackendSchedEvalCallback(IntPtr t, bool ask, IntPtr userData);

    // Create a structure for LlamaModelKvOverrideValue
    [StructLayout(LayoutKind.Explicit)]
    public struct LlamaModelKvOverrideValue
    {
      [FieldOffset(0)]
      public long IntValue;
      [FieldOffset(0)]
      public double FloatValue;
      [FieldOffset(0)]
      public bool BoolValue;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct LlamaGrammarElement
    {
      Int32 Type;
      UInt32 Value;
    }

    // Create a structure for LlamaModelKvOverride
    [StructLayout(LayoutKind.Sequential)]
    public struct LlamaModelKvOverride
    {
      [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 128)]
      public string Key;
      public Int32 Tag;
      public LlamaModelKvOverrideValue Value;
    }

    // Create a structure for LlamaModelParams
    [StructLayout(LayoutKind.Sequential)]
    public struct LlamaModelParams
    {
      public Int32 NGpuLayers;
      public Int32 SplitMode;
      public Int32 MainGpu;

      public IntPtr TensorSplit;

      public IntPtr ProgressCallback;
      public IntPtr ProgressCallbackUserData;

      public IntPtr KvOverrides;
      [MarshalAs(UnmanagedType.I1)]
      public bool VocabOnly;
      [MarshalAs(UnmanagedType.I1)]
      public bool UseMmap;
      [MarshalAs(UnmanagedType.I1)]
      public bool UseMlock;
    }

    // Create a structure for LlamaContextParams
    [StructLayout(LayoutKind.Sequential)]
    public struct LlamaContextParams
    {
      public UInt32 Seed;
      public UInt32 NCtx;
      public UInt32 NBatch;
      public UInt32 NUBatch;
      public UInt32 NSeqMax;
      public UInt32 NThreads;
      public UInt32 NThreadsBatch;
      public LlamaRopeScalingType RopeScalingType;
      public LlamaRopeScalingType PoolingType;
      public float RopeFreqBase;

      // Unity crash factor
      public float RopeFreqScale; // Crash factor???
      public float YarnExtFactor; // Crash factor???
      public float YarnAttnFactor; // Crash factor???
      public float YarnBetaFast; // Crash factor???
      // Unity crash factor

      public float YarnBetaSlow;
      public UInt32 YarnOrigCtx;
      public float DefragThold;
      // public GgmlBackendSchedEvalCallback CbEval;
      public IntPtr CbEval;
      public IntPtr CbEvalData;
      public Int32 TypeK;
      public Int32 TypeV;
      [MarshalAs(UnmanagedType.I1)]
      public bool LogitsAll;
      [MarshalAs(UnmanagedType.I1)]
      public bool Embeddings;
      [MarshalAs(UnmanagedType.I1)]
      public bool OffloadKqv;
      // public GgmlAbortCallback AbortCallback;
      public IntPtr AbortCallback;
      public IntPtr AbortCallbackData;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct LlamaTokenData
    {
      public Int32 Id;
      public float Logit;
      public float P;
    }
    [StructLayout(LayoutKind.Sequential)]
    public struct LlamaTokenDataArrayStruct
    {
      public IntPtr Data;
      public UIntPtr Size; // size_t
      [MarshalAs(UnmanagedType.I1)]
      public bool Sorted;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct LlamaBatchStruct
    {
      public Int32 NTokens;
      public IntPtr Token;
      public IntPtr Embd;
      public IntPtr Pos;
      public IntPtr NSeqId;
      public IntPtr SeqId;
      public IntPtr Logits;
      public Int32 AllPos0;
      public Int32 AllPos1;
      public Int32 AllSeqId;
    }

    public class LlamaCpp
    {
      [DllImport("libllama")]
      public static extern LlamaModelParams llama_model_default_params();

      [DllImport("libllama")]
      public static extern LlamaContextParams llama_context_default_params();

      [DllImport("libllama")]
      public static extern IntPtr llama_load_model_from_file(byte[] filename, LlamaModelParams p);

      [DllImport("libllama")]
      public static extern Int32 llama_vocab_type(IntPtr model);

      [DllImport("libllama")]
      public static extern Int32 llama_n_vocab(IntPtr model);

      [DllImport("libllama")]
      public static extern Int32 llama_n_ctx_train(IntPtr model);

      [DllImport("libllama")]
      public static extern Int32 llama_n_embd(IntPtr model);

      [DllImport("libllama")]
      public static extern float llama_rope_freq_scale_train(IntPtr model);

      [DllImport("libllama")]
      public static extern ulong llama_model_size(IntPtr model);

      [DllImport("libllama")]
      public static extern ulong llama_model_n_params(IntPtr model);

      [DllImport("libllama")]
      public static extern Int32 llama_token_bos(IntPtr model);

      [DllImport("libllama")]
      public static extern Int32 llama_token_eos(IntPtr model);

      [DllImport("libllama")]
      public static extern Int32 llama_token_nl(IntPtr model);

      [DllImport("libllama")]
      public static extern Int32 llama_tokenize(IntPtr model, byte[] text, Int32 text_len, Int32[] tokens, Int32 n_max_tokens, bool add_bos, bool special);

      [DllImport("libllama")]
      public static extern void llama_free_model(IntPtr model);

      [DllImport("libllama")]
      public static extern void llama_free(IntPtr ctx);

      [DllImport("libllama")]
      public static extern IntPtr llama_new_context_with_model(IntPtr model, LlamaContextParams p);

      [DllImport("libllama")]
      public static extern IntPtr llama_token_get_text(IntPtr model, Int32 token);

      [DllImport("libllama")]
      public static extern Int32 llama_token_get_type(IntPtr model, Int32 token);

      [DllImport("libllama")]
      public static extern Int32 llama_model_apply_lora_from_file(IntPtr model, byte[] lora_path, float scale, IntPtr path_base_model, Int32 n_threads);
      [DllImport("libllama")]
      public static extern float llama_token_get_score(IntPtr model, Int32 token);

      [DllImport("libllama")]
      public static extern LlamaBatchStruct llama_batch_init(Int32 n_tokens_alloc, Int32 embd, Int32 n_seq_max);

      [DllImport("libllama")]
      public static extern void llama_batch_free(LlamaBatchStruct batch);

      [DllImport("libllama")]
      public static extern void llama_backend_init();

      [DllImport("libllama")]
      public static extern void llama_numa_init(Int32 numa);

      [DllImport("libllama")]
      public static extern IntPtr llama_max_devices();
      public static Int32 LlamaMaxDevices = (Int32)llama_max_devices();

      [DllImport("libllama")]
      public static extern IntPtr llama_print_system_info();

      [DllImport("libllama")]
      public static extern IntPtr llama_get_model_tensor(IntPtr model, byte[] name);

      [DllImport("libllama")]
      public static extern bool llama_kv_cache_seq_rm(IntPtr ctx, Int32 seqId, Int32 q0, Int32 q1);

      [DllImport("libllama")]
      public static extern UInt32 llama_n_ctx(IntPtr ctx);

      [DllImport("libllama")]
      public static extern Int32 llama_model_meta_count(IntPtr model);

      [DllImport("libllama")]
      public static extern Int32 llama_model_meta_key_by_index(IntPtr model, Int32 i, byte[] buffer, IntPtr bufferSize);

      [DllImport("libllama")]
      public static extern Int32 llama_model_meta_val_str_by_index(IntPtr model, Int32 i, byte[] buffer, IntPtr bufferSize);

      [DllImport("libllama")]
      public static extern Int32 llama_model_desc(IntPtr model, byte[] buf, Int32 buf_size);

      [DllImport("libllama")]
      public static extern Int32 llama_token_prefix(IntPtr model);

      [DllImport("libllama")]
      public static extern Int32 llama_token_middle(IntPtr model);

      [DllImport("libllama")]
      public static extern Int32 llama_token_suffix(IntPtr model);

      [DllImport("libllama")]
      public static extern Int32 llama_token_eot(IntPtr model);

      [DllImport("libllama")]
      public static extern IntPtr llama_get_logits_ith(IntPtr ctx, Int32 i);

      [DllImport("libllama")]
      public static extern Int32 llama_token_to_piece(IntPtr model, Int32 token, byte[] buf, Int32 length);

      [DllImport("libllama")]
      public static extern void llama_reset_timings(IntPtr ctx);

      [DllImport("libllama")]
      public static extern void llama_set_rng_seed(IntPtr ctx, UInt32 seed);

      [DllImport("libllama")]
      public static extern IntPtr llama_get_logits(IntPtr ctx);

      [DllImport("libllama")]
      public static extern Int32 llama_decode(IntPtr ctx, LlamaBatchStruct batch);

      [DllImport("libllama")]
      public static extern void llama_sample_repetition_penalties(IntPtr ctx, IntPtr candidates, IntPtr last_tokens, IntPtr penalty_last_n, float penalty_repeat, float penalty_freq, float penalty_present);

      [DllImport("libllama")]
      public static extern void llama_sample_softmax(IntPtr ctx, IntPtr candidates);

      [DllImport("libllama")]
      public static extern Int32 llama_sample_token_greedy(IntPtr ctx, IntPtr candidates);

      [DllImport("libllama")]
      public static extern void llama_sample_temp(IntPtr ctx, IntPtr candidates, float temp);

      [DllImport("libllama")]
      public static extern Int32 llama_sample_token_mirostat(IntPtr ctx, IntPtr candidates, float tau, float eta, Int32 m, IntPtr mu);

      [DllImport("libllama")]
      public static extern Int32 llama_sample_token_mirostat_v2(IntPtr ctx, IntPtr candidates, float tau, float eta,  IntPtr mu);

      [DllImport("libllama")]
      public static extern void llama_sample_top_k(IntPtr ctx, IntPtr candidates, Int32 k, IntPtr min_keep);

      [DllImport("libllama")]
      public static extern void llama_sample_tail_free(IntPtr ctx, IntPtr candidates, float z, IntPtr min_keep);

      [DllImport("libllama")]
      public static extern void llama_sample_typical(IntPtr ctx, IntPtr candidates, float p, IntPtr min_keep);

      [DllImport("libllama")]
      public static extern void llama_sample_top_p(IntPtr ctx, IntPtr candidates, float p, IntPtr min_keep);

      [DllImport("libllama")]
      public static extern Int32 llama_sample_token(IntPtr ctx, IntPtr candidates);

      [DllImport("libllama")]
      public static extern Int32 llama_sample_min_p(IntPtr ctx, IntPtr candidates, float p, IntPtr minkeep);

      [DllImport("libllama")]
      public static extern Int32 llama_grammar_accept_token(IntPtr ctx, IntPtr grammar, Int32 token);

      [DllImport("libllama")]
      public static extern IntPtr llama_grammar_init(IntPtr rules, IntPtr n_rules, IntPtr start_rule_index);

      [DllImport("libllama")]
      public static extern void llama_grammar_free(IntPtr grammar);
    }
  }
}
