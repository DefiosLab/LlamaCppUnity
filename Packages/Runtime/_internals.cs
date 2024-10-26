#nullable enable
using System;
using System.Runtime.InteropServices;
using System.IO;
using System.Text;
using LlamaCppUnity.LlamaCppWrappers;
using LlamaCppUnity.Speculative;
using LlamaCppUnity.Helper;
using LlamaCppUnity.Grammar;
using System.Collections.Generic;
using System.Linq;

namespace LlamaCppUnity
{
  namespace Internals
  {
    public class LlamaModel
    {
      private static IntPtr _llamaFreeModel = IntPtr.Zero;
      private string _pathModel;
      private IntPtr _model;
      private LlamaModelParams _parameters;
      private bool _verbose;

      public LlamaModel(string pathModel, LlamaModelParams parameters, bool verbose = true)
      {
        _pathModel = pathModel;
        _model = IntPtr.Zero;
        _parameters = parameters;
        _verbose = verbose;

        if (!File.Exists(pathModel))
        {
          throw new ArgumentException($"Model path does not exist: {pathModel}");
        }

        _model = LlamaCpp.llama_load_model_from_file(Encoding.UTF8.GetBytes(pathModel), parameters);

        if (_model == IntPtr.Zero)
        {
          throw new ArgumentException($"Failed to load model from file: {pathModel}");
        }
      }
      ~LlamaModel()
      {
        // Destructor: Release resources
        if (_model != IntPtr.Zero)
        {
          LlamaCpp.llama_free_model(_model);
        }
      }

      public Int32 NCtxTrain()
      {
        if (_model == IntPtr.Zero)
        {
          throw new InvalidOperationException("Model is pointing to null.");
        }

        return LlamaCpp.llama_n_ctx_train(_model);
      }

      public Int32 VocabType()
      {
        if (_model == IntPtr.Zero)
        {
          throw new InvalidOperationException("Model is pointing to null.");
        }

        return LlamaCpp.llama_vocab_type(_model);
      }
      public Int32 NVocab()
      {
        if (_model == IntPtr.Zero)
        {
          throw new InvalidOperationException("Model is pointing to null.");
        }

        return LlamaCpp.llama_n_vocab(_model);
      }
      public Int32 NEmbd()
      {
        if (_model == IntPtr.Zero)
        {
          throw new InvalidOperationException("Model is pointing to null.");
        }

        return LlamaCpp.llama_n_embd(_model);
      }
      public float RopeFreqScaleTrain()
      {
        if (_model == IntPtr.Zero)
        {
          throw new InvalidOperationException("Model is pointing to null.");
        }

        return LlamaCpp.llama_rope_freq_scale_train(_model);
      }
      public ulong Size()
      {
        if (_model == IntPtr.Zero)
        {
          throw new InvalidOperationException("Model is pointing to null.");
        }

        return LlamaCpp.llama_model_size(_model);
      }
      public Int32 TokenBos()
      {
        if (_model == IntPtr.Zero)
        {
          throw new InvalidOperationException("Model is pointing to null.");
        }

        return LlamaCpp.llama_token_bos(_model);
      }
      public ulong NParams()
      {
        if (_model == IntPtr.Zero)
        {
          throw new InvalidOperationException("Model is pointing to null.");
        }

        return LlamaCpp.llama_model_n_params(_model);
      }
      public Int32 TokenEos()
      {
        if (_model == IntPtr.Zero)
        {
          throw new InvalidOperationException("Model is pointing to null.");
        }

        return LlamaCpp.llama_token_eos(_model);
      }
      public Int32 TokenNl()
      {
        if (_model == IntPtr.Zero)
        {
          throw new InvalidOperationException("Model is pointing to null.");
        }

        return LlamaCpp.llama_token_nl(_model);
      }
      public String TokenGetText(Int32 token)
      {
        if (_model == IntPtr.Zero)
        {
          throw new InvalidOperationException("Model is pointing to null.");
        }

        IntPtr textPtr = LlamaCpp.llama_token_get_text(_model, token);
        string? result = Marshal.PtrToStringAnsi(textPtr);

        if (result != null)
        {
          return result;
        }
        else
        {
          throw new InvalidOperationException("Failed to marshal data from the provided pointer.");
        }
      }
      public Int32 TokenGetType(Int32 token)
      {
        if (_model == IntPtr.Zero)
        {
          throw new InvalidOperationException("Model is pointing to null.");
        }

        return LlamaCpp.llama_token_get_type(_model, token);
      }
      public float TokenGetScore(Int32 token)
      {
        if (_model == IntPtr.Zero)
        {
          throw new InvalidOperationException("Model is pointing to null.");
        }

        return LlamaCpp.llama_token_get_score(_model, token);
      }
      public Int32 ApplyLoraFromFile(string lora_path, float scale, string? path_base_model, Int32 n_threads)
      {
        if (_model == IntPtr.Zero)
        {
          throw new InvalidOperationException("Model is pointing to null.");
        }

        byte[] loraPathBytes = Encoding.UTF8.GetBytes(lora_path);
        byte[]? pathBaseModelByte;

        if (path_base_model != null)
        {
          pathBaseModelByte  = Encoding.UTF8.GetBytes(path_base_model);
          IntPtr ptr = Marshal.UnsafeAddrOfPinnedArrayElement(pathBaseModelByte, 0);
          return LlamaCpp.llama_model_apply_lora_from_file(_model, loraPathBytes, scale, ptr, n_threads);
        }
        else
        {
          return LlamaCpp.llama_model_apply_lora_from_file(_model, loraPathBytes, scale, IntPtr.Zero, n_threads);
        }


      }
      public IntPtr GetTensor(LlamaModel llama_model, string name)
      {
        if (_model == IntPtr.Zero)
        {
          throw new InvalidOperationException("Model is pointing to null.");
        }

        byte[] nameBytes = Encoding.UTF8.GetBytes(name);

        return LlamaCpp.llama_get_model_tensor(_model, nameBytes);
      }

      public List<Int32> Tokenize(byte[] text, bool addBos, bool special)
      {
        if (_model == IntPtr.Zero)
        {
          throw new InvalidOperationException("Model is pointing to null.");
        }

        Int32 nCtx = NCtxTrain();
        Int32[] tokens = new Int32[nCtx];
        Int32 nTokens = LlamaCpp.llama_tokenize(_model, text, text.Length, tokens, nCtx, addBos, special);

        if (nTokens < 0)
        {
          nTokens = Math.Abs(nTokens);
          tokens = new Int32[nTokens];
          nTokens = LlamaCpp.llama_tokenize(_model, text, text.Length, tokens, nTokens, addBos, special);

          if (nTokens < 0)
          {
            throw new Exception($"Failed to tokenize: text=\"{CCharUtils.CCharToString(text)}\" n_tokens={nTokens}");
          }
        }

        return new List<Int32>(tokens).GetRange(0, nTokens);
      }
      public byte[] Detokenize(List<Int32> tokens)
      {
        if (_model == IntPtr.Zero)
        {
          throw new InvalidOperationException("Model is pointing to null.");
        }

        List<byte> output = new List<byte>();
        Int32 size = 32;
        byte[] buffer = new byte[size];

        foreach (Int32 token in tokens)
        {
          Int32 n = LlamaCpp.llama_token_to_piece(_model, token, buffer, size);

          if (n > size)
          {
            throw new Exception($"Failed to Detokenize.");
          }

          for (int i = 0; i < n; i++)
          {
            output.Add(buffer[i]);
          }
        }

        if (tokens.Count > 0 && tokens[0] == TokenBos())
        {
          return output.GetRange(1, output.Count - 1).ToArray();
        }
        else
        {
          return output.ToArray();
        }
      }
      public Dictionary<string, string> MetaData()
      {
        if (_model == IntPtr.Zero)
        {
          throw new InvalidOperationException("Model is pointing to null.");
        }

        Dictionary<string, string> metaData = new Dictionary<string, string>();
        Int32 bufferSize = 1024;
        byte[] buffer = new byte[bufferSize];
        Array.Clear(buffer, 0, buffer.Length);

        for (Int32 i = 0; i < LlamaCpp.llama_model_meta_count(_model); i++)
        {
          Int32 nBytes = LlamaCpp.llama_model_meta_key_by_index(_model, i, buffer, (IntPtr)bufferSize);

          if (nBytes > bufferSize)
          {
            bufferSize = nBytes + 1;
            buffer = new byte[bufferSize];
            nBytes = LlamaCpp.llama_model_meta_key_by_index(_model, i, buffer, (IntPtr)bufferSize);
          }

          string key = CCharUtils.CCharToString(buffer);
          Array.Clear(buffer, 0, buffer.Length);
          nBytes = LlamaCpp.llama_model_meta_val_str_by_index(_model, i, buffer, (IntPtr)bufferSize);

          if (nBytes > bufferSize)
          {
            bufferSize = nBytes + 1;
            buffer = new byte[bufferSize];
            nBytes = LlamaCpp.llama_model_meta_val_str_by_index(_model, i, buffer, (IntPtr)bufferSize);
          }

          string value = CCharUtils.CCharToString(buffer);
          metaData[key] = value;
        }

        return metaData;
      }

      public String Desc()
      {
        if (_model == IntPtr.Zero)
        {
          throw new InvalidOperationException("Model is pointing to null.");
        }

        byte[] _buf = new byte[1024];

        LlamaCpp.llama_model_desc(_model, _buf, 1024);

        return CCharUtils.CCharToString(_buf);
      }
      public Int32 TokenPrefix()
      {
        if (_model == IntPtr.Zero)
        {
          throw new InvalidOperationException("Model is pointing to null.");
        }

        return LlamaCpp.llama_token_prefix(_model);
      }
      public Int32 TokenMiddle()
      {
        if (_model == IntPtr.Zero)
        {
          throw new InvalidOperationException("Model is pointing to null.");
        }

        return LlamaCpp.llama_token_middle(_model);
      }
      public Int32 TokenSuffix()
      {
        if (_model == IntPtr.Zero)
        {
          throw new InvalidOperationException("Model is pointing to null.");
        }

        return LlamaCpp.llama_token_suffix(_model);
      }
      public Int32 TokenEot()
      {
        if (_model == IntPtr.Zero)
        {
          throw new InvalidOperationException("Model is pointing to null.");
        }

        return LlamaCpp.llama_token_eot(_model);
      }
      public byte[] TokenToPiece(Int32 token)
      {
        if (_model == IntPtr.Zero)
        {
          throw new InvalidOperationException("Model is pointing to null.");
        }

        Int32 size = 32;
        byte[] buf = new byte[size];

        LlamaCpp.llama_token_to_piece(_model, token, buf, size);

        return buf;
      }
      public static LlamaModelParams DefaultParams()
      {
        return LlamaCpp.llama_model_default_params();
      }

      // LlamaModel Getters
      public IntPtr GetModelIntPtr
      {
        get
        {
          return _model;
        }
      }
    }
    public class LlamaSamplingContext
    {
      private LlamaSamplingParams _params;
      private float[] _mirostatMu;
      private LlamaGrammar?  _grammar;
      private List<Int32> _prev;
      private List<LlamaTokenData> _cur;

      public LlamaSamplingContext(LlamaSamplingParams parametors, LlamaGrammar? grammar)
      {
        _prev = new List<Int32>();
        _cur = new List<LlamaTokenData>();
        _grammar = grammar;
        _params = parametors;
        _mirostatMu = new float[1];
      }
      public List<Int32> Prev
      {
        get { return _prev;}
        set { _prev = value;}
      }
      public void Accept(LlamaContext ctxMain, Int32 id, bool applyGrammar)
      {
        if (applyGrammar && _grammar != null)
        {
          ctxMain.GrammarAcceptToken(_grammar, id);
        }

        _prev.Add(id);
      }
      public Int32 Sample(LlamaContext ctxMain, Int32 idx = 0, float[]? logitsArray = null)
      {
        Int32 nVocab = ctxMain.Model.NVocab();
        Int32 id = 0;

        if (logitsArray == null)
        {
          IntPtr logits = ctxMain.GetLogitsIth(idx);
          logitsArray = new float[nVocab];
          Marshal.Copy(logits, logitsArray, 0, nVocab);
        }

        foreach (var item in _params.logitBias)
        {
          Int32 token = item.Key;
          float logitBias = item.Value;
          logitsArray[token] += logitBias;
        }

        LlamaTokenDataArray tokenDataArray = new LlamaTokenDataArray(nVocab: nVocab);
        tokenDataArray.CopyLogits(logitsArray);

        if (_prev.Count > 0)
        {
          Int32 nlToken = ctxMain.Model.TokenNl();
          float nlLogit = logitsArray[nlToken];
          Int32 prev_idx = Math.Max(0, _prev.Count - _params.penaltyLastN);
          Int32[] lastTokens = _prev.GetRange(prev_idx, _prev.Count - prev_idx).ToArray();
          Int32 lastTokensSize = Math.Min(lastTokens.Length, _params.penaltyLastN);

          if (lastTokensSize > 0)
          {
            IntPtr lastTokensP = Marshal.UnsafeAddrOfPinnedArrayElement(lastTokens, 0);
            ctxMain.SampleRepetitionPenalties(
              tokenDataArray,
              lastTokensP,
              lastTokensSize,
              _params.penaltyRepeat,
              _params.penaltyFreq,
              _params.penaltyPresent);
          }

          if (_params.penalizeNl == false)
          {
            tokenDataArray._candidatesData[nlToken].Logit = nlLogit;
          }
        }

        // Not Impl grammar

        if (_params.temp < 0)
        {
          ctxMain.SampleSoftmax(tokenDataArray);
          id = tokenDataArray._candidatesData[0].Id;
        }
        else if (_params.temp == 0)
        {
          id = ctxMain.SampleTokenGreedy(tokenDataArray);
        }
        else
        {
          if (_params.mirostat == 1)
          {
            Int32 mirostatM = 100;
            ctxMain.SampleTemp(tokenDataArray, _params.temp);
            id = ctxMain.SampleTokenMirostat(
                   tokenDataArray,
                   _params.mirostatTau,
                   _params.mirostatEta,
                   mirostatM,
                   Marshal.UnsafeAddrOfPinnedArrayElement(_mirostatMu, 0)
                 );
          }
          else if (_params.mirostat == 2)
          {
            ctxMain.SampleTemp(tokenDataArray, _params.temp);
            id = ctxMain.SampleTokenMirostatV2(
                   tokenDataArray,
                   _params.mirostatTau,
                   _params.mirostatEta,
                   Marshal.UnsafeAddrOfPinnedArrayElement(_mirostatMu, 0)
                 );
          }
          else
          {
            Int32 minKeep = Math.Max(1, _params.nProbs);
            ctxMain.SampleTopK(
              tokenDataArray, _params.topK, minKeep
            );
            ctxMain.SampleTailFree(
              tokenDataArray, _params.tfsZ, minKeep
            );
            ctxMain.SampleTypical(
              tokenDataArray, _params.typicalP, minKeep
            );

            ctxMain.SampleTopP(
              tokenDataArray, _params.topP, minKeep
            );

            ctxMain.SampleMinP(
              tokenDataArray, _params.minP, minKeep
            );

            ctxMain.SampleTemp(tokenDataArray, _params.temp);
            id = ctxMain.SampleToken(tokenDataArray);
          }

        }

        return id;
      }
    }
    public class LlamaContext
    {
      private IntPtr _llamaFree = IntPtr.Zero;
      // This field is public because it is also accessed by other classes.
      private LlamaModel _model;
      private LlamaContextParams _parameters;
      private bool _verbose;
      // This field is public because it is also accessed by other classes.
      private IntPtr _ctx;

      public LlamaContext(LlamaModel llamaModel, LlamaContextParams parameters, bool verbose = true)
      {
        _model = llamaModel;
        _parameters = parameters;
        _verbose = verbose;

        if (_model.GetModelIntPtr == IntPtr.Zero)
        {
          throw new ArgumentException("Model is pointing to null.");
        }

        // llama_new_context_with_model Reference: https://github.com/abetlen/llama-cpp-python/blob/08b16afe11e7b42adec2fed0a781123383476045/llama_cpp/llama_cpp.py#L987C1-L994C36

        _ctx = LlamaCpp.llama_new_context_with_model(_model.GetModelIntPtr, _parameters);

        if (_ctx == IntPtr.Zero)
        {
          throw new ArgumentException("Failed to create llama_context.");
        }
      }

      public void ResetTimings()
      {
        if (_ctx == IntPtr.Zero)
        {
          throw new ArgumentException("Failed to create llama_context.");
        }

        LlamaCpp.llama_reset_timings(_ctx);
      }
      public void GrammarAcceptToken(LlamaGrammar grammar, Int32 token)
      {
        if (_ctx == IntPtr.Zero)
        {
          throw new InvalidOperationException("ctx is pointing to null.");
        }

        if (grammar.Grammar == null)
        {
          throw new InvalidOperationException("grammar is pointing to null.");
        }

        LlamaCpp.llama_grammar_accept_token(_ctx, (IntPtr)grammar.Grammar, token);
      }
      public void SampleRepetitionPenalties(
        LlamaTokenDataArray candidates,
        IntPtr lastTokensData,
        Int32 penaltyLastN,
        float penaltyRepeat,
        float penaltyFreq,
        float penaltyPresent)
      {
        if (_ctx == IntPtr.Zero)
        {
          throw new ArgumentException("Failed to create llama_context.");
        }

        IntPtr ptr = Marshal.AllocHGlobal(Marshal.SizeOf(typeof(LlamaTokenDataArrayStruct)));
        Marshal.StructureToPtr(candidates.Candidates, ptr, false);
        LlamaCpp.llama_sample_repetition_penalties(
          _ctx,
          ptr,
          lastTokensData,
          (IntPtr)penaltyLastN,
          penaltyRepeat,
          penaltyFreq,
          penaltyPresent
        );
        object? result  = Marshal.PtrToStructure(ptr, typeof(LlamaTokenDataArrayStruct));

        if (result != null)
        {
          candidates.Candidates = (LlamaTokenDataArrayStruct)result;
        }
        else
        {
          throw new InvalidOperationException("Failed to marshal data from the provided pointer.");
        }
      }
      public void SampleSoftmax(LlamaTokenDataArray candidates)
      {
        if (_ctx == IntPtr.Zero)
        {
          throw new ArgumentException("Failed to create llama_context.");
        }

        IntPtr ptr = Marshal.AllocHGlobal(Marshal.SizeOf(typeof(LlamaTokenDataArrayStruct)));
        Marshal.StructureToPtr(candidates.Candidates, ptr, false);
        LlamaCpp.llama_sample_softmax(_ctx, ptr);
        object? result = Marshal.PtrToStructure(ptr, typeof(LlamaTokenDataArrayStruct));

        if (result != null)
        {
          candidates.Candidates = (LlamaTokenDataArrayStruct)result;
        }
        else
        {
          throw new InvalidOperationException("Failed to marshal data from the provided pointer.");
        }
      }
      public void SampleTemp(LlamaTokenDataArray candidates, float temp)
      {
        if (_ctx == IntPtr.Zero)
        {
          throw new ArgumentException("Failed to create llama_context.");
        }

        IntPtr ptr = Marshal.AllocHGlobal(Marshal.SizeOf(typeof(LlamaTokenDataArrayStruct)));
        Marshal.StructureToPtr(candidates.Candidates, ptr, false);
        LlamaCpp.llama_sample_temp(
          _ctx, ptr, temp
        );
        object? result = Marshal.PtrToStructure(ptr, typeof(LlamaTokenDataArrayStruct));

        if (result != null)
        {
          candidates.Candidates = (LlamaTokenDataArrayStruct)result;
        }
        else
        {
          throw new InvalidOperationException("Failed to marshal data from the provided pointer.");
        }
      }
      public Int32 SampleTokenMirostat(LlamaTokenDataArray candidates, float tau, float eta, Int32 m, IntPtr mu)
      {
        if (_ctx == IntPtr.Zero)
        {
          throw new ArgumentException("Failed to create llama_context.");
        }

        IntPtr ptr = Marshal.AllocHGlobal(Marshal.SizeOf(typeof(LlamaTokenDataArrayStruct)));
        Marshal.StructureToPtr(candidates.Candidates, ptr, false);
        Int32 result =  LlamaCpp.llama_sample_token_mirostat(
                          _ctx, ptr, tau, eta, m, mu);
        object? marshalResult = Marshal.PtrToStructure(ptr, typeof(LlamaTokenDataArrayStruct));

        if (marshalResult != null)
        {
          candidates.Candidates = (LlamaTokenDataArrayStruct)marshalResult;
        }
        else
        {
          throw new InvalidOperationException("Pointer is invalid and cannot be used for marshaling.");
        }

        return result;
      }

      public Int32 SampleTokenMirostatV2(LlamaTokenDataArray candidates, float tau, float eta, IntPtr mu)
      {
        if (_ctx == IntPtr.Zero)
        {
          throw new ArgumentException("Failed to create llama_context.");
        }

        IntPtr ptr = Marshal.AllocHGlobal(Marshal.SizeOf(typeof(LlamaTokenDataArrayStruct)));
        Marshal.StructureToPtr(candidates.Candidates, ptr, false);
        Int32 result = LlamaCpp.llama_sample_token_mirostat_v2(
                         _ctx, ptr, tau, eta, mu);
        object? marshalResult = Marshal.PtrToStructure(ptr, typeof(LlamaTokenDataArrayStruct));

        if (marshalResult != null)
        {
          candidates.Candidates = (LlamaTokenDataArrayStruct)marshalResult;
        }
        else
        {
          throw new InvalidOperationException("Failed to marshal data from the provided pointer.");
        }

        return result;
      }
      public void SampleTopK(LlamaTokenDataArray candidates, Int32 k, Int32 minKeep)
      {
        if (_ctx == IntPtr.Zero)
        {
          throw new ArgumentException("Failed to create llama_context.");
        }

        IntPtr ptr = Marshal.AllocHGlobal(Marshal.SizeOf(typeof(LlamaTokenDataArrayStruct)));
        Marshal.StructureToPtr(candidates.Candidates, ptr, false);
        LlamaCpp.llama_sample_top_k(_ctx, ptr, k, (IntPtr)minKeep);
        object? result = Marshal.PtrToStructure(ptr, typeof(LlamaTokenDataArrayStruct));

        if (result != null)
        {
          candidates.Candidates = (LlamaTokenDataArrayStruct)result;
        }
        else
        {
          throw new InvalidOperationException("Failed to marshal data from the provided pointer.");
        }
      }
      public void SampleTailFree(LlamaTokenDataArray candidates, float z, Int32 minKeep)
      {
        if (_ctx == IntPtr.Zero)
        {
          throw new ArgumentException("Failed to create llama_context.");
        }

        IntPtr ptr = Marshal.AllocHGlobal(Marshal.SizeOf(typeof(LlamaTokenDataArrayStruct)));
        Marshal.StructureToPtr(candidates.Candidates, ptr, false);
        LlamaCpp.llama_sample_tail_free(_ctx, ptr, z, (IntPtr)minKeep);
        object? result = Marshal.PtrToStructure(ptr, typeof(LlamaTokenDataArrayStruct));

        if (result != null)
        {
          candidates.Candidates = (LlamaTokenDataArrayStruct)result;
        }
        else
        {
          throw new InvalidOperationException("Failed to marshal data from the provided pointer.");
        }
      }
      public void SampleTypical(LlamaTokenDataArray candidates, float p, Int32 minKeep)
      {
        if (_ctx == IntPtr.Zero)
        {
          throw new ArgumentException("Failed to create llama_context.");
        }

        IntPtr ptr = Marshal.AllocHGlobal(Marshal.SizeOf(typeof(LlamaTokenDataArrayStruct)));
        Marshal.StructureToPtr(candidates.Candidates, ptr, false);
        LlamaCpp.llama_sample_typical(_ctx, ptr, p, (IntPtr)minKeep);
        object? result = Marshal.PtrToStructure(ptr, typeof(LlamaTokenDataArrayStruct));

        if (result != null)
        {
          candidates.Candidates = (LlamaTokenDataArrayStruct)result;
        }
        else
        {
          throw new InvalidOperationException("Failed to marshal data from the provided pointer.");
        }
      }
      public void SampleTopP(LlamaTokenDataArray candidates, float p, Int32 minKeep)
      {
        if (_ctx == IntPtr.Zero)
        {
          throw new ArgumentException("Failed to create llama_context.");
        }

        IntPtr ptr = Marshal.AllocHGlobal(Marshal.SizeOf(typeof(LlamaTokenDataArrayStruct)));
        Marshal.StructureToPtr(candidates.Candidates, ptr, false);
        LlamaCpp.llama_sample_top_p(_ctx, ptr, p, (IntPtr)minKeep);
        object? result = Marshal.PtrToStructure(ptr, typeof(LlamaTokenDataArrayStruct));

        if (result != null)
        {
          candidates.Candidates = (LlamaTokenDataArrayStruct)result;
        }
        else
        {
          throw new InvalidOperationException("Failed to marshal data from the provided pointer.");
        }
      }
      public void SampleMinP(LlamaTokenDataArray candidates, float p, Int32 minKeep)
      {
        if (_ctx == IntPtr.Zero)
        {
          throw new ArgumentException("Failed to create llama_context.");
        }

        IntPtr ptr = Marshal.AllocHGlobal(Marshal.SizeOf(typeof(LlamaTokenDataArrayStruct)));
        Marshal.StructureToPtr(candidates.Candidates, ptr, false);
        LlamaCpp.llama_sample_min_p(_ctx, ptr, p, (IntPtr)minKeep);
        object? result = Marshal.PtrToStructure(ptr, typeof(LlamaTokenDataArrayStruct));

        if (result != null)
        {
          candidates.Candidates = (LlamaTokenDataArrayStruct)result;
        }
        else
        {
          throw new InvalidOperationException("Failed to marshal data from the provided pointer.");
        }
      }
      public Int32 SampleToken(LlamaTokenDataArray candidates)
      {
        if (_ctx == IntPtr.Zero)
        {
          throw new ArgumentException("Failed to create llama_context.");
        }

        IntPtr ptr = Marshal.AllocHGlobal(Marshal.SizeOf(typeof(LlamaTokenDataArrayStruct)));
        Marshal.StructureToPtr(candidates.Candidates, ptr, false);
        Int32 result = LlamaCpp.llama_sample_token(_ctx, ptr);
        object? marshalResult = Marshal.PtrToStructure(ptr, typeof(LlamaTokenDataArrayStruct));

        if (marshalResult != null)
        {
          candidates.Candidates = (LlamaTokenDataArrayStruct)marshalResult;
        }
        else
        {
          throw new InvalidOperationException("Failed to marshal data from the provided pointer.");
        }

        return result;
      }
      public Int32 SampleTokenGreedy(LlamaTokenDataArray candidates)
      {
        if (_ctx == IntPtr.Zero)
        {
          throw new ArgumentException("Failed to create llama_context.");
        }

        IntPtr ptr = Marshal.AllocHGlobal(Marshal.SizeOf(typeof(LlamaTokenDataArrayStruct)));
        Marshal.StructureToPtr(candidates.Candidates, ptr, false);
        Int32 result = LlamaCpp.llama_sample_token_greedy(_ctx, ptr);
        object? marshalResult = Marshal.PtrToStructure(ptr, typeof(LlamaTokenDataArrayStruct));

        if (marshalResult != null)
        {
          candidates.Candidates = (LlamaTokenDataArrayStruct)marshalResult;
        }
        else
        {
          throw new InvalidOperationException("Failed to marshal data from the provided pointer.");
        }

        return result;
      }
      public void SetRngSeed(UInt32 seed)
      {
        if (_ctx == IntPtr.Zero)
        {
          throw new ArgumentException("Failed to create llama_context.");
        }

        LlamaCpp.llama_set_rng_seed(_ctx, seed);
      }
      public IntPtr GetLogits()
      {
        if (_ctx == IntPtr.Zero)
        {
          throw new ArgumentException("Failed to create llama_context.");
        }

        return LlamaCpp.llama_get_logits(_ctx);
      }
      public UInt32 NCtx()
      {
        if (_ctx == IntPtr.Zero)
        {
          throw new ArgumentException("Failed to create llama_context.");
        }

        return LlamaCpp.llama_n_ctx(_ctx);
      }
      public void Decode(LlamaBatch batch)
      {
        // if (batch.GetBatch == null)
        // {
        // 	throw new ArgumentException("Failed to create LlamaBatchStruct.");
        // }
        if (_ctx == IntPtr.Zero)
        {
          throw new ArgumentException("Failed to create llama_context.");
        }

        Int32 returnCode = LlamaCpp.llama_decode(_ctx, batch.GetBatch);

        if (returnCode != 0)
        {
          throw new ApplicationException($"llama_decode returned {returnCode}");
        }

      }

      ~LlamaContext()
      {
        // Destructor: Release resources
        if (_ctx != IntPtr.Zero)
        {
          LlamaCpp.llama_free(_ctx);
        }
      }
      public LlamaModel Model
      {
        get {return _model;}
      }
      // LlamaContext Getters
      public IntPtr GetCtxIntPtr
      {
        get
        {
          return _ctx;
        }
      }

      public void KvCacheSeqRm(Int32 seqId, Int32 q0, Int32 q1)
      {
        if (_ctx == IntPtr.Zero)
        {
          throw new InvalidOperationException("ctx is pointing to null.");
        }

        LlamaCpp.llama_kv_cache_seq_rm(_ctx, seqId, q0, q1);
      }

      public IntPtr GetLogitsIth(Int32 i)
      {
        if (_ctx == IntPtr.Zero)
        {
          throw new InvalidOperationException("ctx is pointing to null.");
        }

        return LlamaCpp.llama_get_logits_ith(_ctx, i);
      }
    }

    public class LlamaBatch
    {
      private IntPtr _llamaBatchFree = IntPtr.Zero;
      private LlamaBatchStruct _batch;
      private Int32 _nTokens;
      private Int32 _embd;
      private UInt32 _nSeqMax;

      public LlamaBatch(Int32 nTokens, Int32 embd, UInt32 nSeqMax, bool verbose = true)
      {
        _nTokens = nTokens;
        _embd = embd;
        _nSeqMax = nSeqMax;
        _batch = LlamaCpp.llama_batch_init(nTokens, embd, (Int32)nSeqMax);
      }

      ~LlamaBatch()
      {
        // Destructor: Release resources
        LlamaCpp.llama_batch_free(_batch);
      }
      public void SetBatch(List<Int32> batch, Int32 nPast, bool logitAll)
      {
        byte logitAllByte = logitAll ? (byte)1 : (byte)0;
        Int32 nTokens = batch.Count;
        _batch.NTokens = nTokens;

        for (int i = 0; i < nTokens; i++)
        {
          Pointer.WriteValue(_batch.Token, i, batch[i], typeof(Int32));
          Pointer.WriteValue(_batch.Pos, i, nPast + i, typeof(Int32));
          IntPtr seqIdRowPtr = (IntPtr)Pointer.ReadValue(_batch.SeqId, i, typeof(IntPtr));
          Pointer.WriteValue(seqIdRowPtr, 0, 0, typeof(Int32));
          Pointer.WriteValue(_batch.NSeqId, i, 1, typeof(Int32));
          Pointer.WriteValue(_batch.Logits, i, logitAllByte, typeof(byte));
        }

        Pointer.WriteValue(_batch.Logits, nTokens - 1, (byte)1, typeof(byte));
      }
      public LlamaBatchStruct GetBatch
      {
        get
        {
          return _batch;
        }
      }
    }
    public class LlamaSamplingParams
    {
      public Int32 nPrev;
      public Int32 nProbs;
      public Int32 topK;
      public float topP;
      public float minP;
      public float tfsZ;
      public float typicalP;
      public float temp;
      public Int32 penaltyLastN;
      public float penaltyRepeat;
      public float penaltyFreq;
      public float penaltyPresent;
      public Int32 mirostat;
      public float mirostatTau;
      public float mirostatEta;
      public bool penalizeNl;
      public string grammar;
      public string cfgNegativePrompt;
      public float cfgScale;
      public Dictionary<Int32, float> logitBias;

      public LlamaSamplingParams(
        Int32 nPrev = 64,
        Int32 nProbs = 0,
        Int32 topK = 40,
        float topP = 0.95f,
        float minP = 0.05f,
        float tfsZ = 1.00f,
        float typicalP = 1.00f,
        float temp = 0.80f,
        Int32 penaltyLastN = 64,
        float penaltyRepeat = 1.10f,
        float penaltyFreq = 0.00f,
        float penaltyPresent = 0.00f,
        Int32 mirostat = 0,
        float mirostatTau = 5.00f,
        float mirostatEta = 0.10f,
        bool penalizeNl = true,
        string grammar = "",
        string cfgNegativePrompt = "",
        float cfgScale = 1.00f,
        Dictionary<Int32, float>? logitBias = null
      )
      {
        // this refers to a member variable of the structure
        this.nPrev = nPrev;
        this.nProbs = nProbs;
        this.topK = topK;
        this.topP = topP;
        this.minP = minP;
        this.tfsZ = tfsZ;
        this.typicalP = typicalP;
        this.temp = temp;
        this.penaltyLastN = penaltyLastN;
        this.penaltyRepeat = penaltyRepeat;
        this.penaltyFreq = penaltyFreq;
        this.penaltyPresent = penaltyPresent;
        this.mirostat = mirostat;
        this.mirostatTau = mirostatTau;
        this.mirostatEta = mirostatEta;
        this.penalizeNl = penalizeNl;
        this.grammar = grammar;
        this.cfgNegativePrompt = cfgNegativePrompt;
        this.cfgScale = cfgScale;
        this.logitBias = logitBias ?? new Dictionary<Int32, float>();
      }
    }
    public class LlamaTokenDataArray
    {
      private Int32 _nVocab;
      public LlamaTokenData[] _candidatesData;
      private LlamaTokenDataArrayStruct _candidates;
      private Int32[] _defaultCandidatesDataId;
      private float[] _defaultCandidatesDataP;

      public LlamaTokenDataArray(Int32 nVocab)
      {
        _nVocab = nVocab;
        _candidatesData = new LlamaTokenData[_nVocab];
        _candidates = new LlamaTokenDataArrayStruct();
        _candidates.Data = Marshal.UnsafeAddrOfPinnedArrayElement(_candidatesData, 0);
        _candidates.Size =  (UIntPtr)_nVocab;
        _candidates.Sorted = false;
        _defaultCandidatesDataId = Enumerable.Range(0, _nVocab).ToArray();
        _defaultCandidatesDataP = new float[_nVocab];
      }
      public void CopyLogits(float[] logits)
      {
        for (Int32 i = 0; i < _candidatesData.Length; i++)
        {
          _candidatesData[i].Id = _defaultCandidatesDataId[i];
          _candidatesData[i].Logit = logits[i];
          _candidatesData[i].P = _defaultCandidatesDataP[i];
        }

        _candidates.Data = Marshal.UnsafeAddrOfPinnedArrayElement(_candidatesData, 0);
        _candidates.Sorted = false;
        _candidates.Size = (UIntPtr)_nVocab;
      }
      public LlamaTokenDataArrayStruct Candidates
      {
        get { return _candidates;}
        set { _candidates = value;}
      }

    }
  }
}
