#nullable enable
#if UNITY_STANDALONE || UNITY_IOS || UNITY_ANDROID
#define UNITY
#endif
using System;
using System.Runtime.InteropServices;
using System.IO;
using System.Text;
using LlamaCppUnity.LlamaCppWrappers;
using LlamaCppUnity.Speculative;
using LlamaCppUnity.Internals;
using LlamaCppUnity.LlamaChatFormat;
using LlamaCppUnity.Tokenizer;
using LlamaCppUnity.Helper;
using LlamaCppUnity.Grammar;
using System.Collections.Generic;
using WebAI;
#if UNITY
using UnityEngine;
#endif

namespace LlamaCppUnity
{
  public class Llama
  {
    private static bool __backendInitialized = false;
    private bool _verbose;
    private GgmlNumaStrategy _numa;
    private string _modelPath;
    private LlamaModelParams _modelParams;
    private List<float>? _tensorSplit;
    private float[] _cTensorSplit;
    private LlamaModelKvOverride[]?  _kvOverridesArray;
    private Dictionary<string, object>? _kvOverrides;
    private UInt32 _nCtx;
    private UInt32 _nBatch;
    private UInt32 _nThreads;
    private UInt32 _nThreadsBatch;
    private LlamaContextParams _contextParams;
    private Int32 _lastNTokensSize;
    private string? _loraBase;
    private float _loraScale;
    private string? _loraPath;
    // private Int32? _cache;
    private LlamaModel _model;
    private LlamaContext _ctx;
    private LlamaBatch _batch;
    private string? _chatFormat;
    private LlamaDraftModel? _draftModel;
    private Int32 _nVocab;
    // private LlamaChatCompletionHandler? chatHandler;
    private Int32? _chatHandler; // Not Impl LlamaChatCompletionHandler
    private Int32 _tokenNl;
    private Int32 _tokenEos;
    private Int32 _nTokens;
    private Int32[] _inputIds;
    private float[,] _scores;
    private float _mirostatMu;
    private LlamaTokenDataArray _candidates;
    private LlamaTokenizer _tokenizer;
    Dictionary<string, string> _metaData;
    public Llama(string modelPath,
                 Int32 nGpuLayers = 0,
                 LlamaSplitMode splitMode = LlamaSplitMode.Layer,
                 Int32 mainGpu = 0,
                 List<float>? tensorSplit = null,
                 bool vocabOnly = false,
                 bool useMmap = true,
                 bool useMlock = false,
                 Dictionary<string, object>? kvOverrides = null,
                 UInt32 seed = 0xFFFFFFFF,
                 UInt32 nCtx = 512,
                 UInt32 nBatch = 512,
                 UInt32? nThreads = null,
                 UInt32? nThreadsBatch = null,
                 LlamaRopeScalingType ropeScalingType = LlamaRopeScalingType.Unspecified,
                 LlamaRopeScalingType poolingType = LlamaRopeScalingType.Unspecified,
                 float ropeFreqBase = 0.0f,
                 float ropeFreqScale = 0.0f,
                 float yarnExtFactor = -1.0f,
                 float yarnAttnFactor = 1.0f,
                 float yarnBetaFast = 32.0f,
                 float yarnBetaSlow = 1.0f,
                 UInt32 yarnOrigCtx = 0,
                 bool logitsAll = false,
                 bool embedding = false,
                 bool offloadKqv = true,
                 Int32 lastNTokensSize = 64,
                 string? loraBase = null,
                 float loraScale = 1.0f,
                 string? loraPath = null,
                 GgmlNumaStrategy numa = GgmlNumaStrategy.Disabled,
                 string? chatFormat = null,
                 Int32? chatHandler = null,
                 LlamaDraftModel? draftModel = null,
                 LlamaTokenizer? tokenizer = null,
                 bool verbose = true,
                 Int32? typeK = null,
                 Int32? typeV = null)
    {
      _cTensorSplit = new float[1];
      _verbose = verbose;

      // Not Impl set_verbose(_verbose)
      // Not Impl suppress_stdout_stderr
      if (!__backendInitialized)
      {
        LlamaCpp.llama_backend_init();
        __backendInitialized = true;
      }

      _numa = numa;

      if (numa != GgmlNumaStrategy.Disabled)
      {
        LlamaCpp.llama_numa_init((Int32)_numa);
      }

      _modelPath = modelPath;

      _modelParams = LlamaCpp.llama_model_default_params();
      _modelParams.NGpuLayers = nGpuLayers == -1 ? 0x7FFFFFFF : nGpuLayers;
      _modelParams.SplitMode = (Int32)splitMode;
      _modelParams.MainGpu = mainGpu;
      _tensorSplit = tensorSplit;

      if (_tensorSplit != null)
      {
        if (_tensorSplit.Count > LlamaCpp.LlamaMaxDevices)
        {
          throw new ArgumentException($"Attempt to split tensors that exceed maximum supported devices. Current LlamaMaxDevices={LlamaCpp.LlamaMaxDevices}");
        }

        _cTensorSplit = new float[LlamaCpp.LlamaMaxDevices];
        _modelParams.TensorSplit = Marshal.UnsafeAddrOfPinnedArrayElement(_cTensorSplit, 0);
      }

      _modelParams.VocabOnly = vocabOnly;
      _modelParams.UseMmap = loraPath == null ? useMmap : false;
      _modelParams.UseMlock = useMlock;
      _kvOverrides = kvOverrides;
      Int32 i = 0;

      if (kvOverrides != null)
      {
        Int32 kvoArrayLen = kvOverrides.Count + 1;
        _kvOverridesArray = new LlamaModelKvOverride[kvoArrayLen];

        foreach (var kvp in kvOverrides)
        {
          string key = kvp.Key;
          object value = kvp.Value;
          _kvOverridesArray[i].Key = key;

          if (value is bool boolValue)
          {
            _kvOverridesArray[i].Tag = (Int32)LlamaKvOverrideType.Bool;
            _kvOverridesArray[i].Value.BoolValue = boolValue;
          }
          else if (value is int intValue)
          {
            _kvOverridesArray[i].Tag = (Int32)LlamaKvOverrideType.Int;
            _kvOverridesArray[i].Value.IntValue = intValue;
          }
          else if (value is float floatValue)
          {
            _kvOverridesArray[i].Tag = (Int32)LlamaKvOverrideType.Float;
            _kvOverridesArray[i].Value.FloatValue = floatValue;
          }
          else
          {
            throw new ArgumentException($"Unknown value type for {key}: {value}");
          }

          i++;
        }

        _kvOverridesArray[_kvOverridesArray.Length - 1].Key = "\0";
        _modelParams.KvOverrides = Marshal.UnsafeAddrOfPinnedArrayElement(_kvOverridesArray, 0);
      }

      _nBatch = (UInt32)Math.Min(nCtx, nBatch);
      _nThreads = nThreads ?? (UInt32)Math.Max(Environment.ProcessorCount / 2, 1);
      _nThreadsBatch = nThreadsBatch ?? (UInt32)Math.Max(Environment.ProcessorCount / 2, 1);

      _contextParams = LlamaCpp.llama_context_default_params();
      _contextParams.Seed = seed;
      _contextParams.NCtx = nCtx;
      _contextParams.NBatch = _nBatch;
      _contextParams.NThreads = _nThreads;
      _contextParams.NThreadsBatch = _nThreadsBatch;
      _contextParams.RopeScalingType = ropeScalingType;
      _contextParams.PoolingType = poolingType;
      _contextParams.RopeFreqBase = ropeFreqBase != 0.0 ? ropeFreqBase : 0;
      _contextParams.RopeFreqScale = ropeFreqScale != 0.0 ? ropeFreqScale : 0;
      _contextParams.YarnExtFactor = yarnExtFactor != 0.0 ? yarnExtFactor : 0;
      _contextParams.YarnAttnFactor = yarnAttnFactor != 0.0 ? yarnAttnFactor : 0;
      _contextParams.YarnBetaFast = yarnBetaFast != 0.0 ? yarnBetaFast : 0;
      _contextParams.YarnBetaSlow = yarnBetaSlow != 0.0 ? yarnBetaSlow : 0;
      _contextParams.YarnOrigCtx = yarnOrigCtx != 0 ? yarnOrigCtx : 0;
      _contextParams.LogitsAll = draftModel == null ? logitsAll : true;

      _contextParams.Embeddings = embedding;
      _contextParams.OffloadKqv = offloadKqv;

      if (typeK != null)
      {
        _contextParams.TypeK = (Int32)typeK;
      }

      if (typeV != null)
      {
        _contextParams.TypeV = (Int32)typeV;
      }

      _lastNTokensSize = lastNTokensSize;

      // _cache = null; //Not Impl BaseLlamaCache

      _loraBase = loraBase;
      _loraScale = loraScale;
      _loraPath = loraPath;

      if (!File.Exists(modelPath))
      {
        throw new ArgumentException($"Model path does not exist: {modelPath}");
      }

      _model = new LlamaModel(pathModel: _modelPath, parameters: _modelParams, verbose: _verbose);

      _tokenizer = tokenizer ?? new LlamaTokenizer(this);

      if (nCtx == 0)
      {
        nCtx = (UInt32)_model.NCtxTrain();
        _nBatch = Math.Min(nCtx, nBatch);
        _contextParams.NCtx = (UInt32)_model.NCtxTrain();
        _contextParams.NBatch = _nBatch;
      }

      _ctx = new LlamaContext(llamaModel: _model, parameters: _contextParams, verbose: _verbose);

      _batch = new LlamaBatch(nTokens: (Int32)_nBatch, embd: 0, nSeqMax: _contextParams.NCtx, verbose: _verbose);

      if (_loraPath != null)
      {
        if (_model.ApplyLoraFromFile(_loraPath, _loraScale, _loraBase, (Int32)_nThreads) == 1)
        {
          throw new Exception($"Failed to apply LoRA from lora path: {_loraPath} to base path: {_loraBase}");
        }
      }

      if (_verbose)
      {
        #if UNITY
        Debug.Log(CCharUtils.IntPtrToStringUtf8(LlamaCpp.llama_print_system_info()));
        #else
        Console.Error.WriteLine(Marshal.PtrToStringAuto(LlamaCpp.llama_print_system_info()));
        #endif
      }

      _chatFormat = chatFormat;
      _chatHandler = chatHandler;

      _draftModel = draftModel;
      _nVocab = NVocab();
      _nCtx = NCtx();
      _tokenNl = TokenNl();
      _tokenEos = TokenEos();

      _candidates = new LlamaTokenDataArray(nVocab: _nVocab);

      _nTokens = 0;
      _inputIds = new Int32[_nCtx];
      _scores = new float[_nCtx, _nVocab];
      _mirostatMu = 2.0f * 5.0f;

      try
      {
        _metaData = _model.MetaData();
      }
      catch (Exception e)
      {
        _metaData = new Dictionary<string, string>();

        if (_verbose)
        {
          #if UNITY
          Debug.Log($"Failed to load metadata: {e.Message}");
          #else
          Console.Error.WriteLine($"Failed to load metadata: {e.Message}");
          #endif
        }
      }

      if (_verbose)
      {
        #if UNITY
        StringBuilder logStr = new StringBuilder("Model metadata:\n");

        foreach (var pair in _metaData)
        {
          logStr.AppendFormat("Key: {0}, Value: {1}\n", pair.Key, pair.Value);
        }

        Debug.Log(logStr.ToString());

        #else
        Console.Error.WriteLine($"Model metadata:");

        foreach (var pair in _metaData)
        {
          Console.WriteLine("Key: " + pair.Key + ", Value: " + pair.Value);
        }

        #endif
      }

      if (_chatFormat == null &&
          _chatHandler == null &&
          _metaData.ContainsKey("tokenizer.chat_template"))
      {
        chatFormat = LlamaChatFormatHelper.GuessChatFormatFromGgufMetaData(_metaData);

        if (chatFormat != null)
        {
          _chatFormat = chatFormat;

          if (_verbose)
          {
            #if UNITY
            Debug.Log($"Guessed chat format: {chatFormat}");
            #else
            Console.Error.WriteLine($"Guessed chat format: {chatFormat}");
            #endif
          }
        }
        else
        {
          string template = _metaData["tokenizer.chat_template"];
          Int32 eosTokenId, bosTokenId;

          try
          {
            eosTokenId = Int32.Parse(_metaData["tokenizer.ggml.eos_token_id"]);
          }
          catch (Exception)
          {
            eosTokenId = TokenEos();
          }

          try
          {
            bosTokenId = Int32.Parse(_metaData["tokenizer.ggml.bos_token_id"]);
          }
          catch (Exception)
          {
            bosTokenId = TokenBos();
          }

          string eosToken = _model.TokenGetText(eosTokenId);
          string bosToken = _model.TokenGetText(bosTokenId);

          if (_verbose)
          {
            #if UNITY
            Debug.Log($"Using gguf chat template: {template}");
            Debug.Log($"Using chat eos_token: {eosToken}");
            Debug.Log($"Using chat bos_token: {bosToken}");
            #else
            Console.Error.WriteLine($"Using gguf chat template: {template}");
            Console.Error.WriteLine($"Using chat eos_token: {eosToken}");
            Console.Error.WriteLine($"Using chat bos_token: {bosToken}");
            #endif
          }

          // Not Impl chat_handler
        }
      }

      if (_chatFormat == null && _chatHandler == null)
      {
        _chatFormat = "llama-2";

        if (_verbose)
        {
          #if UNITY
          Debug.Log($"Using fallback chat format: {chatFormat}");
          #else
          Console.Error.WriteLine($"Using fallback chat format: {chatFormat}");
          #endif
        }
      }
    }
    public string Run(string prompt,
                      string? suffix = null,
                      UInt32 maxTokens = 16,
                      float temperature = 0.8f,
                      float topP = 0.95f,
                      float minP = 0.05f,
                      float typicalP = 1.0f,
                      Int32? logProbs = null,
                      bool echo = false,
                      List<string>? stop = null,
                      float frequencyPenalty = 0.0f,
                      float presencePenalty = 0.0f,
                      float repeatPenalty = 1.1f,
                      Int32 topK = 40,
                      Int32? seed = null,
                      float tfsZ = 1.0f,
                      Int32 mirostatMode = 0,
                      float mirostatTau = 5.0f,
                      float mirostatEta = 0.1f,
                      string? model = null,
                      LlamaGrammar? grammar = null)
    {
      IEnumerator<string> enumerator =  _CreateCompletion(prompt: prompt, suffix: suffix, maxTokens: maxTokens, temperature: temperature, topP: topP, minP: minP, typicalP: typicalP, logProbs: logProbs, echo: echo, stop: stop, frequencyPenalty: frequencyPenalty, presencePenalty: presencePenalty, repeatPenalty: repeatPenalty, topK: topK, stream: false, seed: seed, tfsZ: tfsZ, mirostatMode: mirostatMode, mirostatTau: mirostatTau, mirostatEta: mirostatEta, model: model, grammar: grammar).GetEnumerator();
      enumerator.MoveNext();
      return enumerator.Current;
    }
    public IEnumerable<string> RunStream(string prompt,
                                         string? suffix = null,
                                         UInt32 maxTokens = 16,
                                         float temperature = 0.8f,
                                         float topP = 0.95f,
                                         float minP = 0.05f,
                                         float typicalP = 1.0f,
                                         Int32? logProbs = null,
                                         bool echo = false,
                                         List<string>? stop = null,
                                         float frequencyPenalty = 0.0f,
                                         float presencePenalty = 0.0f,
                                         float repeatPenalty = 1.1f,
                                         Int32 topK = 40,
                                         Int32? seed = null,
                                         float tfsZ = 1.0f,
                                         Int32 mirostatMode = 0,
                                         float mirostatTau = 5.0f,
                                         float mirostatEta = 0.1f,
                                         string? model = null,
                                         LlamaGrammar? grammar = null)
    {
      foreach (string text in _CreateCompletion(prompt: prompt, suffix: suffix, maxTokens: maxTokens, temperature: temperature, topP: topP, minP: minP, typicalP: typicalP, logProbs: logProbs, echo: echo, stop: stop, frequencyPenalty: frequencyPenalty, presencePenalty: presencePenalty, repeatPenalty: repeatPenalty, topK: topK, stream: true, seed: seed, tfsZ: tfsZ, mirostatMode: mirostatMode, mirostatTau: mirostatTau, mirostatEta: mirostatEta, model: model, grammar: grammar))
      {
        yield return text;
      }
    }

    public IEnumerable<string> _CreateCompletion(string prompt,
        string? suffix = null,
        UInt32 maxTokens = 16,
        float temperature = 0.8f,
        float topP = 0.95f,
        float minP = 0.05f,
        float typicalP = 1.0f,
        Int32? logProbs = null,
        bool echo = false,
        List<string>? stop = null,
        float frequencyPenalty = 0.0f,
        float presencePenalty = 0.0f,
        float repeatPenalty = 1.1f,
        Int32 topK = 40,
        bool stream = false,
        Int32? seed = null,
        float tfsZ = 1.0f,
        Int32 mirostatMode = 0,
        float mirostatTau = 5.0f,
        float mirostatEta = 0.1f,
        string? model = null,
        LlamaGrammar? grammar = null
                                                )
    {
      string completionId = $"cmpl-{Guid.NewGuid()}";
      Int32 created = (int)(DateTimeOffset.UtcNow - new DateTimeOffset(1970, 1, 1, 0, 0, 0, TimeSpan.Zero)).TotalSeconds;
      List<Int32> completionTokens;

      if (prompt.Length > 0)
      {
        completionTokens = new List<Int32>();
      }
      else
      {
        completionTokens = new List<Int32> { TokenBos() };
      }

      List<Int32> promptTokens;

      if (prompt != "")
      {
        promptTokens = new List<Int32>(Tokenize(Encoding.UTF8.GetBytes(prompt), special: true));
      }
      else
      {
        promptTokens = new List<Int32> { TokenBos() };
      }

      // Int32 returnedTokens = 0;


      string modeName = model != null ? model : _modelPath ;


      //Not Impl logit_bias_processor
      // if(logitBias != null){
      // 	  Dictionary<int, float> logitBiasMap = new Dictionary<int, float>();
      // 	  foreach (var kvp in logitBias)
      // 	  {
      // 	      int key = Convert.ToInt32(kvp.Key);
      // 	      float value = Convert.ToFloat(kvp.Value);
      // 	      logitBiasMap[key] = value;
      // 	  }
      // }

      if (_verbose)
      {
        _ctx.ResetTimings();
      }

      if (promptTokens.Count >= _nCtx)
      {
        throw new ArgumentException($"Requested tokens ({promptTokens.Count}) exceed context window of {LlamaCpp.llama_n_ctx(_ctx.GetCtxIntPtr)}");
      }

      if (maxTokens <= 0)
      {
        maxTokens = _nCtx -  (UInt32)promptTokens.Count;
      }

      maxTokens = maxTokens + (UInt32)promptTokens.Count < _nCtx ? maxTokens : (_nCtx - (UInt32)promptTokens.Count);
      List<byte[]> stopSequences = new List<byte[]>();
      // List<List<byte>> stopSequences = new List<List<byte>>();

      if (stop != null)
      {

        foreach (var s in stop)
        {
          stopSequences.Add(Encoding.UTF8.GetBytes(s));
        }
      }

      if (logProbs != null && _contextParams.LogitsAll == false)
      {
        throw new InvalidOperationException("logProbs is not supported for models created with LogitsAll=False");
      }

      // Not Impl cache
      // if(_cache!=null){
      // 	  try{
      // 	      cacheItem = _cache[promptTokens];
      // 	      cachePrefixLen = Llama.
      // 	  }
      // }

      if (seed != null)
      {
        _ctx.SetRngSeed((UInt32)seed);
      }

      // string finishReason = "length";
      Int32 multibyteFix = 0;
      byte[] text = new byte[1];
      byte[] allText = new byte[1];
      byte[] remainingText = new byte[10];
      Int32 remainingLength = 0;

      foreach (Int32 token in Generate(tokens: promptTokens,
                                       topK: topK,
                                       topP: topP,
                                       minP: minP,
                                       typicalP: typicalP,
                                       temp: temperature,
                                       tfsZ: tfsZ,
                                       mirostatMode: mirostatMode,
                                       mirostatTau: mirostatTau,
                                       mirostatEta: mirostatEta,
                                       grammar: grammar
                                      ))
      {
        if (token == _tokenEos)
        {
          text = Detokenize(completionTokens, prevTokens: promptTokens);
          // finishReason = "stop";
          break;
        }

        completionTokens.Add(token);
        allText = Detokenize(completionTokens, prevTokens: promptTokens);
        // Int32 multibyteFix;

        if (allText.Length >= 3)
        {
          for (int k = 3; k > 0; k--)
          {
            byte chr = allText[allText.Length - k];

            foreach (var (num, pattern) in new (int, int)[] { (2, 192), (3, 224), (4, 240) })
            {
              if (num > k && (pattern & chr) == pattern)
              {
                multibyteFix = 2 - k;
              }
            }
          }
        }

        if (multibyteFix > 0)
        {
          multibyteFix -= 1;
          continue;
        }


        List<byte[]> anyStop = new List<byte[]>();

        foreach (byte[] s in stopSequences)
        {
          if (Encoding.UTF8.GetString(allText).Contains(Encoding.UTF8.GetString(s)))
          {
            anyStop.Add(s);
          }
        }



        if (anyStop.Count > 0)
        {
          byte[] firstStop = anyStop[0];
          Int32 rangeFirstStop = Encoding.UTF8.GetString(allText).IndexOf(Encoding.UTF8.GetString(firstStop));
          text = new byte[rangeFirstStop];

          for (int k = 0; k < rangeFirstStop; k++)
          {
            text[k] = allText[k];
          }

          // finishReason = "stop";
          break;
        }

        // Not Impl Stream
        if (completionTokens.Count >= maxTokens)
        {
          text = Detokenize(completionTokens, prevTokens: promptTokens);
          // finishReason = "length";
          break;
        }

        if (stream)
        {
          for (int i = 0; i < remainingText.Length; i++)
          {
            remainingText[i] = 0;
          }

          remainingText = new byte[allText.Length - remainingLength];

          for (int i = 0; i < allText.Length - remainingLength; i++)
          {
            remainingText[i] = allText[remainingLength + i];
          }

          string result = CCharUtils.CCharToString(remainingText);

          if (!result.Contains("\uFFFD"))
          {
            remainingLength = allText.Length;
            yield return result;
          }
        }

        // break;
      }

      if (stream)
      {
        for (int i = 0; i < remainingText.Length; i++)
        {
          remainingText[i] = 0;
        }

        remainingText = new byte[allText.Length - remainingLength];

        for (int i = 0; i < allText.Length - remainingLength; i++)
        {
          remainingText[i] = allText[remainingLength + i];
        }

        string result = CCharUtils.CCharToString(remainingText);

        if (!result.Contains("\uFFFD"))
        {
          remainingLength = allText.Length;
          yield return result;
        }
      }
      else
      {
        yield return CCharUtils.CCharToString(text);
      }
    }
    public byte[] Detokenize(List<Int32> tokens, List<Int32>? prevTokens = null)
    {
      return _tokenizer.Detokenize(tokens, prevTokens: prevTokens);
    }
    public Int32 Sample(
      Int32 topK = 40,
      float topP = 0.95f,
      float minP = 0.05f,
      float typicalP = 1.0f,
      float temp = 0.80f,
      float repeatPenalty = 1.1f,
      bool reset = true,
      float frequencyPenalty = 0.0f,
      float presencePenalty = 0.0f,
      float tfsZ = 1.0f,
      Int32 mirostatMode = 0,
      float mirostatTau = 5.0f,
      float mirostatEta = 0.1f,
      bool penalizeNl = true,
      LlamaGrammar? grammar = null,
      Int32? idx = null)
    {
      Int32 cols = _scores.GetLength(1);
      float[] logits = new float[cols];

      if (idx == null)
      {
        Int32 lastRowIndex = _scores.GetLength(0) - 1;

        for (Int32 i = 0; i < cols; i++)
        {
          logits[i] = _scores[lastRowIndex, i];
        }
      }
      else
      {

        for (Int32 i = 0; i < cols; i++)
        {
          logits[i] = _scores[(Int32)idx, i];
        }
      }

      // Not Impl logitprocessor
      LlamaSamplingParams samplingParams = new LlamaSamplingParams(topK: topK,
          topP: topP,
          minP: minP,
          tfsZ: tfsZ,
          typicalP: typicalP,
          temp: temp,
          penaltyLastN: _lastNTokensSize,
          penaltyRepeat: repeatPenalty,
          penaltyFreq: frequencyPenalty,
          penaltyPresent: presencePenalty,
          mirostat: mirostatMode,
          mirostatTau: mirostatTau,
          mirostatEta: mirostatEta,
          penalizeNl: penalizeNl);
      LlamaSamplingContext samplingContext = new LlamaSamplingContext(samplingParams, grammar);
      samplingContext.Prev = EvalTokens();
      Int32 id = samplingContext.Sample(_ctx, logitsArray: logits);


      samplingContext.Accept(ctxMain: _ctx, id: id, applyGrammar: grammar != null);
      return id;

    }

    public IEnumerable<Int32> Generate(
      List<Int32> tokens,
      Int32 topK = 40,
      float topP = 0.95f,
      float minP = 0.05f,
      float typicalP = 1.0f,
      float temp = 0.80f,
      float repeatPenalty = 1.1f,
      bool reset = true,
      float frequencyPenalty = 0.0f,
      float presencePenalty = 0.0f,
      float tfsZ = 1.0f,
      Int32 mirostatMode = 0,
      float mirostatTau = 5.0f,
      float mirostatEta = 0.1f,
      bool penalizeNl = true,
      LlamaGrammar? grammar = null
    )
    {
      _mirostatMu = 2 * mirostatTau;

      if (reset && _nTokens > 0)
      {
        Int32 longestPrefix = 0;

        for (int i = 0; i < Math.Min(_inputIds.Length, tokens.Count - 1); i++)
        {
          if (_inputIds[i] == tokens[i])
          {
            longestPrefix += 1;
          }
          else
          {
            break;
          }
        }

        if (longestPrefix > 0)
        {
          if (_verbose)
          {
            #if UNITY
            Debug.Log($"Llama.generate: prefix-match hit");
            #else
            Console.Error.WriteLine($"Llama.generate: prefix-match hit");
            #endif
          }

          reset = false;
          tokens = tokens.GetRange(longestPrefix, tokens.Count - longestPrefix);
          _nTokens = longestPrefix;
        }
      }

      if (reset)
      {
        Reset();
      }

      if (grammar != null)
      {
        grammar.Reset();
      }

      Int32 sampleIdx = _nTokens + tokens.Count - 1;

      while (true)
      {
        Eval(tokens);

        while (sampleIdx < _nTokens)
        {
          Int32 token = Sample(
                          topK: topK,
                          topP: topP,
                          minP: minP,
                          typicalP: typicalP,
                          temp: temp,
                          repeatPenalty: repeatPenalty,
                          frequencyPenalty: frequencyPenalty,
                          presencePenalty: presencePenalty,
                          tfsZ: tfsZ,
                          mirostatMode: mirostatMode,
                          mirostatTau: mirostatTau,
                          mirostatEta: mirostatEta,
                          penalizeNl: penalizeNl,
                          grammar: grammar,
                          idx: sampleIdx
                        );
          sampleIdx += 1;
          // if (stoppingCriteria != null && stoppingCriteria(_inputIdx, _scores[_socre.GetLength(0)-1])){
          // 	  return;
          // }
          // tokensOrNone = yield token;
          // tokens.clear();
          // tokens.append(token);
          yield return token;

          tokens.Clear();
          tokens.Add(token);

          if (sampleIdx < _nTokens && token != _inputIds[sampleIdx])
          {
            _nTokens = sampleIdx;
            _ctx.KvCacheSeqRm(-1, _nTokens, -1);
            break;
          }
        }

        // Not Impl drafModel
        // if(_draftModel != null){
        //     for(int i=0;i<tokens.Count;i++){
        // 	_inputIds[_nTokens + i] = tokens[i];
        //     }
        //     draftTokens = _draftModel(....
        // }
        //break;
      }

      // return 0;
    }
    public List<int> EvalTokens()
    {
      List<Int32> tokens = new List<Int32>();
      Int32 count = Math.Min(_nTokens, _inputIds.Length);

      for (int i = 0; i < count; i++)
      {
        tokens.Add(_inputIds[i]);

        if (tokens.Count > _nCtx) // サイズが _n_ctx を超えた場合
        {
          tokens.RemoveAt(0);  // 最も古い要素を削除
        }
      }

      return tokens;
    }
    public void Reset()
    {
      _nTokens = 0;
    }
    public List<Int32> Tokenize(byte[] text, bool addBos = true, bool special = false)
    {
      return _tokenizer.Tokenize(text, addBos, special);
    }
    public Int32 NVocab()
    {
      return _model.NVocab();
    }
    public UInt32 NCtx()
    {
      return _ctx.NCtx();
    }
    public Int32 TokenNl()
    {
      return _model.TokenNl();
    }
    public Int32 TokenEos()
    {
      return _model.TokenEos();
    }
    public Int32 TokenBos()
    {
      return _model.TokenBos();
    }
    public void Eval(List<Int32> tokens)
    {
      if (_ctx.GetCtxIntPtr == IntPtr.Zero)
      {
        throw new ArgumentException("Failed to create llama_context.");
      }

      _ctx.KvCacheSeqRm(-1, _nTokens, -1);

      for (Int32 k = 0; k < tokens.Count; k += (Int32)_nBatch)
      {
        List<Int32> batch = tokens.GetRange(k, (Int32)Math.Min(tokens.Count - k, _nBatch));
        Int32 nPast = _nTokens;
        Int32 nTokens = batch.Count;
        _batch.SetBatch(batch, nPast, _contextParams.LogitsAll);
        _ctx.Decode(_batch);

        for (int i = 0; i < nTokens; i++)
        {
          _inputIds[nPast + i] = batch[i];
        }


        if (_contextParams.LogitsAll)
        {
          Int32 rows = nTokens;
          Int32 cols = _nVocab;
          float[] logits = new float[rows * cols];
          IntPtr ctxLogits = _ctx.GetLogits();

          for (Int32 i = 0; i < rows * cols; i++)
          {
            logits[i] = (float)Pointer.ReadValue(ctxLogits, i, typeof(float));
          }

          for (Int32 i = 0; i < rows; i++)
          {
            for (Int32 j = 0; j < cols; j++)
            {
              Int32 idx = i * cols + j;
              _scores[nPast + i, j] = logits[idx];
            }
          }
        }
        else
        {
          Int32 rows = 1;
          Int32 cols = _nVocab;
          float[] logits = new float[rows * cols];
          IntPtr ctxLogits = _ctx.GetLogits();

          for (Int32 i = 0; i < rows * cols; i++)
          {
            logits[i] = (float)Pointer.ReadValue(ctxLogits, i, typeof(float));
          }

          for (Int32 i = 0; i < rows; i++)
          {
            for (Int32 j = 0; j < cols; j++)
            {
              Int32 idx = i * cols + j;
              _scores[nPast + nTokens - 1, j] = logits[idx];
            }
          }


        }

        _nTokens += nTokens;
      }


    }
    public LlamaModel GetModel
    {
      get {  return _model;}
    }
  }
  // public class LogitsProcessorList{
  //     public float[] LogitBiasProcessor(Int32[] inputIds, float[] scores)
  //     {
  // 	  float[] newScores = (float[])scores.Clone();

  // 	  foreach (var kvp in logitBiasMap)
  // 	  {
  // 	      int inputId = kvp.Key;
  // 	      float score = kvp.Value;
  // 	      newScores[inputId] = score + scores[inputId];
  // 	  }

  // 	  return newScores;
  //     }
  //     public float[] Run(Int32[] inputIds, float[] scores, List<){

  //     }
  // }
  //     List<LogitsProcessor>

  public class WebAIAPI
  {
    private Client? _webAIAPI;

    public WebAIAPI(string apiKey, string apiUrl)
    {
      _webAIAPI = new Client(apiKey, apiUrl);
    }

    public ResponseWrapper RunAPI(
      string model,
      string prompt,
      float? temperature = 1.0f,
      float? topP = 1.0f,
      float? frequencyPenalty = 0.0f,
      float? presencePenalty = 0.0f,
      Int32? seed = null,
      Int32? maxTokens = null
    )
    {
      ResponseWrapper jsonResponse = _webAIAPI.GenerateResponse(
                                       model: model,
                                       prompt: prompt,
                                       temperature: temperature,
                                       topP: topP,
                                       frequencyPenalty: frequencyPenalty,
                                       presencePenalty: presencePenalty,
                                       seed: seed,
                                       maxTokens: maxTokens
                                     );

      return jsonResponse;
    }
  }
}
