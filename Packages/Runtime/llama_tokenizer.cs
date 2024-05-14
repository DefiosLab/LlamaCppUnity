#nullable enable
using System;
using System.Runtime.InteropServices;
using System.IO;
using System.Text;
using LlamaCppUnity.LlamaCppWrappers;
using LlamaCppUnity.Speculative;
using LlamaCppUnity.Internals;
using LlamaCppUnity.LlamaChatFormat;
using LlamaCppUnity.Helper;
using System.Collections.Generic;

namespace LlamaCppUnity
{
  namespace Tokenizer
  {
    public abstract class BaseLlamaTokenizer
    {
      public abstract List<Int32> Tokenize(byte[] text, bool addBos = true, bool special = true);
      public abstract byte[] Detokenize(List<Int32> tokens, List<Int32>? prevTokens = null);
    }
    public class LlamaTokenizer : BaseLlamaTokenizer
    {
      LlamaModel _model;
      public LlamaTokenizer(Llama llama)
      {
        _model = llama.GetModel;
      }
      public override List<Int32> Tokenize(byte[] text, bool addBos = true, bool special = true)
      {
        return _model.Tokenize(text, addBos, special);
      }
      public override byte[] Detokenize(List<Int32> tokens, List<Int32>? prevTokens = null)
      {
        return _model.Detokenize(tokens);
      }
      public List<Int32> Encode(string text, bool addBos = true, bool special = true)
      {
        return Tokenize(Encoding.UTF8.GetBytes(text), addBos, special);
      }
      public string Decode(List<Int32> tokens)
      {
        return CCharUtils.CCharToString(Detokenize(tokens));
      }
    }
  }
}
