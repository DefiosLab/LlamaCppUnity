#nullable enable
using System;
using System.Runtime.InteropServices;
using System.IO;
using System.Text;
using LlamaCppUnity.LlamaCppWrappers;
using LlamaCppUnity.Speculative;
using LlamaCppUnity.Helper;
using System.Collections.Generic;
using System.Linq;

namespace LlamaCppUnity
{
  namespace Grammar
  {
    public class LlamaGrammar
    {
      LlamaGrammarElement[,] _grammarRules;
      Int32 _nRules;
      Int32 _startRuleIndex;
      IntPtr? _grammar;
      // List<List<LlamaGrammarElement>> _elementLists;
      // List<LlamaGrammarElement[]> _element_arrays;
      public LlamaGrammar(ParseState parsedGrammar)
      {
        _grammarRules = parsedGrammar.CRules();
        _nRules = _grammarRules.GetLength(0);
        _startRuleIndex = parsedGrammar._symbolIds["root"];
        Init();
      }
      ~LlamaGrammar()
      {
        if (_grammar != null)
        {
          LlamaCpp.llama_grammar_free((IntPtr)_grammar);
          _grammar = null;
        }
      }
      public void Reset()
      {
        if (_grammar != null)
        {
          LlamaCpp.llama_grammar_free((IntPtr)_grammar);
        }

        Init();
      }
      public void Init()
      {
        IntPtr[] _Rules = new IntPtr[_nRules];
        int rows = _grammarRules.GetLength(0);
        int cols = _grammarRules.GetLength(1);

        for (int i = 0; i < _nRules; i++)
        {
          _Rules[i] = Marshal.UnsafeAddrOfPinnedArrayElement(_grammarRules, i * cols);
        }

        _grammar = LlamaCpp.llama_grammar_init(_Rules[0], (IntPtr)_nRules, (IntPtr)_startRuleIndex);
      }
      public IntPtr? Grammar
      {
        get {return _grammar;}
      }
    }
    public class ParseState
    {
      public Dictionary<string, Int32> _symbolIds;
      public LlamaGrammarElement[,]_rules;
      public ParseState()
      {
        _symbolIds = new Dictionary<string, Int32>();
        _rules = new LlamaGrammarElement[1, 1];
      }
      public LlamaGrammarElement[,] CRules()
      {
        Int32 rows = _rules.GetLength(0);
        Int32 col = _rules.GetLength(1);
        LlamaGrammarElement[,] ret = new LlamaGrammarElement[rows, col] ;

        for (int i = 0; i < rows; i++)
        {
          for (int j = 0; j < col; j++)
          {
            ret[i, j] = _rules[i, j];
          }
        }

        return ret;
      }
    }
  }
}
