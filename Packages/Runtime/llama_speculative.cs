#nullable enable
using System;
using System.Collections.Generic;
using System.Numerics;
using LlamaCppUnity.LlamaCppWrappers;

namespace LlamaCppUnity
{
  namespace Speculative
  {
    public abstract class LlamaDraftModel
    {
      public abstract int[] Call(int[] inputIds, params KeyValuePair<string, object>[] kwargs);
    }
  }
}
