#if UNITY_STANDALONE || UNITY_IOS || UNITY_ANDROID
#define UNITY
#endif
#if UNITY
using UnityEngine;
#endif
using System;
using System.Text;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Linq;
using LlamaCppUnity;


namespace LlamaCppUnityTest
{
  public class LlamaTest
  {
    public static void GenerateTest()
    {
      Llama test = new Llama("Assets/Models/llama-2-7b-chat.Q2_K.gguf", verbose: true);
      int seed = 20;
      string prompt = "Q: Name the planets in the solar system? A: ";
      string result = test.Run(prompt, seed: seed);
      #if UNITY
      Debug.Log($"{prompt} {result}");
      #else
      Console.WriteLine($"{prompt} {result}");
      #endif

      #if UNITY
      StringBuilder allText = new StringBuilder();

      foreach (string text in test.RunStream(prompt, seed: seed))
      {
        allText.Append(text);
      }

      Debug.Log(allText.ToString());
      #else

      foreach (string text in test.RunStream(prompt, seed: seed))
      {
        Console.Write(text);
      }

      Console.WriteLine("");

      #endif

      if (result != "1. Venus, 2. Mercury, 3. Mars,")
      {
        throw new Exception("The results do not match");
      }
    }
  }
}
