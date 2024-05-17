#nullable enable
#if UNITY_STANDALONE || UNITY_IOS || UNITY_ANDROID
#define UNITY
#endif
using System;
using System.Text;
using System.Collections.Generic;
using System.Runtime.InteropServices;
#if UNITY
using UnityEngine;
#endif
using System.Linq;
using LlamaCppUnity.LlamaCppWrappers;
using LlamaCppUnity.Internals;
using LlamaCppUnityTest;

#if UNITY
public class LlamaSample : MonoBehaviour
{
  void Start()
  {
#else
public class LlamaSample
{
  static void Main()
  {
#endif
    LlamaTest.GenerateTest();
    #if UNITY
    Debug.Log("All tests successful.");
    #else
    Console.WriteLine("All tests successful.");
    #endif
  }
}

