#nullable enable
using System;
using System.Runtime.InteropServices;
using System.IO;
using System.Text;
using LlamaCppUnity.LlamaCppWrappers;
using LlamaCppUnity.Speculative;
using LlamaCppUnity.Internals;
using LlamaCppUnity.LlamaChatFormat;
using LlamaCppUnity.Tokenizer;
using System.Collections.Generic;

namespace LlamaCppUnity
{
  namespace Helper
  {
    public class Pointer
    {
      public static object ReadValue(IntPtr ptr, Int32 offset, Type type)
      {
        IntPtr targetPtr;

        if (type == typeof(Int32))
        {
          targetPtr = IntPtr.Add(ptr, sizeof(Int32) * offset);
          return Marshal.ReadInt32(targetPtr);
        }
        else if (type == typeof(float))
        {
          targetPtr = IntPtr.Add(ptr, sizeof(float) * offset);
          Int32 intValue = Marshal.ReadInt32(targetPtr);
          return BitConverter.ToSingle(BitConverter.GetBytes(intValue), 0);
        }
        else if (type == typeof(IntPtr))
        {
          targetPtr = IntPtr.Add(ptr, IntPtr.Size * offset);
          return Marshal.ReadIntPtr(targetPtr);
        }
        else if (type == typeof(byte))
        {
          targetPtr = IntPtr.Add(ptr, sizeof(byte) * offset);
          return Marshal.ReadByte(targetPtr);
        }
        else
        {
          throw new NotSupportedException($"Type {type} is not supported.");
        }
      }
      public static void WriteValue(IntPtr ptr, Int32 offset, object value, Type type)
      {
        IntPtr targetPtr;

        if (type == typeof(Int32))
        {
          targetPtr = IntPtr.Add(ptr, sizeof(Int32) * offset);
          Marshal.WriteInt32(targetPtr, (Int32)value);
        }
        else if (type == typeof(float))
        {
          targetPtr = IntPtr.Add(ptr, sizeof(float) * offset);
          Marshal.WriteInt32(targetPtr, BitConverter.ToInt32(BitConverter.GetBytes((float)value), 0));
        }
        else if (type == typeof(byte))
        {
          targetPtr = IntPtr.Add(ptr, sizeof(byte) * offset);
          Marshal.WriteByte(targetPtr, (byte)value);
        }
        else
        {
          throw new NotSupportedException($"Type {type} is not supported.");
        }
      }
    }

    public class CCharUtils
    {
      public static string CCharToString(byte[] bytes)
      {
        return Encoding.UTF8.GetString(bytes).TrimEnd('\0');
      }
      public static string IntPtrToStringUtf8(IntPtr ptr)
      {
        int len = 0;

        while (Marshal.ReadByte(ptr, len) != 0)
        {
          len++;
        }

        byte[] buffer = new byte[len];
        Marshal.Copy(ptr, buffer, 0, buffer.Length);
        return Encoding.UTF8.GetString(buffer);
      }
    }
  }
}
