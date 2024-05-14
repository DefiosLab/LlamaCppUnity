#nullable enable
using System;
using System.Runtime.InteropServices;
using System.IO;
using System.Text;
using LlamaCppUnity.LlamaCppWrappers;
using LlamaCppUnity.Speculative;
using LlamaCppUnity.Internals;
using System.Collections.Generic;

namespace LlamaCppUnity
{
  namespace LlamaChatFormat
  {
    public struct Chatml
    {
      public static string ChatTemplate = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}";
      public static string BosToken =  "<s>";
      public static string EosToken = "<|im_end|>";
    }
    public struct MistralInstruct
    {
      public static string ChatTemplate = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token + ' ' }}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}";
      public static string BosToken = "<s>";
      public static string EosToken = "</s>";
    }
    public struct  MixtralInstruct
    {
      public static string ChatTemplate = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}";

    }
    public class LlamaChatFormatHelper
    {
      public static string? GuessChatFormatFromGgufMetaData(Dictionary<string, string> metadata)
      {

        if (!metadata.ContainsKey("tokenizer.chat_template"))
        {
          return null;
        }

        if (metadata["tokenizer.chat_template"] == Chatml.ChatTemplate)
        {
          return "chatml";
        }

        if (metadata["tokenizer.chat_template"] == MistralInstruct.ChatTemplate ||
            metadata["tokenizer.chat_template"] == MixtralInstruct.ChatTemplate)
        {
          return "mistral-instruct";
        }

        return null;
      }
    }
  }
}
