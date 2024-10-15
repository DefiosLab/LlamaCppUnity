#nullable enable
#if UNITY_STANDALONE || UNITY_IOS || UNITY_ANDROID
#define UNITY
#endif
#if UNITY
using UnityEngine;
#else
using System.Text.Json;
#endif
using System;
using System.Net;
using System.Net.Http;
using System.Collections.Generic;

namespace WebAI
{
  // TODO: Unityと.netでjson構造のやつも分ける
  #if UNITY
  [Serializable]
  public class requestBody
  {
    public string? model;
    public Message[]? messages;
    public Dictionary<string, object> additionalParameters = new Dictionary<string, object>();

    [Serializable]
    public class Message
    {
      public string? role;
      public string? content;
    }
  }
  #endif

  [Serializable]
  public class ResponseBody
  {
    public Choise[]? choices;
    public Error? error;

    [Serializable]
    public class Choise
    {
      public Message? message;
    }

    [Serializable]
    public class Message
    {
      public string? content;
    }

    [Serializable]
    public class Error
    {
      public string? message;
    }
  }

  [Serializable]
  public class ResponseWrapper
  {
    public bool? isSuccess;
    public string? content;
    public Int32? statusCode;
    public string? errorMessage;
  }

  public class Client
  {
    private readonly string _apiKey;
    private readonly string _apiUrl;
    private readonly HttpClient _client = new HttpClient();

    public Client(string apiKey, string apiUrl)
    {
      _apiKey = apiKey;
      _apiUrl = apiUrl;
      _client.DefaultRequestHeaders.Add("Authorization", $"Bearer {_apiKey}");
    }

    public ResponseWrapper GenerateResponse(
      string model,
      string prompt,
      float? temperature,
      float? topP,
      Int32? topK,
      float? frequencyPenalty,
      float? presencePenalty,
      float? repetitionPenalty,
      float? minP,
      float? topA,
      Int32? seed,
      Int32? maxTokens
    )
    {
      #if UNITY
      var requestBody = new requestBody
      {
        model = model,
        messages = new[]
        {
          new requestBody.Message
          {
            role = "user",
            content = prompt
          }
        }
      };

      // Add additional parameters
      if (temperature.HasValue) requestBody.additionalParameters["temperature"] = temperature.Value;

      if (topP.HasValue) requestBody.additionalParameters["top_p"] = topP.Value;

      if (topK.HasValue) requestBody.additionalParameters["top_k"] = topK.Value;

      if (frequencyPenalty.HasValue) requestBody.additionalParameters["frequency_penalty"] = frequencyPenalty.Value;

      if (presencePenalty.HasValue) requestBody.additionalParameters["presence_penalty"] = presencePenalty.Value;

      if (repetitionPenalty.HasValue) requestBody.additionalParameters["repetition_penalty"] = repetitionPenalty.Value;

      if (minP.HasValue) requestBody.additionalParameters["min_p"] = minP.Value;

      if (topA.HasValue) requestBody.additionalParameters["top_a"] = topA.Value;

      if (seed.HasValue) requestBody.additionalParameters["seed"] = seed.Value;

      if (maxTokens.HasValue) requestBody.additionalParameters["max_tokens"] = maxTokens.Value;

      #else
      var messages = new List<Dictionary<string, string>>
      {
        new Dictionary<string, string>
        {
          {"role", "user"},
          {"content", prompt}
        }
      };

      var requestBody = new Dictionary<string, object>
      {
        {"model", model},
        {"messages", messages},
        { "temperature", temperature ?? null },
        { "top_p", topP ?? null },
        { "top_k", topK ?? null },
        { "frequency_penalty", frequencyPenalty ?? null },
        { "presence_penalty", presencePenalty ?? null },
        { "repetition_penalty", repetitionPenalty ?? null },
        { "min_p", minP ?? null },
        { "top_a", topA ?? null },
        { "seed", seed ?? null },
        { "max_tokens", maxTokens ?? null }
      };
      #endif

      #if UNITY
      var request = JsonUtility.ToJson(requestBody);
      var content = new StringContent(request, System.Text.Encoding.UTF8, "application/json");

      try
      {
        HttpResponseMessage response = _client.PostAsync(_apiUrl, content).Result;
        string responseBody = response.Content.ReadAsStringAsync().Result;
        var parsedResponse = JsonUtility.FromJson<ResponseBody>(responseBody);

        if (response.IsSuccessStatusCode)
        {
          var successResponseWrapper = new ResponseWrapper
          {
            isSuccess = true,
            content = parsedResponse.choices[0].message.content,
            statusCode = (int)response.StatusCode,
            errorMessage = (string)null
          };

          return successResponseWrapper;
        }
        else
        {
          string errorMessage = response.StatusCode == HttpStatusCode.Unauthorized
                                ? "Authorization Error: Invalid API Key"
                                : $"Error: {parsedResponse.error.message}";

          var errorResponseWrapper = new ResponseWrapper
          {
            isSuccess = false,
            content = (string)null,
            statusCode = (int)response.StatusCode,
            errorMessage = errorMessage
          };

          return errorResponseWrapper;
        }
      }
      catch (HttpRequestException e)
      {
        var exceptionResponseWrapper = new ResponseWrapper
        {
          isSuccess = false,
          content = (string)null,
          statusCode = (int)HttpStatusCode.InternalServerError,
          errorMessage = $"Request Error: {e.Message}"
        };

        return exceptionResponseWrapper;
      }

      #else
      var request = JsonSerializer.Serialize(requestBody);
      var content = new StringContent(request, System.Text.Encoding.UTF8, "application/json");

      try
      {
        HttpResponseMessage response = _client.PostAsync(_apiUrl, content).Result;
        string responseBody = response.Content.ReadAsStringAsync().Result;
        string? parsedResponse;

        if (response.IsSuccessStatusCode)
        {
          using (JsonDocument doc = JsonDocument.Parse(responseBody))
          {
            JsonElement root = doc.RootElement;
            JsonElement choices = root.GetProperty("choices");
            JsonElement firstChoice = choices[0];
            JsonElement message = firstChoice.GetProperty("message");
            JsonElement messageContent = message.GetProperty("content");
            parsedResponse = messageContent.ToString();
          };

          var successResponseWrapper = new ResponseWrapper
          {
            isSuccess = true,
            content = parsedResponse,
            statusCode = (int)response.StatusCode,
            errorMessage = (string)null
          };

          return successResponseWrapper;
        }
        else
        {
          using (JsonDocument doc = JsonDocument.Parse(responseBody))
          {
            JsonElement root = doc.RootElement;
            JsonElement error = root.GetProperty("error");
            JsonElement message = error.GetProperty("message");
            parsedResponse = message.ToString();
          };

          string errorMessage = response.StatusCode == HttpStatusCode.Unauthorized
                                ? "Authorization Error: Invalid API Key"
                                : $"Error: {parsedResponse}";

          var errorResponseWrapper = new ResponseWrapper
          {
            isSuccess = false,
            content = (string)null,
            statusCode = (int)response.StatusCode,
            errorMessage = errorMessage
          };

          return errorResponseWrapper;
        }
      }
      catch (HttpRequestException e)
      {
        var exceptionResponseWrapper = new ResponseWrapper
        {
          isSuccess = false,
          content = (string)null,
          statusCode = (int)HttpStatusCode.InternalServerError,
          errorMessage = $"Request Error: {e.Message}"
        };

        return exceptionResponseWrapper;
      }

      #endif

    }
  }
}