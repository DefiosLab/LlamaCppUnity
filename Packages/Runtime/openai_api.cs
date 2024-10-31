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
  #if UNITY
  public class RequestBody
  {
    public string model = "";
    public Message[] messages = new Message[0];
    public float temperature = 1.0f;
    public float top_p = 1.0f;
    public float frequency_penalty = 0.0f;
    public float presence_penalty = 0.0f;
    public int seed = -1;
    public int max_tokens = -1;

    [Serializable]
    public class Message
    {
      public string role;
      public string content;
    }

    public static RequestBody Create(
      string model,
      string prompt,
      float? temperature,
      float? topP,
      float? frequencyPenalty,
      float? presencePenalty,
      int? seed,
      int? maxTokens
    )
    {
      var body = new RequestBody
      {
        model = model,
        messages = new[] {
          new Message {
            role = "user",
            content = prompt
          }
        }
      };

      if (temperature.HasValue) body.temperature = temperature.Value;

      if (topP.HasValue) body.top_p = topP.Value;

      if (frequencyPenalty.HasValue) body.frequency_penalty = frequencyPenalty.Value;

      if (presencePenalty.HasValue) body.presence_penalty = presencePenalty.Value;

      if (seed.HasValue) body.seed = seed.Value;

      if (maxTokens.HasValue) body.max_tokens = maxTokens.Value;

      return body;
    }

    public RequestBody PrepareForSerialization()
    {
      var cleaned = new RequestBody
      {
        model = this.model,
        messages = this.messages,
        temperature = this.temperature,
        top_p = this.top_p,
        frequency_penalty = this.frequency_penalty,
        presence_penalty = this.presence_penalty
      };

      if (this.seed != -1) cleaned.seed = this.seed;

      if (this.max_tokens != -1) cleaned.max_tokens = this.max_tokens;

      return cleaned;
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
      float? frequencyPenalty,
      float? presencePenalty,
      Int32? seed,
      Int32? maxTokens
    )
    {
      #if UNITY
      var body = RequestBody.Create(
                   model,
                   prompt,
                   temperature,
                   topP,
                   frequencyPenalty,
                   presencePenalty,
                   seed,
                   maxTokens
                 );

      var requestBody = body.PrepareForSerialization();

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
        { "frequency_penalty", frequencyPenalty ?? null },
        { "presence_penalty", presencePenalty ?? null },
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

        // Console.WriteLine(responseBody);
        Debug.Log(responseBody);

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