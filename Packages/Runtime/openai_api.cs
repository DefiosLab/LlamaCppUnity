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

namespace OpenAI
{
  [Serializable]
  public class requestBody
  {
    public string? model;
    public Message[]? messages;
    public Dictionary<string, object> additional_parameters = new Dictionary<string, object>();

    [Serializable]
    public class Message
    {
      public string? role;
      public string? content;
    }
  }

  [Serializable]
  public class responseBody
  {
    public Choise[]? choices;

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
    private const string _apiUrl = "https://openrouter.ai/api/v1/chat/completions";
    private readonly HttpClient _client = new HttpClient();

    public Client(string apiKey)
    {
      _apiKey = apiKey;
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

      // Add optional parameters
      if (temperature.HasValue) requestBody.additional_parameters["temperature"] = temperature.Value;

      if (topP.HasValue) requestBody.additional_parameters["top_p"] = topP.Value;

      if (topK.HasValue) requestBody.additional_parameters["top_k"] = topK.Value;

      if (frequencyPenalty.HasValue) requestBody.additional_parameters["frequency_penalty"] = frequencyPenalty.Value;

      if (presencePenalty.HasValue) requestBody.additional_parameters["presence_penalty"] = presencePenalty.Value;

      if (repetitionPenalty.HasValue) requestBody.additional_parameters["repetition_penalty"] = repetitionPenalty.Value;

      if (minP.HasValue) requestBody.additional_parameters["min_p"] = minP.Value;

      if (topA.HasValue) requestBody.additional_parameters["top_a"] = topA.Value;

      if (seed.HasValue) requestBody.additional_parameters["seed"] = seed.Value;

      if (maxTokens.HasValue) requestBody.additional_parameters["max_tokens"] = maxTokens.Value;

      #if UNITY
      var request = JsonUtility.ToJson(requestBody);
      var content = new StringContent(request, System.Text.Encoding.UTF8, "application/json");

      try
      {
        HttpResponseMessage response = _client.PostAsync(_apiUrl, content).Result;
        string responseBody = response.Content.ReadAsStringAsync().Result;
        var parsedResponse = JsonUtility.FromJson<responseBody>(responseBody);

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