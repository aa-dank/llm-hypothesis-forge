import os
import openai
import json
import math
import logging
from dotenv import load_dotenv
from google import genai
from google.genai import types
from huggingface_hub import InferenceClient
from typing import Dict, Any, List, Optional, Union

TOP_LOGPROBS = 5

logger = logging.getLogger(__name__)

class BasicOpenAI:
    """
    A wrapper for the OpenAI API that provides simplified access to ChatCompletions.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-2024-08-06"):
        """
        Initialize the OpenAI client.
        
        Args:
            api_key: OpenAI API key (optional, will use environment variable if not provided)
            model: The model to use for completions
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        logger.info(f"Initialized BasicOpenAI with model: {model}")
        
    def complete(self, 
                prompt: str, 
                system_message: str = "You are a helpful assistant.", 
                temperature: float = 0.0,
                max_tokens: int = 800,
                json_response: bool = False) -> Union[str, Dict]:
        """
        Generate a completion for the given prompt.
        
        Args:
            prompt: The user prompt
            system_message: The system message to use
            temperature: Controls randomness (0.0 = deterministic)
            max_tokens: Maximum number of tokens to generate
            json_response: Whether to parse the response as JSON
            
        Returns:
            The completion text or parsed JSON object
        """
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        
        response_format = {"type": "json_object"} if json_response else None
        
        try:
            logger.debug(f"Calling OpenAI API with model {self.model}, temperature={temperature}, max_tokens={max_tokens}")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format
            )
            
            # Extract the completion text which is in the first choice
            result = response.choices[0].message.content
            
            # Parse JSON if requested
            if json_response:
                logger.debug("Parsing JSON response from OpenAI")
                return json.loads(result)
            return result
            
        except Exception as e:
            logger.error(f"Error in OpenAI API call: {e}")
            raise

    def output_perplexity_score(self, prompt: str, top_logprobs: int = TOP_LOGPROBS) -> float:
        """
        Compute the perplexity of a given prompt using OpenAI's Chat Completions API.

        Args:
            prompt: The text prompt to evaluate

        Returns:
            Perplexity score as a float
        """
        try:
            logger.debug(f"Calculating perplexity for text of length {len(prompt)} characters")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1,
                temperature=0,
                logprobs=True,
                top_logprobs=top_logprobs,
            )

            # Extract the log probabilities of the generated tokens
            logprobs = response.choices[0].logprobs.content

            # Calculate the average negative log likelihood
            avg_nll = -sum(token.logprob for token in logprobs) / len(logprobs)

            # Compute perplexity
            perplexity = math.exp(avg_nll)
            
            logger.info(f"Calculated perplexity score: {perplexity}")
            return perplexity

        except Exception as e:
            logger.error(f"Error calculating perplexity: {e}")
            raise


class BasicGoogleGemini:
    """
    A wrapper for the Google Gemini API using the google-genai SDK.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-flash-001"):
        """
        Initialize the Google Gemini client.

        Args:
            api_key: Google API key (optional; uses env var GOOGLE_API_KEY if not provided)
            model: The model to use for generation (e.g., "gemini-1.5-flash-001")
        """
        load_dotenv()  # Load .env

        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key is required. "
                "Set GOOGLE_API_KEY environment variable or pass it directly."
            )

        self.client = genai.Client(api_key=self.api_key,
                                   http_options=types.HttpOptions(api_version="v1"))
        self.model_name = model
        logger.info(f"Initialized BasicGoogleGemini with model: {model}")

    def complete(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 800,
        json_response: bool = False
    ) -> Union[str, Dict]:
        """
        Generate content for the given prompt using Google Gemini.

        Args:
            prompt: The user prompt
            system_message: Optional system instruction
            temperature: Controls randomness (0.0 = deterministic)
            max_tokens: Max tokens to generate
            json_response: Whether to parse the output as JSON

        Returns:
            The generated text or parsed JSON object
        """
        # Build contents as a single string or list of strings
        if system_message:
            contents: Union[str, List[str]] = [system_message, prompt]
        else:
            contents = prompt

        # Configure generation parameters
        config: Dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_tokens
        }

        try:
            logger.debug(
                f"Calling Google Gemini API: model={self.model_name}, "
                f"temp={temperature}, max_tokens={max_tokens}"
            )
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config
            )
            result = response.text  # shorthand for the top candidate's content

            if json_response:
                logger.debug("Parsing JSON response from Gemini")
                # strip code fences if any
                cleaned = result.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
                return json.loads(cleaned)

            return result

        except Exception as e:
            logger.error(f"Error in Google Gemini API call: {e}")
            raise

    def output_perplexity_score(self, prompt: str) -> float:
        """
        Compute perplexity of a given prompt by requesting one-token continuation
        and retrieving log-probabilities from the response.

        Args:
            prompt: The text prompt to evaluate

        Returns:
            Perplexity score as a float
        """
        # Request exactly one token and ask for logprobs
        config: Dict[str, Any] = {
            "temperature": 0.0,
            "max_output_tokens": 1,
            "response_logprobs": True,
            "logprobs": 5  # retrieve up to 10 top logprobs per token
        }

        try:
            logger.debug(f"Calculating perplexity for input length {len(prompt)} characters")
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config
            )

            # Extract the logprob results for the chosen token(s)
            candidate = response.candidates[0]
            logres = candidate.logprobs_result  # LogprobsResult object
            # use the chosen_candidates list for the actual generated token(s)
            log_probs = [tok.log_probability for tok in logres.chosen_candidates]
            avg_nll = -sum(log_probs) / len(log_probs)
            perplexity = math.exp(avg_nll)

            logger.info(f"Calculated perplexity score: {perplexity}")
            return perplexity

        except Exception as e:
            logger.error(f"Error calculating perplexity: {e}")
            raise


class TogetherAIxHuggingFace:
    """
    A wrapper for Together.ai models via Hugging Face InferenceClient, for calculating perplexity.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "deepseek-ai/DeepSeek-R1"):
        """
        Initialize the Hugging Face Together client.

        Args:
            api_key: Hugging Face API key (must have Together access)
            model: The Together-hosted model to use
        """
        if api_key is None:
            api_key = os.getenv("HUGGINGFACE_API_KEY")
        if not api_key:
            raise ValueError("HUGGINGFACE_API_KEY not set")

        self.client = InferenceClient(provider="together", api_key=api_key)
        self.model = model
        logger.info(f"Initialized TogetherAIxHuggingFace with model: {model}")

    def perplexity_score(self, prompt: str) -> float:
        """
        Calculate perplexity for the input prompt using Together.ai model via Hugging Face.

        Args:
            prompt: The input string to score

        Returns:
            Perplexity score as float
        """
        try:
            logger.debug("Sending request with echo=True and logprobs=1")
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=0,
                temperature=0,
                extra_body={
                    "logprobs": 1,
                    "echo": True
                }
            )

            completion_values = list(completion.values())
            prompt_logprobs = None

            for val in completion_values:
                if isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
                    if "logprobs" in val[0] and "tokens" in val[0]["logprobs"]:
                        tokens = val[0]["logprobs"]["tokens"]
                        logprobs = val[0]["logprobs"]["token_logprobs"]
                        logger.debug(f"Prompt tokens: {tokens}")
                        logger.debug(f"Token logprobs: {logprobs}")
                        prompt_logprobs = [lp for lp in logprobs if lp is not None]
                        break

            if not prompt_logprobs:
                raise ValueError("No logprobs found for prompt tokens.")

            avg_neg_logprob = -sum(prompt_logprobs) / len(prompt_logprobs)
            perplexity = math.exp(avg_neg_logprob)
            logger.info(f"Perplexity for prompt: {perplexity:.4f}")
            return perplexity

        except Exception as e:
            logger.error(f"Error during perplexity calculation: {e}")
            raise


class TogetherClient:
    """
    A wrapper for Together AI using the OpenAI-compatible API to compute perplexity scores.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        """
        Initialize the Together client through the OpenAI API interface.

        Args:
            api_key: Together AI API key (optional, will use env var if not provided)
            model: The Together model to use (OpenAI-compatible string)
        """
        self.api_key = api_key or os.getenv("TOGETHER_API_KEY")
        if not self.api_key:
            raise ValueError("TOGETHER_API_KEY must be set in the environment or passed explicitly.")

        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://api.together.xyz/v1"
        )
        self.model = model
        logger.info(f"Initialized TogetherClient with model: {model}")

    def perplexity_score(self, prompt: str) -> float:
        """
        Calculate perplexity for a given prompt by retrieving token logprobs from Together AI.

        Args:
            prompt: The input prompt string

        Returns:
            Perplexity score as a float
        """
        try:
            logger.debug("Requesting logprobs with echo=True from Together AI via OpenAI client")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=0,
                temperature=0,
                extra_body={
                    "echo": True,
                    "logprobs": 1
                }
            )

            # Extract logprobs from the response
            if hasattr(response, 'choices') and response.choices:
                choice = response.choices[0]
                if hasattr(choice, 'logprobs') and choice.logprobs:
                    token_logprobs = choice.logprobs.token_logprobs

                    # Filter out None values
                    valid_logprobs = [lp for lp in token_logprobs if lp is not None]

                    if not valid_logprobs:
                        raise ValueError("No valid token logprobs returned.")

                    avg_neg_logprob = -sum(valid_logprobs) / len(valid_logprobs)
                    perplexity = math.exp(avg_neg_logprob)
                    logger.info(f"Perplexity: {perplexity:.4f}")
                    return perplexity

            raise ValueError("Logprobs missing in Together response.")

        except Exception as e:
            logger.error(f"Error during perplexity calculation: {e}")
            raise