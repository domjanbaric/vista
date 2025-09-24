import numpy as np
import spacy
import re
import asyncio
import httpx
from openai import APITimeoutError, APIConnectionError, InternalServerError, RateLimitError,AsyncOpenAI
from tenacity import (
    retry,
    wait_random_exponential,
    stop_after_attempt,
    retry_if_exception_type,
)
import json
import pandas as pd
import time
from tqdm.asyncio import tqdm_asyncio
import logging



nlp = spacy.load("en_core_web_sm")


# Create OpenAI async client with extended timeout
client = AsyncOpenAI(
    api_key=API_KEY,
    timeout=httpx.Timeout(60.0)  # Extend default timeout (10s → 60s)
)

# Limit concurrent API calls to avoid rate limits or overload
CONCURRENCY_LIMIT = 5
semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

# ─────────────────────────────────────────────
# ✅ get_embedding function with retry and semaphore
# ─────────────────────────────────────────────

@retry(
    wait=wait_random_exponential(min=1, max=30),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type((APITimeoutError, APIConnectionError, InternalServerError, RateLimitError)),
    retry_error_callback=lambda retry_state: None,
    reraise=False  # Don't raise on final failure
)
async def get_embedding(text: str) -> list[float] | None:
    """
    Gets an embedding from OpenAI for the provided text.
    Truncates text if it's too long. Retries on timeouts, rate limits, and server errors.
    Returns None on final failure.
    """
    try:
        if len(text)/2.5 > 8000:
            text_in = text[:11500]
        else:
            text_in = text

        async with semaphore:
            response = await client.embeddings.create(
                model="text-embedding-ada-002",
                input=text_in
            )
            return response.data[0].embedding

    except Exception as e:
        logging.warning(f"Embedding failed: {e}")
        return None
# ─────────────────────────────────────────────
# ✅ openai_response_async function with retry and semaphore
# ─────────────────────────────────────────────

@retry(
    wait=wait_random_exponential(min=1, max=30),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type((APITimeoutError, APIConnectionError, InternalServerError, RateLimitError)),
    retry_error_callback=lambda retry_state: None,
    reraise=False  # Don't raise on final failure
)
async def openai_response_async(prompt: str) -> str | None:
    """
    Sends a prompt to GPT-4o and returns the response.
    Retries on timeouts, rate limits, and server errors.
    Returns None on failure.
    """
    try:
        async with semaphore:
            completion = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
            )
            return completion.choices[0].message.content.strip()

    except Exception as e:
        logging.warning(f"Chat completion failed: {e}")
        return None


def norm(vector):#Normalizes vectors beetwen [-1,1]
    vector = np.array(vector)
    max = np.max(vector)
    min = np.min(vector)
    vector = 2 * (vector - min) / (max - min) - 1
    return list(vector)

def stopwords(text_list):#Removes stopwords from texts
    for i in range(len(text_list)):
        cleared_text = nlp(text_list[i])
        cleared_text = [token.text for token in cleared_text if not token.is_stop]
        cleared_text = ' '.join(cleared_text)
        text_list[i] = cleared_text
    return text_list
def nouns(text):
    x = nlp(text)
    x = [token.text for token in x if token.pos_=="NOUN"]
    x = ' '.join(x)
    text = x
    return text

def text_parse(text_list):
    combined_text = ''.join(text_list)
    combined_text = re.sub(r'[\r\n]+', '\n', combined_text)

    # Split the text into paragraphs based on double newlines
    paragraphs = combined_text.split('\n\n')

    # Remove extra whitespace and join paragraphs with double newlines
    formatted_text = '\n\n'.join(paragraph.strip() for paragraph in paragraphs)
    formatted_text = formatted_text.replace('\xa0', '\n')
    # formatted_text=formatted_text.replace('\xa0', ' ')
    formatted_text = re.sub(r'\n+', '\n', formatted_text)  # Mozda nam ovo triba ali moze bit i white space izmedu
    formatted_text = re.sub(r'\t+', '\t', formatted_text)  # Mozda nam ovo triba ali moze bit i white space izmedu

    return formatted_text

async def text_to_summary(text):
    prompt=f"""
    You will be given a text and your task is to make a FOUR summaries of that text and put &&& between them. the text will be
    given below divided by triple delimiters. Don't write anything but the summaries so NO introduction NO conclusion etc.
    '''{text}'''
    """
    summary= await openai_response_async(prompt)
    summary=summary.split('&&&')
    return summary
async def make_tweets(tweet):
    tweets=[]
    for i in range(3):
        prompt=f"""
        You will be given a text of a tweet and I want you to make a tweet with same meaning as this but different phrases
        and also try to make your tweet the similar size as the one I will give you.
        You will also be given a list of tweets called tweets, try and make yours differ from tweets in that list, if the list is empty just ignore it.
        '''{tweet}'''
        '''{tweets}'''
        """
        response=await openai_response_async(prompt)
        tweets.append(response)
    tweets.append(tweet)
    return tweets

async def text_to_tweet(text):
    prompt=f"""
    You are a social media manager and your job is to make given texts into tweets. You will be given a text and you need to make tweet out of it.
    follow the standard rules for making tweets, dont make them too long,put sometimes some emojis (you dont have to always put them)... 
    Also remember that you work for a big company so the tweets need to be professional not too much childish.
    '''{text}'''
    """
    tweet=await openai_response_async(prompt)
    return tweet
async def process_one(text,id):
    try:
        text=text_parse(text)
        summaries = await text_to_summary(text)
        text_nouns = nouns(text)
        summaries_stop=stopwords(summaries)
        summaries_nouns=[nouns(summary) for summary in summaries]
        summ_stop_vec = await asyncio.gather(*[get_embedding(text) for text in summaries_stop])
        summ_noun_vec= await asyncio.gather(*[get_embedding(text) for text in summaries_nouns])
        summ_stop_vec = [norm(vec) for vec in summ_stop_vec]
        summ_noun_vec = [norm(vec) for vec in summ_noun_vec]
        text_vec = norm(await get_embedding(text_nouns))
        tweet=await text_to_tweet(text)
        tweet_vec = norm(await get_embedding(tweet))
        return {'id':id,'text':text,'summaries':summaries,'text vector':text_vec,'summary stop vector':summ_stop_vec,'summary noun vector':summ_noun_vec,'is tweet':False,'tweet vector':tweet_vec}
    except:
        return {}
async def process_tweet(tweet,id):
    try:
        tweets=await make_tweets(tweet)
        tweets_vec=await asyncio.gather(*[get_embedding(text) for text in tweets])
        tweets_vec=[norm(vec) for vec in tweets_vec]
        return {'id':id,'text':tweet,'tweet vector 1':tweets_vec[0],'tweet vector 2':tweets_vec[1],'tweet vector 3':tweets_vec[2],'tweet vector 4':tweets_vec[3],'is tweet':True}
    except:
        return {}
async def process_list(file_path,text_list,tweet_list,last_id=0):
    results= await asyncio.gather(*[process_one(text_list[index],last_id+1+index) for index in range(len(text_list))])
    print(len(results))
    results=[res for res in results if res!={}]
    print(len(results))
    results_tweet=await asyncio.gather(*[process_tweet(tweet_list[index],len(text_list)+1+index) for index in range(len(tweet_list))])
    print(len(results_tweet))
    results_tweet = [res for res in results_tweet if res != {}]
    print(len(results_tweet))
    results.extend(results_tweet)
    with open(file_path,'w') as f:
        json.dump(results,f,ensure_ascii=False,indent=4)
    a = [index for index in range(len(results)) if results[index]=={}]
    print(a)
    return




# ------------------------------ CLI entrypoint -------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="VISTA preprocessing: turn raw texts + tweets into JSON with summaries, embeddings, and tweet variants."
    )
    parser.add_argument(
        "--texts",
        type=str,
        required=True,
        help="Path to JSON file containing a list of texts (each text may be a list of fragments).",
    )
    parser.add_argument(
        "--tweets",
        type=str,
        required=True,
        help="Path to JSON file containing a list of tweet strings.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output JSON file where results will be stored.",
    )
    parser.add_argument(
        "--last-id",
        type=int,
        default=0,
        help="Starting offset for IDs (default: 0).",
    )

    args = parser.parse_args()

    # Load inputs
    with open(args.texts, "r", encoding="utf-8") as f:
        text_list = json.load(f)
    with open(args.tweets, "r", encoding="utf-8") as f:
        tweet_list = json.load(f)

    # Run pipeline
    asyncio.run(
        process_list(
            file_path=args.output,
            text_list=text_list,
            tweet_list=tweet_list,
            last_id=args.last_id,
        )
    )
    print(f"[VISTA] Preprocessing complete → {args.output}")