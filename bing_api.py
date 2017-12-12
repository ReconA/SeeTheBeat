# -*- coding: utf-8 -*-

import http.client, urllib.parse, json, urllib.request
import pprint
import random


# **********************************************
# *** Update or verify the following values. ***
# **********************************************

# Replace the subscriptionKey string value with your valid subscription key.
subscriptionKey = "ba96c1f65c8444bf95c4b7bf974ff94a"

# Verify the endpoint URI.  At this writing, only one endpoint is used for Bing
# search APIs.  In the future, regional endpoints may be available.  If you
# encounter unexpected authorization errors, double-check this value against
# the endpoint for your Bing search instance in your Azure dashboard.
host = "api.cognitive.microsoft.com"
path = "/bing/v7.0/images/search"


def BingImageSearch(search):
    "Performs a Bing image search and returns the results."

    headers = {'Ocp-Apim-Subscription-Key': subscriptionKey}
    conn = http.client.HTTPSConnection(host)
    query = urllib.parse.quote(search)
    conn.request("GET", path + "?q=" + query, headers=headers)
    response = conn.getresponse()
    headers = [k + ": " + v for (k, v) in response.getheaders()
                   if k.startswith("BingAPIs-") or k.startswith("X-MSEdge-")]
    return headers, response.read().decode("utf8")

def getImage(term, file_name):
    print('Searching images for: ', term)

    if len(subscriptionKey) == 32:
        headers, result = BingImageSearch(term)
        #print("\nRelevant HTTP Headers:\n")
        #print("\n".join(headers))
        dic = json.loads(json.dumps(json.loads(result)))
        pic_results = dic['value']

        # select a 
        length = len(pic_results)
        x = random.randint(0, length)
        url = pic_results[x]['contentUrl']
        urllib.request.urlretrieve(url, file_name)

    else:
        print("Invalid Bing Search API subscription key!")
        print("Please paste yours into the source code.")

