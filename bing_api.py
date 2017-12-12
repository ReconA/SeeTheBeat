# -*- coding: utf-8 -*-

import http.client, urllib.parse, json, urllib.request
import random
import ssl

# **********************************************
# *** Update or verify the following values. ***
# **********************************************

# Replace the subscriptionKey string value with your valid subscription key.
subscription_key = "ba96c1f65c8444bf95c4b7bf974ff94a"

# Verify the endpoint URI.  At this writing, only one endpoint is used for Bing
# search APIs.  In the future, regional endpoints may be available.  If you
# encounter unexpected authorization errors, double-check this value against
# the endpoint for your Bing search instance in your Azure dashboard.
host = "api.cognitive.microsoft.com"
path = "/bing/v7.0/images/search"


def bing_image_search(search):
    "Performs a Bing image search and returns the results."

    headers = {'Ocp-Apim-Subscription-Key': subscription_key}
    conn = http.client.HTTPSConnection(host)
    query = urllib.parse.quote(search)
    conn.request("GET", path + "?q=" + query, headers=headers)
    response = conn.getresponse()
    headers = [k + ": " + v for (k, v) in response.getheaders()
                   if k.startswith("BingAPIs-") or k.startswith("X-MSEdge-")]
    return headers, response.read().decode("utf8")


def get_image(term, file_name):
    print('Searching images for: ', term)

    if len(subscription_key) == 32:
        headers, result = bing_image_search(term)
        dic = json.loads(json.dumps(json.loads(result)))
        pic_results = dic['value']

        length = len(pic_results)

        tries = True
        while tries:
            try:
                tries = False
                x = random.randint(0, length-1)
                url = pic_results[x]['contentUrl']
                urllib.request.urlretrieve(url, file_name)
            # If a download fails, just try again.
            except urllib.error.HTTPError:
                tries = True 
            except urllib.error.URLError:
                tries = True 
            except ssl.CertificateError:
                tries = True
            
    else:
        print("Invalid Bing Search API subscription key!")
        print("Please paste yours into the source code.")

