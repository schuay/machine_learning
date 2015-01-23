from datetime import datetime

CREATED_AT = "created_at"
ID = "id"
TEXT = "text"
RETWEETED_STATUS = "retweeted_status"
RETWEET_COUNT = "retweet_count"

"""This function converts date fields we care
about (such as "created_at") into proper datetime objects."""
def to_date(tweet):
    if CREATED_AT not in tweet:
        tweet[CREATED_AT] = datetime.now()
        return tweet
    if type(tweet[CREATED_AT]) is datetime:
        return tweet
    # strptime does not always accept %z for the timezone in 2.7, so we have
    # to handle it manually. The initial format is: Sun Oct 20 19:48:26 +0000 2013
    datestr = tweet[CREATED_AT]
    dt = datetime.strptime(datestr[0:20] + datestr[26:], "%a %b %d %H:%M:%S %Y")
    tweet[CREATED_AT] = dt
    return tweet

def to_ascii(tweet):
    """Converts tweet text to ascii, replacing invalid chars with '?'."""
    tweet[TEXT] = tweet[TEXT].encode('ascii', 'replace')
    return tweet
