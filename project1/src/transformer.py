import re

import tweet

class TweetTransformerI:
    def transform(self, obj):
        """Takes a tweet object and returns a transformed version."""
        raise NotImplementedError("Please implement this yourself.")

class IdentityTransformer(TweetTransformerI):
    def transform(self, obj):
        return obj

# TODO: This regex is a bit overeager and transforms strings such as '....' to TOKEN_URL.
class UrlTransformer(TweetTransformerI):
    """This transformer replaces any url within tweet texts with
       'TOKEN_URL'. Regex taken from http://net.tutsplus.com/tutorials/other/8-regular-expressions-you-should-know/."""
    def __init__(self):
        self.__pattern = re.compile('(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w\.-]*)*\/?')

    def transform(self, obj):
        obj[tweet.TEXT] = self.__pattern.sub('TOKEN_URL', obj[tweet.TEXT])
        return obj

class UserTransformer(TweetTransformerI):
    """This transformer replaces any twitter username within tweet texts with
       'TOKEN_USER'. Regex taken from http://shahmirj.com/blog/extracting-twitter-usertags-using-regex."""
    def __init__(self):
        self.__pattern = re.compile('(?<=^|(?<=[^a-zA-Z0-9-_\\.]))@([A-Za-z]+[A-Za-z0-9_]+)')

    def transform(self, obj):
        obj[tweet.TEXT] = self.__pattern.sub('TOKEN_USER', obj[tweet.TEXT])
        return obj

class MulticharTransformer(TweetTransformerI):
    """This transformer replaces chars repeated more than two times with exactly
       two occurrences of the character."""
    def __init__(self):
        self.__pattern = re.compile(r'([a-zA-Z0-9!?])\1{2,}')

    def transform(self, obj):
        obj[tweet.TEXT] = self.__pattern.sub(r'\1\1', obj[tweet.TEXT])
        return obj

class SequenceTransformer(TweetTransformerI):
    """This transformer runs a list of transformers in sequence."""
    def __init__(self, transformers):
        self.__transformers = transformers

    def transform(self, obj):
        o = obj;
        for p in self.__transformers:
            o = p.transform(o)
        return o
