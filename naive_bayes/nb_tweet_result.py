
class NBTweetResult:
    def __init__(self, tweet_id, pred, score, real_value):
        self.tweet_id = tweet_id
        self.prediction = pred
        self.score = score
        self.real_value = real_value
        self.is_correct = "correct" if (pred == real_value) else "wrong"
