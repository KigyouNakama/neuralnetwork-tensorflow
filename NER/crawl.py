#Import the necessary methods from tweepy library
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream

#Variables that contains the user credentials to access Twitter API
access_token = "2403042456-d7qNuHA8eMQb8PMDlOfRbrvsUugAJB3gSkBxKbM"
access_token_secret = "EnCLykTVKX1JkghyEpWlbh1rPC1jO2btF4DYr2FIZyasZ"
consumer_key = "r3jnXsNHiGUPgnR391EoR5pbu"
consumer_secret = "M3vI0XNfudYsv0VF0QTFuvS4srk7Pu9PLOkYAY3dK0Jx0PpS6u"


#This is a basic listener that just prints received tweets to stdout.
class StdOutListener(StreamListener):

    def on_data(self, data):
        print (data)
        return True

    def on_error(self, status):
        print ("error %d"%status)


if __name__ == '__main__':

    #This handles Twitter authetification and the connection to Twitter Streaming API
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, l)

    #This line filter Twitter Streams to capture data by the keywords: 'python', 'javascript', 'ruby'
    linh = stream.filter(track='syria')
    print("type %s"%type(linh))
    print("length %d"%len(linh))

