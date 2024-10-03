
import rrnlp.models.SearchBot as SearchBot
search_bot = SearchBot.PubmedQueryGeneratorBot(device='cpu')
topic = search_bot.generate_review_topic('statins on heart health')
print(topic)

