from typing import Union
from pathlib import Path
import json 
from hazm import word_tokenize, Normalizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import arabic_reshaper
from bidi.algorithm import get_display
from loguru import logger

Data_Dir = Path("../data").resolve()


class ChatStatisics:
    def __init__(self, chat_json: Union[str, Path]):

        #load chat data
        logger.info(f"Loading chat data from {chat_json}")
        with open(Path(chat_json)) as f:
            self.chat_data = json.load(f)

        self.Normalizer = Normalizer()
        logger.info("Loading stopwords")
        #load stopwords
        Stopwords = open(Data_Dir / 'Stopwords.txt').read().splitlines()
        self.Stopwords = list(map(self.Normalizer.normalize, Stopwords))

    def generatr_word_cloud(self, output_dir: Union[str, Path]):
        logger.info("Generating word cloud")
        text = ''
        for msg in self.chat_data['messages']:
            if type(msg['text']) is str:
                tokens = word_tokenize(msg['text'])
                tokens = list(filter(lambda x: x not in self.Stopwords, tokens))
                text += f"{' '.join(tokens)}"

        #normalize and reshape text
        logger.info("Normalizing and reshaping text")
        text = arabic_reshaper.reshape(text)
        text = get_display(text)
        text = self.Normalizer.normalize(text)
        text = arabic_reshaper.reshape(text)
        text = get_display(text)

        #generate wordcloud
        wordcloud = WordCloud(width=800, height=800,
                              font_path=str(Data_Dir / 'arial.ttf'), 
                              background_color='white'
                              ).generate(text)

        logger.info(f"Saving word cloud to {output_dir}")
        wordcloud.to_file(str(Path(output_dir) / 'wordcloud.png'))

if __name__ == "__main__":
    chat_stats = ChatStatisics(chat_json=Data_Dir / "Hamedless.json")
    chat_stats.generatr_word_cloud(output_dir=Data_Dir)
    print("Done!")
