import os
import logging
from datetime import datetime

from matplotlib.pyplot import imread, imsave
import numpy as np
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from pickler import get_full_user_data, save_user_data, save_all, load_all
from json import load, dump

BAD_EMOTION_THRESHOLD = 4
TOKEN = "868031376:AAGdASZ8GANAa3L5nyHZEZiHX4Yais8_DCg"
os.environ['TOKEN'] = TOKEN
updater = Updater(token=os.environ.get('TOKEN'), use_context=True)
dispatcher = updater.dispatcher
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)


def is_everything_bad(emotions):
    for emotion in emotions[-BAD_EMOTION_THRESHOLD:]:
        if emotion >= 0:
            return False
    return True

def gif(update,context):
    #context.bot.send_document(chat_id=update.effective_chat.id, document=open(f'giphy.gif', 'rb'))
    context.bot.send_document(chat_id=update.effective_chat.id, document=open(f'jendas1.gif', 'rb'))

if __name__ == '__main__':
    gif_handler = CommandHandler('gif', gif)
    dispatcher.add_handler(gif_handler)
    updater.start_polling()
