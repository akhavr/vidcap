# Uses this https://github.com/python-telegram-bot/python-telegram-bot

from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import logging


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class Notifier:
    def __init__(self, token):
        self.tg_chat_id = 334424084
        self.tg_context = None

        self.updater = Updater(token, use_context=True)
        # Get the dispatcher to register handlers
        dp = self.updater.dispatcher
        # on different commands - answer in Telegram
        dp.add_handler(CommandHandler("start", lambda up, con: self.start(up, con)))
        # log all errors
        dp.add_error_handler(lambda: self.error)
        # Start the Bot
        self.updater.start_polling()
        return

    def start(self, update, context):
        """Send a message when the command /start is issued."""
        assert update.message.chat_id == self.tg_chat_id  # make sure only I can connect now
        self.tg_chat_id = update.message.chat_id
        self.tg_context = context
        update.message.reply_text('Hi! {}'.format(self.tg_chat_id))

    def error(self, update, context):
        """Log Errors caused by Updates."""
        logger.warning('Update "%s" caused error "%s"', update, context.error)

    def notify(self, msg):
        print('Chat id {}'.format(self.tg_chat_id))
        if not (self.tg_chat_id and self.tg_context):
            return
        print(msg)
        self.tg_context.bot.send_message(self.tg_chat_id, msg)


